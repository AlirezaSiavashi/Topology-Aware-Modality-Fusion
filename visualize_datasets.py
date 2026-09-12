"""
visualize_datasets.py
=====================
Visualize 2-3 samples from each preprocessed dataset.

For each sample shows:
  Row 1 – Image: axial / coronal / sagittal centre slices
  Row 2 – Label overlay (colour-coded classes) on same slices
  Row 3 – Auxiliary maps: vessel-union | SDF | skeleton+curvature
           (skipped for unlabelled / cross-domain sets)

Output: one PNG per dataset saved to  visualizations/<DatasetName>/
        plus an overview mosaic  visualizations/overview_mosaic.png

Usage
-----
  # activate the venv first
  source /scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/nnunet_venv/bin/activate

  # run from brain_external_dataset/
  python visualize_datasets.py

  # only specific datasets
  python visualize_datasets.py --datasets 100 101 110

  # change number of samples
  python visualize_datasets.py --n 3
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")          # must be before pyplot import (no display)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import SimpleITK as sitk

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent
RAW_ROOT    = BASE / "preprocessed" / "nnUNet_raw"
AUX_ROOT    = BASE / "preprocessed" / "aux"
VIS_ROOT    = BASE / "visualizations"

# Dataset metadata: id → (dir_suffix, aux_name_or_None)
DATASETS = {
    100: ("Dataset100_TopBrain_CT",            "TopBrain_CT"),
    101: ("Dataset101_TopBrain_MR",            "TopBrain_MR"),
    102: ("Dataset102_TopCoW_CT",              "TopCoW_CT"),
    103: ("Dataset103_TopCoW_MR",              "TopCoW_MR"),
    104: ("Dataset104_TopCoW_joint",           None),   # no separate aux (same subjs as 102/103)
    110: ("Dataset110_MSD_HepaticVessel",      "MSD_HepaticVessel"),
    200: ("Dataset200_CrossDomain_CTA_ISLES2024", None),
    201: ("Dataset201_CrossDomain_MRA_IXI_HH",   None),
    202: ("Dataset202_CrossDomain_MRA_Lausanne",  None),
    210: ("Dataset210_ITKTubeTK_MRA",          None),
}

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette for up to 43 classes  (0 = transparent background)
# ─────────────────────────────────────────────────────────────────────────────
def _build_label_cmap(n_classes: int):
    """Return a ListedColormap where index 0 is fully transparent."""
    base = plt.cm.get_cmap("tab20", max(n_classes, 2))
    colors = [(0, 0, 0, 0)]                           # class 0 → transparent
    for i in range(1, n_classes):
        r, g, b, _ = base(i % 20)
        colors.append((r, g, b, 0.65))                # semi-transparent overlay
    return mcolors.ListedColormap(colors)


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_vol(path: Path) -> np.ndarray:
    """Load NIfTI → numpy float32, shape (Z, Y, X)."""
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    return arr


def centre_slices(vol: np.ndarray):
    """Return (axial, coronal, sagittal) centre slices."""
    z, y, x = vol.shape
    return vol[z // 2], vol[:, y // 2, :], vol[:, :, x // 2]


def norm01(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample figure
# ─────────────────────────────────────────────────────────────────────────────
PLANE_NAMES = ["Axial", "Coronal", "Sagittal"]


def plot_sample(img_path: Path, lbl_path: Optional[Path], aux_base: Optional[Path],
                subject_id: str, label_names: dict, out_path: Path) -> None:
    """Create and save the per-sample figure."""

    img_vol  = load_vol(img_path)
    lbl_vol  = load_vol(lbl_path).astype(np.int32) if (lbl_path and lbl_path.exists()) else None

    # ── load aux maps ──────────────────────────────────────────────────────
    has_aux = False
    vu_vol = sdf_vol = skel_vol = curv_vol = None
    if aux_base:
        vu_p   = aux_base / "vessel_union"  / f"{subject_id}.nii.gz"
        sdf_p  = aux_base / "sdf"           / f"{subject_id}.nii.gz"
        skel_p = aux_base / "skeleton"      / f"{subject_id}.nii.gz"
        curv_p = aux_base / "curvature"     / f"{subject_id}.nii.gz"
        if vu_p.exists():
            vu_vol   = load_vol(vu_p).astype(np.int32)
            has_aux  = True
        if sdf_p.exists():
            sdf_vol  = load_vol(sdf_p)
        if skel_p.exists():
            skel_vol = load_vol(skel_p).astype(np.int32)
        if curv_p.exists():
            curv_vol = load_vol(curv_p)

    n_rows = 2 + (1 if has_aux else 0)
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4, n_rows * 3.6),
                             facecolor="#1a1a1a")
    fig.suptitle(f"{subject_id}", color="white", fontsize=13, y=1.01)

    n_cls  = max(label_names.keys()) + 1 if label_names else 2
    lcmap  = _build_label_cmap(n_cls)

    def _show(ax, slice2d, title, cmap="gray", vmin=None, vmax=None,
              overlay=None, overlay_cmap=None, overlay_vmin=0, overlay_vmax=None):
        ax.set_facecolor("#1a1a1a")
        ax.imshow(slice2d, cmap=cmap, vmin=vmin, vmax=vmax,
                  origin="lower", aspect="equal", interpolation="nearest")
        if overlay is not None:
            ov = ax.imshow(overlay, cmap=overlay_cmap,
                           vmin=overlay_vmin, vmax=overlay_vmax,
                           origin="lower", aspect="equal", interpolation="nearest",
                           alpha=0.7)
        ax.set_title(title, color="white", fontsize=8, pad=3)
        ax.axis("off")

    # ── Row 0: image ──────────────────────────────────────────────────────
    img_slices = centre_slices(img_vol)
    p1, p99 = np.percentile(img_vol, 1), np.percentile(img_vol, 99)
    for col, (sl, name) in enumerate(zip(img_slices, PLANE_NAMES)):
        _show(axes[0, col], sl, name, cmap="gray", vmin=p1, vmax=p99)
    axes[0, 0].set_ylabel("Image", color="white", fontsize=9, labelpad=6)

    # ── Row 1: label overlay ───────────────────────────────────────────────
    if lbl_vol is not None:
        lbl_slices = centre_slices(lbl_vol)
        for col, (isl, lsl, name) in enumerate(zip(img_slices, lbl_slices, PLANE_NAMES)):
            _show(axes[1, col], isl, name, cmap="gray", vmin=p1, vmax=p99,
                  overlay=lsl, overlay_cmap=lcmap,
                  overlay_vmin=0, overlay_vmax=n_cls - 1)
        axes[1, 0].set_ylabel("Labels", color="white", fontsize=9, labelpad=6)

        # legend (up to 20 classes shown)
        # label_names: {class_id(int) -> class_name(str)}
        present = sorted(int(c) for c in set(lbl_vol.flat) if int(c) > 0)[:20]
        patches = []
        for cls_id in present:
            r, g, b, _ = lcmap(cls_id)
            patches.append(mpatches.Patch(color=(r, g, b),
                                          label=label_names.get(cls_id, str(cls_id))))
        if patches:
            axes[1, 2].legend(handles=patches, loc="upper right",
                              fontsize=5.5, framealpha=0.3,
                              labelcolor="white", facecolor="#333333",
                              ncol=2 if len(patches) > 10 else 1)
    else:
        for col in range(3):
            axes[1, col].text(0.5, 0.5, "no label", color="gray",
                              ha="center", va="center", transform=axes[1, col].transAxes)
            axes[1, col].set_facecolor("#1a1a1a")
            axes[1, col].axis("off")
        axes[1, 0].set_ylabel("Labels", color="white", fontsize=9, labelpad=6)

    # ── Row 2: auxiliary maps ──────────────────────────────────────────────
    if has_aux:
        ax_row = axes[2]
        axes[2, 0].set_ylabel("Aux maps", color="white", fontsize=9, labelpad=6)

        # col 0: vessel union (binary)
        if vu_vol is not None:
            vu_sl = centre_slices(vu_vol)[0]   # axial
            _show(ax_row[0], vu_sl, "Vessel union (axial)",
                  cmap="hot", vmin=0, vmax=1)
        else:
            ax_row[0].axis("off")

        # col 1: SDF (axial)
        if sdf_vol is not None:
            sdf_sl = centre_slices(sdf_vol)[0]
            _show(ax_row[1], sdf_sl, "SDF (axial)",
                  cmap="RdBu_r", vmin=-1, vmax=1)
        else:
            ax_row[1].axis("off")

        # col 2: skeleton coloured by curvature (axial)
        if skel_vol is not None and curv_vol is not None:
            skel_sl = centre_slices(skel_vol)[0].astype(float)
            curv_sl = centre_slices(curv_vol)[0]
            # show image background, overlay skeleton pixels coloured by curvature
            img_sl  = centre_slices(img_vol)[0]
            ax_row[2].set_facecolor("#1a1a1a")
            ax_row[2].imshow(norm01(img_sl), cmap="gray", origin="lower",
                             aspect="equal", interpolation="nearest")
            masked_curv = np.ma.masked_where(skel_sl == 0, curv_sl)
            sc = ax_row[2].imshow(masked_curv, cmap="plasma",
                                  vmin=0, vmax=np.percentile(curv_sl[skel_sl > 0], 95)
                                              if skel_sl.any() else 1,
                                  origin="lower", aspect="equal",
                                  interpolation="nearest", alpha=0.9)
            plt.colorbar(sc, ax=ax_row[2], fraction=0.03, pad=0.02).ax.yaxis.label.set_color("white")
            ax_row[2].set_title("Skeleton curvature (axial)", color="white", fontsize=8, pad=3)
            ax_row[2].axis("off")
        elif skel_vol is not None:
            skel_sl = centre_slices(skel_vol)[0]
            _show(ax_row[2], skel_sl, "Skeleton (axial)", cmap="hot")
        else:
            ax_row[2].axis("off")

    plt.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset runner
# ─────────────────────────────────────────────────────────────────────────────
def process_dataset(ds_id: int, n_samples: int) -> List[Path]:
    """Visualize n_samples from dataset ds_id. Returns list of saved PNGs."""
    if ds_id not in DATASETS:
        print(f"[WARN] Dataset {ds_id} not in registry — skipping.")
        return []

    dir_name, aux_name = DATASETS[ds_id]
    ds_raw = RAW_ROOT / dir_name
    if not ds_raw.exists():
        print(f"[WARN] {ds_raw} not found — skipping.")
        return []

    meta_path = ds_raw / "dataset.json"
    with open(meta_path) as f:
        meta = json.load(f)

    label_names = {int(v): k for k, v in meta.get("labels", {}).items()}  # id→name
    ds_name     = meta.get("name", dir_name)

    imgs_dir = ds_raw / "imagesTr"
    lbls_dir = ds_raw / "labelsTr"
    aux_base = AUX_ROOT / aux_name if aux_name else None

    # collect image files, skip macOS ._  artefacts
    img_files = sorted(
        f for f in imgs_dir.glob("*_0000.nii.gz") if not f.name.startswith("._")
    )
    if not img_files:
        print(f"[WARN] No images in {imgs_dir} — skipping.")
        return []

    # pick evenly spaced samples
    indices  = np.linspace(0, len(img_files) - 1, min(n_samples, len(img_files)),
                           dtype=int)
    selected = [img_files[i] for i in indices]

    out_dir  = VIS_ROOT / ds_name
    saved    = []

    print(f"\n{'─'*60}")
    print(f"Dataset {ds_id:3d}  {ds_name}  ({len(img_files)} subjects)")
    print(f"  Labels: {len(label_names)-1 if 0 in label_names.values() else len(label_names)} classes")
    print(f"  Visualising {len(selected)} samples …")

    for img_path in selected:
        # subject id: strip _0000.nii.gz
        subj_id  = img_path.name.replace("_0000.nii.gz", "")
        lbl_path = lbls_dir / f"{subj_id}.nii.gz"
        out_png  = out_dir / f"{subj_id}.png"

        try:
            plot_sample(
                img_path    = img_path,
                lbl_path    = lbl_path if lbl_path.exists() else None,
                aux_base    = aux_base,
                subject_id  = subj_id,
                label_names = label_names,
                out_path    = out_png,
            )
            saved.append(out_png)
        except Exception as exc:
            print(f"  [ERROR] {subj_id}: {exc}")

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Overview mosaic  (one thumbnail per dataset)
# ─────────────────────────────────────────────────────────────────────────────
def make_overview_mosaic(saved_per_ds: Dict[int, List[Path]]) -> None:
    """Tile the first PNG from each dataset into a single overview figure."""
    entries = [(ds_id, paths[0]) for ds_id, paths in saved_per_ds.items() if paths]
    if not entries:
        return

    n = len(entries)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 7, nrows * 5.5),
                             facecolor="#111111")
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for ax, (ds_id, png_path) in zip(axes_flat, entries):
        img = plt.imread(str(png_path))
        ax.imshow(img)
        ax.set_title(DATASETS[ds_id][0].replace("Dataset", "DS"),
                     color="white", fontsize=8, pad=3)
        ax.axis("off")

    # hide unused axes
    for ax in axes_flat[len(entries):]:
        ax.axis("off")

    plt.suptitle("Dataset Overview — first sample per dataset",
                 color="white", fontsize=14, y=1.01)
    plt.tight_layout(pad=0.3)
    out = VIS_ROOT / "overview_mosaic.png"
    fig.savefig(out, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nOverview mosaic → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Visualise preprocessed nnU-Net datasets (server-safe, saves PNGs)."
    )
    parser.add_argument(
        "--datasets", nargs="+", type=int,
        default=list(DATASETS.keys()),
        help="Dataset IDs to visualise (default: all). E.g. --datasets 100 101 110",
    )
    parser.add_argument(
        "--n", type=int, default=2,
        help="Number of samples per dataset (default: 2)",
    )
    parser.add_argument(
        "--no-mosaic", action="store_true",
        help="Skip the overview mosaic.",
    )
    args = parser.parse_args()

    VIS_ROOT.mkdir(parents=True, exist_ok=True)
    saved_per_ds: Dict[int, List[Path]] = {}

    for ds_id in args.datasets:
        saved_per_ds[ds_id] = process_dataset(ds_id, args.n)

    if not args.no_mosaic:
        make_overview_mosaic(saved_per_ds)

    total = sum(len(v) for v in saved_per_ds.values())
    print(f"\nDone. {total} PNGs saved under {VIS_ROOT}/")


if __name__ == "__main__":
    main()
