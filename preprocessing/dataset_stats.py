"""
dataset_stats.py
================
Inspect and validate ALL raw datasets (before preprocessing) and all
preprocessed outputs (after preprocessing).

Outputs:
  - Console summary table
  - CSV file: <OUTPUT_ROOT>/dataset_stats.csv
  - JSON file: <OUTPUT_ROOT>/dataset_stats.json

Usage
-----
  # Inspect raw source data (before preprocessing)
  python dataset_stats.py --mode raw

  # Inspect preprocessed output (after preprocessing)
  python dataset_stats.py --mode preprocessed

  # Both
  python dataset_stats.py --mode all
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Per-file stats
# ──────────────────────────────────────────────────────────────────────────────

def get_image_stats(path: Path) -> Dict:
    """Extract key statistics from a single NIfTI file."""
    try:
        img  = sitk.ReadImage(str(path))
        arr  = sitk.GetArrayFromImage(img).astype(np.float32)
        sp   = img.GetSpacing()   # (x, y, z)
        sz   = img.GetSize()      # (x, y, z)
        dire = img.GetDirection()

        return {
            "file":        path.name,
            "shape_xyz":   list(sz),
            "spacing_xyz": [round(float(s), 4) for s in sp],
            "dtype":       str(img.GetPixelIDTypeAsString()),
            "min":         round(float(arr.min()), 4),
            "max":         round(float(arr.max()), 4),
            "mean":        round(float(arr.mean()), 4),
            "std":         round(float(arr.std()), 4),
            "nonzero_vox": int((arr != 0).sum()),
            "total_vox":   int(arr.size),
            "orientation": _direction_to_orientation(dire),
            "status":      "ok",
        }
    except Exception as e:
        return {"file": path.name, "status": f"ERROR: {e}"}


def get_label_stats(path: Path) -> Dict:
    """Extract label-specific statistics (unique values, class volumes)."""
    try:
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img).astype(np.int32)
        sp  = img.GetSpacing()
        vox_vol_mm3 = float(sp[0] * sp[1] * sp[2])

        unique, counts = np.unique(arr, return_counts=True)
        class_volumes  = {int(u): round(int(c) * vox_vol_mm3, 2)
                          for u, c in zip(unique, counts) if u > 0}
        return {
            "file":          path.name,
            "shape_xyz":     list(img.GetSize()),
            "spacing_xyz":   [round(float(s), 4) for s in sp],
            "unique_labels": [int(u) for u in unique],
            "num_classes":   int(len(unique)) - 1,   # exclude background
            "fg_voxels":     int((arr > 0).sum()),
            "total_voxels":  int(arr.size),
            "fg_ratio_pct":  round(100.0 * (arr > 0).mean(), 3),
            "class_vol_mm3": class_volumes,
            "status":        "ok",
        }
    except Exception as e:
        return {"file": path.name, "status": f"ERROR: {e}"}


def _direction_to_orientation(direction: Tuple) -> str:
    """Convert ITK direction cosine matrix to orientation string (e.g. LPS)."""
    d = np.array(direction).reshape(3, 3)
    labels = "RLAPIS"   # R/L, A/P, I/S
    orient = ""
    for col in range(3):
        axis = np.argmax(np.abs(d[:, col]))
        sign = d[axis, col]
        if axis == 0:
            orient += "L" if sign > 0 else "R"
        elif axis == 1:
            orient += "P" if sign > 0 else "A"
        else:
            orient += "S" if sign > 0 else "I"
    return orient


# ──────────────────────────────────────────────────────────────────────────────
# Dataset-level inspection
# ──────────────────────────────────────────────────────────────────────────────

def inspect_directory(
    name: str,
    imgs_dir: Path,
    lbls_dir: Optional[Path] = None,
    max_subjects: int = 5,
) -> Dict:
    """
    Summarise one dataset directory.
    Returns a dict with per-subject and aggregate statistics.
    """
    result = {"dataset": name, "subjects": [], "aggregate": {}}

    img_files = sorted(imgs_dir.glob("*.nii.gz")) if imgs_dir.exists() else []
    if not img_files:
        result["aggregate"]["error"] = f"No .nii.gz files in {imgs_dir}"
        return result

    # Sample up to max_subjects for detailed stats (full run is slow)
    sample = img_files[:max_subjects]
    result["aggregate"]["total_subjects"] = len(img_files)
    result["aggregate"]["sampled"] = len(sample)

    spacings = []
    shapes   = []
    errors   = []

    for img_path in sample:
        stats = get_image_stats(img_path)
        subject = {"image": stats}

        if lbls_dir and lbls_dir.exists():
            # Match label file
            stem     = img_path.name.replace("_0000.nii.gz", "").replace(".nii.gz", "")
            lbl_path = lbls_dir / f"{stem}.nii.gz"
            if lbl_path.exists():
                subject["label"] = get_label_stats(lbl_path)
                # Sanity check: image and label shapes must match
                if (subject["image"].get("shape_xyz") !=
                        subject["label"].get("shape_xyz")):
                    subject["shape_mismatch"] = True
                    errors.append(f"Shape mismatch: {img_path.name}")
            else:
                subject["label"] = {"status": f"NOT FOUND: {lbl_path.name}"}

        result["subjects"].append(subject)

        if stats.get("status") == "ok":
            spacings.append(stats["spacing_xyz"])
            shapes.append(stats["shape_xyz"])

    if spacings:
        sp_arr = np.array(spacings)
        sh_arr = np.array(shapes)
        result["aggregate"].update({
            "spacing_mean_xyz":   [round(float(v), 4) for v in sp_arr.mean(axis=0)],
            "spacing_min_xyz":    [round(float(v), 4) for v in sp_arr.min(axis=0)],
            "spacing_max_xyz":    [round(float(v), 4) for v in sp_arr.max(axis=0)],
            "shape_mean_xyz":     [round(float(v), 1) for v in sh_arr.mean(axis=0)],
            "anisotropy_max":     round(float(sp_arr.max() / (sp_arr.min() + 1e-8)), 2),
        })

    if errors:
        result["aggregate"]["errors"] = errors

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Raw dataset inspection
# ──────────────────────────────────────────────────────────────────────────────

def inspect_raw() -> List[Dict]:
    results = []

    # TopBrain CT
    results.append(inspect_directory(
        "TopBrain_CT_raw",
        cfg.TOPBRAIN_ROOT / "imagesTr_topbrain_ct",
        cfg.TOPBRAIN_ROOT / "labelsTr_topbrain_ct",
    ))

    # TopBrain MR
    results.append(inspect_directory(
        "TopBrain_MR_raw",
        cfg.TOPBRAIN_ROOT / "imagesTr_topbrain_mr",
        cfg.TOPBRAIN_ROOT / "labelsTr_topbrain_mr",
    ))

    # TopCoW CT
    cow_imgs = cfg.TOPCOW_ROOT / "imagesTr"
    cow_lbls = cfg.TOPCOW_ROOT / "cow_seg_labelsTr"

    if cow_imgs.exists():
        ct_imgs_dir  = cow_imgs   # share directory; filter in inspect
        ct_files_all = [f for f in sorted(cow_imgs.glob("*.nii.gz")) if "_ct_" in f.name]
        mr_files_all = [f for f in sorted(cow_imgs.glob("*.nii.gz")) if "_mr_" in f.name]

        logger.info("TopCoW: %d CT + %d MR images", len(ct_files_all), len(mr_files_all))

        # Manually inspect sample – build a temp dir-like structure
        for mod, files_all in [("CT", ct_files_all), ("MR", mr_files_all)]:
            sample    = files_all[:5]
            spacings  = []
            shapes    = []
            subjects  = []
            for f in sample:
                stem  = f.name.replace("_0000.nii.gz", "")
                lbl   = cow_lbls / f"{stem}.nii.gz"
                img_s = get_image_stats(f)
                lbl_s = get_label_stats(lbl) if lbl.exists() else {"status": "missing"}
                subjects.append({"image": img_s, "label": lbl_s})
                if img_s.get("status") == "ok":
                    spacings.append(img_s["spacing_xyz"])
                    shapes.append(img_s["shape_xyz"])
            agg = {"total_subjects": len(files_all), "sampled": len(sample)}
            if spacings:
                import numpy as _np
                sp_arr = _np.array(spacings)
                sh_arr = _np.array(shapes)
                agg.update({
                    "spacing_mean_xyz": [round(float(v), 4) for v in sp_arr.mean(axis=0)],
                    "spacing_min_xyz":  [round(float(v), 4) for v in sp_arr.min(axis=0)],
                    "spacing_max_xyz":  [round(float(v), 4) for v in sp_arr.max(axis=0)],
                    "shape_mean_xyz":   [round(float(v), 1) for v in sh_arr.mean(axis=0)],
                    "anisotropy_max":   round(float(sp_arr.max() / (sp_arr.min() + 1e-8)), 2),
                })
            results.append({"dataset": f"TopCoW_{mod}_raw",
                             "subjects": subjects, "aggregate": agg})

    # Cross-domain subsets
    for key, path in cfg.CROSS_DOMAIN_BRAIN.items():
        imgs = path / "imagesTr"
        lbls = path / "cow_seg_labelsTr"
        if imgs.exists():
            results.append(inspect_directory(f"{key}_raw", imgs, lbls))

    # External datasets (only if paths exist)
    for name, root in cfg.EXTERNAL_DATASETS.items():
        for imgs_subdir in ["imagesTr", "train/images", "images",
                            "Normal", "training/images"]:
            imgs = root / imgs_subdir
            if imgs.exists():
                results.append(inspect_directory(f"{name}_raw", imgs, max_subjects=3))
                break

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessed output inspection
# ──────────────────────────────────────────────────────────────────────────────

def inspect_preprocessed() -> List[Dict]:
    nnunet_raw = cfg.OUTPUT_ROOT / "nnUNet_raw"
    results    = []

    if not nnunet_raw.exists():
        logger.warning("nnUNet_raw not found: %s  (run preprocessing first)", nnunet_raw)
        return results

    for ds_dir in sorted(nnunet_raw.iterdir()):
        if not ds_dir.is_dir():
            continue
        imgs_dir = ds_dir / "imagesTr"
        lbls_dir = ds_dir / "labelsTr"
        r = inspect_directory(ds_dir.name, imgs_dir, lbls_dir, max_subjects=3)

        # Also check aux maps exist
        aux_root = cfg.OUTPUT_ROOT / "aux" / ds_dir.name.split("_", 1)[1]
        for aux_name in ["vessel_union", "sdf", "skeleton", "curvature"]:
            aux_dir = aux_root / aux_name
            count   = len(list(aux_dir.glob("*.nii.gz"))) if aux_dir.exists() else 0
            r.setdefault("aux", {})[aux_name] = count

        results.append(r)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Pretty-print table
# ──────────────────────────────────────────────────────────────────────────────

def print_summary_table(results: List[Dict]) -> None:
    col_w = [30, 8, 20, 20, 12]
    header = ["Dataset", "N", "Spacing (mm, mean XYZ)", "Shape (mean XYZ)", "Anisotropy"]
    sep    = "  ".join("-" * w for w in col_w)
    fmt    = "  ".join(f"{{:<{w}}}" for w in col_w)

    print("\n" + "=" * sum(col_w) + "=" * (2 * (len(col_w) - 1)))
    print(fmt.format(*header))
    print(sep)

    for r in results:
        agg  = r.get("aggregate", {})
        n    = str(agg.get("total_subjects", "?"))
        sp   = agg.get("spacing_mean_xyz")
        sh   = agg.get("shape_mean_xyz")
        anis = str(agg.get("anisotropy_max", "?"))
        sp_str = f"{sp[0]:.3f},{sp[1]:.3f},{sp[2]:.3f}" if sp else "?"
        sh_str = f"{int(sh[0])},{int(sh[1])},{int(sh[2])}" if sh else "?"
        errs = agg.get("errors", [])
        print(fmt.format(r["dataset"][:30], n, sp_str, sh_str, anis))
        if errs:
            for e in errs:
                print(f"  !! {e}")

    print(sep)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# CSV + JSON export
# ──────────────────────────────────────────────────────────────────────────────

def save_results(results: List[Dict], prefix: str) -> None:
    out_dir = cfg.OUTPUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = out_dir / f"{prefix}_stats.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved JSON: %s", json_path)

    # CSV (aggregate only)
    csv_path = out_dir / f"{prefix}_stats.csv"
    rows = []
    for r in results:
        agg = r.get("aggregate", {})
        sp  = agg.get("spacing_mean_xyz", [None, None, None])
        sh  = agg.get("shape_mean_xyz",   [None, None, None])
        rows.append({
            "dataset":          r["dataset"],
            "total_subjects":   agg.get("total_subjects"),
            "spacing_x":        sp[0], "spacing_y": sp[1], "spacing_z": sp[2],
            "shape_x":          sh[0], "shape_y":   sh[1], "shape_z":   sh[2],
            "anisotropy_max":   agg.get("anisotropy_max"),
            "errors":           "; ".join(agg.get("errors", [])),
        })

    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Saved CSV:  %s", csv_path)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inspect dataset statistics.")
    parser.add_argument(
        "--mode",
        default="raw",
        choices=["raw", "preprocessed", "all"],
        help="What to inspect.",
    )
    args = parser.parse_args()

    if args.mode in ("raw", "all"):
        logger.info("Inspecting RAW datasets ...")
        raw_results = inspect_raw()
        print_summary_table(raw_results)
        save_results(raw_results, "raw")

    if args.mode in ("preprocessed", "all"):
        logger.info("Inspecting PREPROCESSED outputs ...")
        pre_results = inspect_preprocessed()
        if pre_results:
            print_summary_table(pre_results)
            save_results(pre_results, "preprocessed")
        else:
            logger.info("No preprocessed data found yet.")


if __name__ == "__main__":
    main()
