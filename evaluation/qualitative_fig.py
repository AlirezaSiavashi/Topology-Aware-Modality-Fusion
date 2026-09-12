#!/usr/bin/env python3
"""
Qualitative comparison figure for the Results section.

Two rendering choices matter for honesty:

  * Label panels use a FRONT-SURFACE projection -- the first non-zero voxel
    along each ray -- not a maximum-intensity projection. A raw MIP over a
    label map returns the numerically largest class index, which would
    systematically favour high-numbered classes over the ones actually in
    front.
  * The crop is computed once per row from the union bounding box of every
    panel in that row, so all columns show exactly the same field of view.

The image column is a MIP restricted to the axial slab containing the CoW.
A whole-volume CTA MIP is dominated by skull, which hides the vessels.
"""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np

NAMES = ['BA','R-PCA','L-PCA','R-ICA','R-MCA','L-ICA','L-MCA',
         'R-Pcom','L-Pcom','Acom','R-ACA','L-ACA','3rd-A2']
CMAP = plt.get_cmap('tab20')
COLORS = np.vstack([[1, 1, 1], [CMAP(i)[:3] for i in range(13)]])

RAW = Path("preprocessed/nnUNet_raw/Dataset104_TopCoW_joint")
RES = Path("results/nnUNet/Dataset104_TopCoW_joint")
COLS = [("Angiogram", "IMAGE"), ("Ground truth", None),
        ("nnU-Net", "nnUNetTrainer"),
        ("w/o state objectives", "nnUNetTrainerM3CoW_NoStateLoss"),
        ("M3CoW (ours)", "nnUNetTrainerM3CoW")]


def load_lab(cid, trainer):
    p = RAW / "labelsTr" / f"{cid}.nii.gz" if trainer is None else \
        RES / f"{trainer}__nnUNetPlans__3d_fullres/fold_0/validation/{cid}.nii.gz"
    return np.asanyarray(nib.load(str(p)).dataobj).astype(int)


def surface(lab, z0, z1):
    """Label of the first non-zero voxel along z, within the CoW slab."""
    sub = lab[:, :, z0:z1]
    m = sub > 0
    first = m.argmax(axis=2)
    i, j = np.indices(first.shape)
    return np.where(m.any(axis=2), sub[i, j, first], 0).astype(int)


def box3(labs, pad=6, zpad=4):
    nz = [np.argwhere(l > 0) for l in labs if (l > 0).any()]
    a = np.vstack(nz)
    return (max(0, a[:, 0].min() - pad), a[:, 0].max() + pad,
            max(0, a[:, 1].min() - pad), a[:, 1].max() + pad,
            max(0, a[:, 2].min() - zpad), a[:, 2].max() + zpad)


def main(cases, out):
    n = len(cases)
    fig, axes = plt.subplots(n, len(COLS), figsize=(10.2, 2.15 * n))
    axes = np.atleast_2d(axes)
    for r, (cid, note, stats) in enumerate(cases):
        labs = [load_lab(cid, t) for _, t in COLS if t != "IMAGE"]
        y0, y1, x0, x1, z0, z1 = box3(labs)
        vol = np.asanyarray(nib.load(str(RAW / "imagesTr" / f"{cid}_0000.nii.gz")).dataobj)
        mip = vol[:, :, z0:z1].max(axis=2)[y0:y1, x0:x1]
        lo, hi = np.percentile(mip, [2, 99.5])
        k = 0
        for c, (cname, tr) in enumerate(COLS):
            ax = axes[r, c]
            if tr == "IMAGE":
                ax.imshow(np.rot90(np.clip((mip - lo) / max(hi - lo, 1e-6), 0, 1)),
                          cmap="gray", interpolation="bilinear")
            else:
                ax.imshow(np.rot90(COLORS[np.clip(surface(labs[k], z0, z1), 0, 13)[y0:y1, x0:x1]]),
                          interpolation="nearest")
                k += 1
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_linewidth(0.4); sp.set_color("0.7")
            if r == 0: ax.set_title(cname, fontsize=9, pad=4)
            if stats.get(cname): ax.set_xlabel(stats[cname], fontsize=7.5, labelpad=2)
        axes[r, 0].set_ylabel(note, fontsize=8)
    handles = [Patch(facecolor=COLORS[i + 1], label=nm) for i, nm in enumerate(NAMES)]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=[0, 0.055, 1, 1])
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    cases = [
        ("topcow_ct_034", "CTA  fragmented baseline",
         {"nnU-Net": "Dice 0.39   B0 2.82", "w/o state objectives": "Dice 0.20   B0 0.64",
          "M3CoW (ours)": "Dice 0.77   B0 0.27"}),
        ("topcow_mr_136", "MRA  topology restored",
         {"nnU-Net": "Dice 0.77   B0 0.18", "w/o state objectives": "Dice 0.79   B0 0.09",
          "M3CoW (ours)": "Dice 0.82   B0 0.00"}),
        ("topcow_ct_152", "CTA  typical case",
         {"nnU-Net": "Dice 0.81   B0 0.08", "w/o state objectives": "Dice 0.80   B0 0.00",
          "M3CoW (ours)": "Dice 0.81   B0 0.08"}),
    ]
    main(cases, Path("figures/qualitative.png"))
