#!/usr/bin/env python3
"""
Render paired CTA / MRA for the same TopCoW subject, side by side.

TopCoW is subject-paired (topcow_ct_XXX and topcow_mr_XXX are the same
patient) but the two volumes are NOT co-registered -- different shape,
spacing and origin. So this shows maximum-intensity projections, which is how
angiography is read anyway, rather than matched slices.

The point is to see *why* the model is 0.116 Dice worse on CTA: what is
actually visible in each modality, and what competes with the vessels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "preprocessed/nnUNet_raw/Dataset104_TopCoW_joint"

# Dataset104 label values
NAMES = {1: "BA", 2: "R-PCA", 3: "L-PCA", 4: "R-ICA", 5: "R-MCA", 6: "L-ICA",
         7: "L-MCA", 8: "R-Pcom", 9: "L-Pcom", 10: "Acom", 11: "R-ACA",
         12: "L-ACA", 13: "3rd-A2"}
# Highlight the classes where CTA is worst (measured gaps)
FOCUS = {8: "R-Pcom", 9: "L-Pcom", 10: "Acom", 13: "3rd-A2"}


def load(mod: str, sid: str):
    img = nib.load(str(RAW / "imagesTr" / f"topcow_{mod}_{sid}_0000.nii.gz"))
    lab = nib.load(str(RAW / "labelsTr" / f"topcow_{mod}_{sid}.nii.gz"))
    return np.asarray(img.dataobj, dtype=np.float32), np.asarray(lab.dataobj), img


def mip(vol: np.ndarray, axis: int = 2, lab: np.ndarray = None) -> np.ndarray:
    """Thin-slab MIP centred on the Circle of Willis.

    A full-volume MIP is useless for CTA: bone is brighter than contrast-filled
    vessels, so the projection is pure skull. Radiologists read a thin slab for
    the same reason. The slab is centred on the labelled vessels.
    """
    if lab is not None and (lab > 0).any():
        idx = np.where((lab > 0).any(axis=tuple(i for i in range(3) if i != axis)))[0]
        c = int((idx.min() + idx.max()) / 2)
        half = max(12, int((idx.max() - idx.min()) * 0.6))
        sl = [slice(None)] * 3
        sl[axis] = slice(max(0, c - half), min(vol.shape[axis], c + half))
        vol = vol[tuple(sl)]
    return vol.max(axis=axis)


def label_mip(lab: np.ndarray, axis: int = 2) -> np.ndarray:
    """Project labels by taking the max label id along the axis."""
    return lab.max(axis=axis)


def window(x: np.ndarray, lo=1.0, hi=99.5) -> np.ndarray:
    a, b = np.percentile(x, lo), np.percentile(x, hi)
    return np.clip((x - a) / max(b - a, 1e-6), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=["005", "017", "042"])
    ap.add_argument("--axis", type=int, default=2, help="MIP axis (2=axial)")
    ap.add_argument("--out", default="preprocessing_vis/paired_cta_mra.png")
    args = ap.parse_args()

    n = len(args.subjects)
    fig, axes = plt.subplots(n, 4, figsize=(17, 4.4 * n))
    if n == 1:
        axes = axes[None, :]

    cmap = plt.get_cmap("tab20")

    for r, sid in enumerate(args.subjects):
        for c_off, mod in ((0, "ct"), (2, "mr")):
            vol, lab, nii = load(mod, sid)
            m = window(mip(vol, args.axis, lab), 2.0, 99.9)
            lm = mip(lab.astype(np.float32), args.axis, lab).astype(np.int32)

            ax = axes[r, c_off]
            ax.imshow(m.T, cmap="gray", origin="lower")
            ax.set_title(f"{'CTA' if mod=='ct' else 'MRA'} {sid}\n"
                         f"{vol.shape}  {np.round(nii.header.get_zooms(),2)}mm",
                         fontsize=9)
            ax.axis("off")

            ax = axes[r, c_off + 1]
            ax.imshow(m.T, cmap="gray", origin="lower")
            overlay = np.zeros((*lm.T.shape, 4))
            for v in np.unique(lm):
                if v == 0:
                    continue
                mask = (lm.T == v)
                col = cmap((int(v) % 20) / 20.0)
                overlay[mask] = (*col[:3], 0.85)
            ax.imshow(overlay, origin="lower")
            present = [NAMES.get(int(v), str(int(v)))
                       for v in np.unique(lm) if v != 0]
            focus_here = [f for k, f in FOCUS.items() if k in np.unique(lm)]
            ax.set_title(f"{'CTA' if mod=='ct' else 'MRA'} labels ({len(present)})\n"
                         f"small vessels: {', '.join(focus_here) if focus_here else 'none'}",
                         fontsize=9)
            ax.axis("off")

    fig.suptitle("Paired CTA / MRA, same patients (not co-registered) — "
                 "axial maximum-intensity projection", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = BASE / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
