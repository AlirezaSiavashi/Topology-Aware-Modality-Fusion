#!/usr/bin/env python3
"""
Materialise external-set ground truth in the prediction grid, with labels
remapped to the Dataset104 convention.

Two things have to be reconciled before any zero-shot number means anything:

1. Geometry. The nnU-Net raw images for Datasets 200-202 were cropped to a
   brain ROI during preprocessing, so predictions do not share a grid with the
   original `cow_seg_labelsTr` volumes (0/26 and 0/20 shapes matched). GT is
   resampled into the prediction grid with nearest-neighbour interpolation via
   the affines.

2. Label values. The external sets keep TopCoW's original convention, in which
   3rd-A2 is 15; Dataset104 remaps that to 13 and leaves 1-12 unchanged. Only
   that single remap is needed -- but without it 3rd-A2 scores zero and every
   other class is unaffected, which is exactly the kind of silent partial
   failure that survives review.

Note also that training/compute_zeroshot_metrics.py carries a scrambled label
map (5:L-ICA, 6:R-MCA, 8:R-ACA, 10:R-Pcom, 12:Acom) that does not match
Dataset104's dataset.json (5:R-MCA, 6:L-ICA, 8:R-Pcom, 10:Acom, 12:L-ACA).
Per-class numbers from that script are mislabelled; this module reads the
mapping from dataset.json instead of hardcoding it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

# External (original TopCoW) value -> Dataset104 value. Identity for 1..12.
EXTERNAL_TO_DS104 = {15: 13}


def resample_label(label_nii, target_nii) -> np.ndarray:
    """Nearest-neighbour resample of a label volume onto the target grid."""
    lab = np.asarray(label_nii.dataobj).astype(np.int32)
    shape = target_nii.shape
    if lab.shape == shape and np.allclose(label_nii.affine, target_nii.affine):
        return lab

    grid = np.indices(shape).reshape(3, -1)
    homo = np.vstack([grid, np.ones((1, grid.shape[1]))])
    xform = np.linalg.inv(label_nii.affine) @ target_nii.affine
    coords = (xform @ homo)[:3]
    out = map_coordinates(lab, coords, order=0, mode="constant", cval=0)
    return out.reshape(shape).astype(np.int32)


def remap(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    for src, dst in EXTERNAL_TO_DS104.items():
        out[arr == src] = dst
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_src", required=True, help="original cow_seg_labelsTr")
    ap.add_argument("--gt_suffix", default="_full_LPS_Mask")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dataset_json", default=None,
                    help="verify the target label space (optional)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.dataset_json:
        labels = json.loads(Path(args.dataset_json).read_text())["labels"]
        vals = sorted(int(v) for v in labels.values())
        print(f"target label space: {vals}")

    preds = sorted(Path(args.pred_dir).glob("*.nii.gz"))
    if not preds:
        raise SystemExit(f"no predictions in {args.pred_dir}")

    written, missing, seen = 0, [], set()
    for p in preds:
        stem = p.name.replace(".nii.gz", "")
        src = Path(args.gt_src) / f"{stem}{args.gt_suffix}.nii.gz"
        if not src.exists():
            missing.append(stem)
            continue
        tgt_nii = nib.load(str(p))
        arr = remap(resample_label(nib.load(str(src)), tgt_nii))
        seen |= set(np.unique(arr).tolist())
        nib.save(nib.Nifti1Image(arr.astype(np.uint8), tgt_nii.affine,
                                 tgt_nii.header), out / p.name)
        written += 1

    print(f"wrote {written} resampled GT volumes to {out}")
    print(f"label values present after remap: {sorted(int(s) for s in seen)}")
    if missing:
        print(f"WARNING: {len(missing)} predictions had no GT: {missing[:5]}")


if __name__ == "__main__":
    main()
