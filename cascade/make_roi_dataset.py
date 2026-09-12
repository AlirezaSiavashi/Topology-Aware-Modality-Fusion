#!/usr/bin/env python3
"""
make_roi_dataset.py
===================
Build the stage-2 dataset of the CoW cascade: the ORIGINAL TopCoW volumes at
their native spacing, cropped to the Circle-of-Willis region.

Why this is the biggest single lever on this dataset
----------------------------------------------------
Measured over all 250 cases of Dataset104:

  * The CoW bounding box is **3.24 %** of the volume. Foreground is 0.076 % of
    the whole volume but **2.35 % inside the box** -- 31x denser. Every loss
    that depends on seeing vessels (and all six state losses do) currently
    spends most of its gradient on background.
  * The 112x160x128 training patch contains the **whole** CoW in only
    **175/250 cases (70 %)**. In the other 30 % the ring is cut, so `L_ring`
    and `L_mirror` are scored on a fragment and the presence head is asked
    about vessels that were cropped away rather than absent. Those two losses
    are the ones the rotational state exists to serve.
  * Preprocessing to 0.5 mm isotropic **discards real resolution**: native MRA
    is 0.297 mm in-plane (0.5 mm is a 1.7x loss) and native CTA is 0.43 mm.
    A 1.5 mm Pcom is ~5 voxels across natively and ~3 after resampling -- and
    Pcom/Acom/3rd-A2 are exactly the classes that fail (0.47/0.46/0.59/0.31).
    Cropping first is what makes keeping the native resolution affordable.

So the cascade buys three things at once: class balance, an unfragmented ring,
and the resolution the current pipeline throws away.

Stage 1 needs no training
-------------------------
Measured on the 50 held-out cases: a **15 mm margin** around the predicted
foreground bounding box of an ALREADY-TRAINED Dataset104 model contains the
full ground-truth CoW in **50/50 cases (100 %)**; 8 mm suffices for 96 %.
Localisation is far easier than segmentation, so any existing checkpoint serves
as the localiser and no stage-1 run is needed. See `predict_cascade.py`.

Train/test ROI consistency
--------------------------
Training crops use the GT box + the same 15 mm margin; inference crops use the
predicted box + 15 mm. The predicted box is a median of 3.0 mm off the GT box,
so the two ROI distributions are close. `--jitter-mm` adds random slack at
build time if you want to widen that further.

Usage
-----
  aneur/bin/python3 cascade/make_roi_dataset.py --dataset-id 107
  nnUNetv2_plan_and_preprocess -d 107 -c 3d_fullres --verify_dataset_integrity
"""
from __future__ import annotations

import argparse, json, os, shutil, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import SimpleITK as sitk

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = f"{BASE}/dataset/15692630/TopCoW2024_Data_Release"
RAW = f"{BASE}/preprocessed/nnUNet_raw"
PRE = f"{BASE}/preprocessed/nnUNet_preprocessed"

LABELS = {"background": 0, "BA": 1, "R-PCA": 2, "L-PCA": 3, "R-ICA": 4,
          "R-MCA": 5, "L-ICA": 6, "L-MCA": 7, "R-Pcom": 8, "L-Pcom": 9,
          "Acom": 10, "R-ACA": 11, "L-ACA": 12, "3rd-A2": 13}

# The TopCoW release numbers the third A2 as **15**, not 13; verified over all
# 250 cases, the release uses {0..12, 15} and Dataset104 uses {0..13}. This is
# the single remap Dataset104's builder applies, and it must be reproduced here
# or the ROI dataset's classes silently disagree with every existing result.
LABEL_REMAP = {15: 13}


def crop_case(args):
    case, out_img, out_lbl, margin_mm, jitter_mm, seed = args
    img = sitk.ReadImage(f"{SRC}/imagesTr/{case}_0000.nii.gz")
    lbl = sitk.ReadImage(f"{SRC}/cow_seg_labelsTr/{case}.nii.gz")

    if img.GetSize() != lbl.GetSize():
        # Geometry must agree before we can share one index-space crop.
        lbl = sitk.Resample(lbl, img, sitk.Transform(), sitk.sitkNearestNeighbor,
                            0, lbl.GetPixelID())

    a = sitk.GetArrayFromImage(lbl)                      # (z, y, x)
    if any(np.any(a == k) for k in LABEL_REMAP):
        a = a.copy()
        for k, v in LABEL_REMAP.items():
            a[a == k] = v
        remapped = sitk.GetImageFromArray(a)
        remapped.CopyInformation(lbl)
        lbl = remapped
    idx = np.argwhere(a > 0)
    if len(idx) == 0:
        return case, None
    lo, hi = idx.min(0), idx.max(0) + 1

    spacing_zyx = np.array(img.GetSpacing())[::-1]
    m = np.ceil(margin_mm / spacing_zyx).astype(int)
    if jitter_mm > 0:
        rng = np.random.default_rng(seed)
        j = np.ceil(rng.uniform(0, jitter_mm, 3) / spacing_zyx).astype(int)
        m = m + j
    lo = np.maximum(lo - m, 0)
    hi = np.minimum(hi + m, np.array(a.shape))

    # sitk slicing is x,y,z and carries origin/direction across correctly
    sl = (slice(int(lo[2]), int(hi[2])), slice(int(lo[1]), int(hi[1])),
          slice(int(lo[0]), int(hi[0])))
    sitk.WriteImage(img[sl], out_img, True)
    sitk.WriteImage(lbl[sl], out_lbl, True)
    return case, dict(orig_size=list(img.GetSize()),
                      orig_spacing=list(img.GetSpacing()),
                      orig_origin=list(img.GetOrigin()),
                      crop_lo_zyx=[int(v) for v in lo],
                      crop_hi_zyx=[int(v) for v in hi],
                      roi_size=list(img[sl].GetSize()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", type=int, default=107)
    ap.add_argument("--name", default="CoW_ROI")
    ap.add_argument("--margin-mm", type=float, default=15.0,
                    help="15 mm contains the GT CoW for 50/50 held-out cases "
                         "when the box comes from a trained model's prediction")
    ap.add_argument("--jitter-mm", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--copy-split-from", default="Dataset104_TopCoW_joint",
                    help="reuse this dataset's folds so the held-out subjects "
                         "stay byte-identical to every other experiment")
    args = ap.parse_args()

    ds = f"Dataset{args.dataset_id:03d}_{args.name}"
    out = f"{RAW}/{ds}"
    os.makedirs(f"{out}/imagesTr", exist_ok=True)
    os.makedirs(f"{out}/labelsTr", exist_ok=True)

    cases = sorted(f[:-len("_0000.nii.gz")]
                   for f in os.listdir(f"{SRC}/imagesTr") if f.endswith("_0000.nii.gz"))
    print(f"{ds}: cropping {len(cases)} cases, margin {args.margin_mm} mm"
          + (f" + up to {args.jitter_mm} mm jitter" if args.jitter_mm else ""))

    jobs = [(c, f"{out}/imagesTr/{c}_0000.nii.gz", f"{out}/labelsTr/{c}.nii.gz",
             args.margin_mm, args.jitter_mm, i) for i, c in enumerate(cases)]
    geom, skipped = {}, []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (case, g) in enumerate(ex.map(crop_case, jobs)):
            if g is None:
                skipped.append(case)
            else:
                geom[case] = g
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(cases)}")

    json.dump(geom, open(f"{out}/roi_geometry.json", "w"), indent=1)
    json.dump({"channel_names": {"0": "CTA_or_MRA"},
               "labels": LABELS,
               "numTraining": len(geom),
               "file_ending": ".nii.gz",
               "description": f"TopCoW CoW-ROI crops at native spacing, "
                              f"{args.margin_mm} mm margin (stage 2 of the cascade)"},
              open(f"{out}/dataset.json", "w"), indent=1)

    # Reuse the existing folds. TopCoW ct_i / mr_i are the SAME SUBJECT, so a
    # fresh random split would leak; copying is not just convenience.
    src_split = f"{PRE}/{args.copy_split_from}/splits_final.json"
    if os.path.exists(src_split):
        os.makedirs(f"{PRE}/{ds}", exist_ok=True)
        shutil.copy(src_split, f"{PRE}/{ds}/splits_final.json")
        print(f"  copied folds from {args.copy_split_from} "
              f"(nnU-Net honours an existing splits_final.json)")
    else:
        print(f"  WARNING: {src_split} not found -- nnU-Net will make a random "
              f"split, which LEAKS (ct_i and mr_i are one subject)")

    sizes = np.array([g["roi_size"] for g in geom.values()])
    orig = np.array([g["orig_size"] for g in geom.values()])
    print(f"\nwrote {len(geom)} cases, skipped {len(skipped)}: {skipped}")
    print(f"ROI size voxels  median {np.median(sizes,0).astype(int)}  "
          f"max {sizes.max(0)}")
    print(f"ROI / original volume: {np.mean(sizes.prod(1)/orig.prod(1))*100:.1f}%")
    print(f"\nnext:  nnUNetv2_plan_and_preprocess -d {args.dataset_id} "
          f"-c 3d_fullres --verify_dataset_integrity")


if __name__ == "__main__":
    main()
