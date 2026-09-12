#!/usr/bin/env python3
"""
predict_cascade.py
==================
Two-stage CoW inference: localise, crop, segment at native resolution, paste back.

  stage 1   an ALREADY-TRAINED Dataset104 model on the full volume
            -> binary foreground -> bounding box + 15 mm
  stage 2   the Dataset107_CoW_ROI model on that crop
            (0.60 x 0.3525 x 0.3525 mm, vs 0.5 mm isotropic for stage 1)
  paste     stage-2 labels back into the original geometry

Stage 1 needs no dedicated training. Measured on the 50 held-out cases, a 15 mm
margin around the predicted box of an existing model (MF-SSM, class-avg Dice
0.688) contains the full ground-truth CoW in 50/50 cases; 8 mm covers 96 %.
Localisation is a far easier problem than segmentation, so spending a 40-hour
run on a dedicated localiser buys nothing.

Failure mode to watch: if stage 1 predicts nothing at all, there is no box. The
script falls back to the whole volume and says so, rather than emitting an empty
segmentation that would silently score 0.

Usage
-----
  cascade/predict_cascade.py \
      --images  evaluation/results/CLEAN/heldout_imgs_mr \
      --out     evaluation/results/CLEAN/cascade_m3cow_mr \
      --stage1-dataset 104 --stage1-trainer nnUNetTrainer \
      --stage2-dataset 107 --stage2-trainer nnUNetTrainerM3CoW \
      --modality mr --gpu 0
"""
from __future__ import annotations

import argparse, glob, json, os, shutil, subprocess, sys, tempfile

import numpy as np
import SimpleITK as sitk

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV = ("/scratch/siyavash/Alireza_thesis/external_dataset/"
        "rsna-intracranial-aneurysm-detection/nnunet_venv")


def env(gpu):
    e = dict(os.environ)
    e.update(nnUNet_raw=f"{BASE}/preprocessed/nnUNet_raw",
             nnUNet_preprocessed=f"{BASE}/preprocessed/nnUNet_preprocessed",
             nnUNet_results=f"{BASE}/results/nnUNet",
             CUDA_VISIBLE_DEVICES=str(gpu))
    return e


def predict(inp, out, dataset, trainer, gpu, modality, fold=0, tta=False):
    e = env(gpu)
    # Both MF-SSM and M3-CoW select B^m from this; sliding-window inference
    # never sees the batch keys, so it has to come from the environment.
    e["MFSSM_MODALITY"] = {"ct": "0", "mr": "1"}[modality]
    cmd = [f"{VENV}/bin/nnUNetv2_predict", "-i", inp, "-o", out,
           "-d", str(dataset), "-c", "3d_fullres", "-f", str(fold),
           "-tr", trainer, "-chk", "checkpoint_final.pth", "-npp", "4", "-nps", "4"]
    if not tta:
        cmd.append("--disable_tta")
    print(f"  $ {' '.join(cmd[1:])}")
    subprocess.run(cmd, env=e, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="dir of *_0000.nii.gz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage1-dataset", type=int, default=104)
    ap.add_argument("--stage1-trainer", default="nnUNetTrainer")
    ap.add_argument("--stage2-dataset", type=int, default=107)
    ap.add_argument("--stage2-trainer", default="nnUNetTrainerM3CoW")
    ap.add_argument("--margin-mm", type=float, default=15.0)
    ap.add_argument("--modality", choices=("ct", "mr"), required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--keep-temp", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="cascade_", dir="/tmp")
    s1, crops, s2 = f"{tmp}/stage1", f"{tmp}/crops", f"{tmp}/stage2"
    for d in (s1, crops, s2):
        os.makedirs(d)

    print(f"[1/4] localiser: Dataset{args.stage1_dataset} / {args.stage1_trainer}")
    predict(args.images, s1, args.stage1_dataset, args.stage1_trainer,
            args.gpu, args.modality, args.fold, tta=False)

    print(f"[2/4] cropping to predicted box + {args.margin_mm} mm")
    geom, fellback = {}, []
    for f in sorted(glob.glob(f"{args.images}/*_0000.nii.gz")):
        case = os.path.basename(f)[:-len("_0000.nii.gz")]
        img = sitk.ReadImage(f)
        pred = sitk.ReadImage(f"{s1}/{case}.nii.gz")
        a = sitk.GetArrayFromImage(pred)
        idx = np.argwhere(a > 0)
        shape = np.array(a.shape)
        if len(idx) == 0:
            lo, hi = np.zeros(3, int), shape          # no box -> whole volume
            fellback.append(case)
        else:
            sp = np.array(img.GetSpacing())[::-1]
            m = np.ceil(args.margin_mm / sp).astype(int)
            lo = np.maximum(idx.min(0) - m, 0)
            hi = np.minimum(idx.max(0) + 1 + m, shape)
        sl = (slice(int(lo[2]), int(hi[2])), slice(int(lo[1]), int(hi[1])),
              slice(int(lo[0]), int(hi[0])))
        sitk.WriteImage(img[sl], f"{crops}/{case}_0000.nii.gz", True)
        geom[case] = dict(lo=[int(v) for v in lo], hi=[int(v) for v in hi],
                          shape=[int(v) for v in shape], ref=f)
    if fellback:
        print(f"  WARNING: stage 1 found no foreground in {len(fellback)} case(s), "
              f"fell back to the whole volume: {fellback}")

    print(f"[3/4] segmenter: Dataset{args.stage2_dataset} / {args.stage2_trainer}")
    predict(crops, s2, args.stage2_dataset, args.stage2_trainer,
            args.gpu, args.modality, args.fold, tta=args.tta)

    print(f"[4/4] pasting back into original geometry -> {args.out}")
    for case, g in geom.items():
        roi = sitk.ReadImage(f"{s2}/{case}.nii.gz")
        ref = sitk.ReadImage(g["ref"])
        full = np.zeros(g["shape"], dtype=np.uint8)
        lo, hi = g["lo"], g["hi"]
        full[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = \
            sitk.GetArrayFromImage(roi).astype(np.uint8)
        out = sitk.GetImageFromArray(full)
        out.CopyInformation(ref)          # exact original geometry, so the
        sitk.WriteImage(out, f"{args.out}/{case}.nii.gz", True)   # evaluator
    json.dump(geom, open(f"{args.out}/cascade_geometry.json", "w"), indent=1)

    if args.keep_temp:
        print(f"temp kept at {tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"done: {len(geom)} cases -> {args.out}")


if __name__ == "__main__":
    main()
