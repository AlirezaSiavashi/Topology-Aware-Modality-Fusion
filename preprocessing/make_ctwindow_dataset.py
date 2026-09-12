#!/usr/bin/env python3
"""
Build Dataset105: Dataset104 with the CTA channel re-windowed, MRA untouched.

Controlled single-variable experiment. Everything -- geometry, labels, splits,
MRA intensities -- is identical to Dataset104; only the CTA intensity mapping
changes.

What this does and does not test
--------------------------------
It does NOT reduce vessel/background ambiguity. Windowing is clip + linear
rescale, i.e. monotone, and monotone maps preserve rank order, so the fraction
of background voxels brighter than the median vessel is invariant: measured at
14.99% for every candidate window from (-200,700) to (0,400). nnU-Net's
subsequent z-score is affine and equally powerless. That invariance is itself
useful -- it means the 13.9% vs 0.10% CTA/MRA ambiguity ratio is a property of
the modalities, not of a preprocessing choice.

What it DOES change is how much numerical resolution is spent on the vessel
intensity range. Under the current (-200,700) window, vessels (HU 84-307,
median 176) occupy ~25% of the range and >50% is spent on bone; median
vessel-to-brain separation is 0.177. Under (0,450) that separation is 0.355.

So the hypothesis under test is narrow: given fixed ambiguity, does allocating
more dynamic range to vessels improve CTA segmentation? A null result isolates
ambiguity -- rather than contrast scaling -- as the binding constraint.

Window choice: (0, 450) maximises vessel/brain separation among candidates
while clipping only 1.5% of vessel voxels; (0,400) separates slightly better
(0.399) but clips 2.9%, which would map the brightest vessel voxels onto the
same value as bone.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "preprocessed/nnUNet_raw"
SRC = RAW / "Dataset104_TopCoW_joint"
DST = RAW / "Dataset105_TopCoW_ctwin"

# Window Dataset104's CTA was written with (preprocessing/config.py CT_WINDOW).
OLD_LO, OLD_HI = -200.0, 700.0
NEW_LO, NEW_HI = 0.0, 450.0


def rewindow(v: np.ndarray) -> np.ndarray:
    """[0,1] under the old window -> [0,1] under the new one, via HU."""
    hu = v * (OLD_HI - OLD_LO) + OLD_LO
    return np.clip((hu - NEW_LO) / (NEW_HI - NEW_LO), 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    (DST / "imagesTr").mkdir(parents=True, exist_ok=True)
    (DST / "labelsTr").mkdir(parents=True, exist_ok=True)

    imgs = sorted((SRC / "imagesTr").glob("*.nii.gz"))
    n_ct = n_mr = 0
    for p in imgs:
        out = DST / "imagesTr" / p.name
        if args.dry_run:
            continue
        if "_ct_" in p.name:
            nii = nib.load(str(p))
            arr = rewindow(np.asarray(nii.dataobj, dtype=np.float32))
            nib.save(nib.Nifti1Image(arr.astype(np.float32), nii.affine, nii.header), out)
            n_ct += 1
        else:
            shutil.copy2(p, out)          # MRA byte-identical
            n_mr += 1

    for p in sorted((SRC / "labelsTr").glob("*.nii.gz")):
        if not args.dry_run:
            shutil.copy2(p, DST / "labelsTr" / p.name)

    ds = json.loads((SRC / "dataset.json").read_text())
    ds["name"] = "TopCoW_ctwin"
    ds["dataset_id"] = 105
    ds["description"] = (f"Dataset104 with CTA re-windowed from "
                         f"({OLD_LO:.0f},{OLD_HI:.0f}) to ({NEW_LO:.0f},{NEW_HI:.0f}) HU; "
                         f"MRA unchanged. Controlled single-variable experiment.")
    if not args.dry_run:
        (DST / "dataset.json").write_text(json.dumps(ds, indent=2))

    print(f"CTA re-windowed: {n_ct}   MRA copied unchanged: {n_mr}")
    print(f"wrote {DST}")


if __name__ == "__main__":
    main()
