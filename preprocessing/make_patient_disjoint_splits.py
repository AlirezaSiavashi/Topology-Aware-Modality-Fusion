#!/usr/bin/env python3
"""Rebuild Dataset104 (TopCoW joint CTA+MRA) splits so they are patient-disjoint.

TopCoW is subject-paired: topcow_ct_XXX and topcow_mr_XXX are the same patient
(verified: CoW class-set Jaccard 0.991 for true pairs vs 0.859 for random pairs).
nnU-Net's default split is per-volume, so the naive Dataset104 split puts a
patient's CTA in train and their MRA in val -- the model sees the held-out
subject's vasculature in the other modality. That inflates every joint number.

This script derives Dataset104 folds from the Dataset102 (CT) folds, which are
already patient-aligned with Dataset103 (MR), giving:

    DS104 fold k train = {ct_i, mr_i : i in DS102 fold k train}
    DS104 fold k val   = {ct_i, mr_i : i in DS102 fold k val}

so DS102 / DS103 / DS104 share one patient partition and are directly
comparable, with no subject appearing in both train and val in any modality.
"""

import argparse
import json
from pathlib import Path

PREP = Path(__file__).resolve().parent.parent / "preprocessed" / "nnUNet_preprocessed"
REF_DS = "Dataset102_TopCoW_CT"
TGT_DS = "Dataset104_TopCoW_joint"


def patient_id(case: str) -> str:
    """topcow_ct_017 -> 017"""
    return case.split("_")[-1]


def build_splits(ref_splits, joint_cases):
    by_patient = {}
    for case in joint_cases:
        by_patient.setdefault(patient_id(case), []).append(case)

    out = []
    for fold in ref_splits:
        entry = {}
        for key in ("train", "val"):
            cases = []
            for ref_case in fold[key]:
                cases.extend(by_patient.get(patient_id(ref_case), []))
            entry[key] = sorted(cases)
        out.append(entry)
    return out


def verify(splits):
    """Raise if any patient appears in both train and val of the same fold."""
    for k, fold in enumerate(splits):
        tr = {patient_id(c) for c in fold["train"]}
        va = {patient_id(c) for c in fold["val"]}
        overlap = tr & va
        if overlap:
            raise SystemExit(
                f"fold {k}: {len(overlap)} patients in both train and val: "
                f"{sorted(overlap)[:10]}"
            )
        n_ct = sum(1 for c in fold["val"] if "_ct_" in c)
        n_mr = sum(1 for c in fold["val"] if "_mr_" in c)
        print(
            f"  fold {k}: train {len(fold['train']):>3} "
            f"val {len(fold['val']):>3} (CT {n_ct}, MR {n_mr})  "
            f"patients train/val {len(tr)}/{len(va)}  leakage 0"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print only, do not write")
    args = ap.parse_args()

    ref = json.loads((PREP / REF_DS / "splits_final.json").read_text())
    tgt_path = PREP / TGT_DS / "splits_final.json"
    current = json.loads(tgt_path.read_text())
    joint_cases = sorted({c for f in current for c in f["train"] + f["val"]})

    print(f"reference: {REF_DS} ({len(ref)} folds)")
    print(f"target:    {TGT_DS} ({len(joint_cases)} cases)")

    splits = build_splits(ref, joint_cases)
    covered = {c for f in splits for c in f["train"] + f["val"]}
    missing = set(joint_cases) - covered
    if missing:
        raise SystemExit(f"cases not assigned to any fold: {sorted(missing)}")

    print("verifying patient-disjointness:")
    verify(splits)

    if args.dry_run:
        print("\ndry run - nothing written")
        return
    tgt_path.write_text(json.dumps(splits, indent=4))
    print(f"\nwrote {tgt_path}")


if __name__ == "__main__":
    main()
