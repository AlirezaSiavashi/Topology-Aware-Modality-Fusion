#!/usr/bin/env python3
"""
Compute Dice scores for zero-shot cross-domain predictions.
Resamples GT labels to prediction space before comparison.
"""
import os
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/nnunet_venv/lib/python3.9/site-packages")
import nibabel as nib
from scipy.ndimage import map_coordinates

BASE = Path("/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset")

TOPCOW_LABELS = {
    0: "Background", 1: "BA", 2: "R-PCA", 3: "L-PCA", 4: "R-ICA",
    5: "L-ICA", 6: "R-MCA", 7: "L-MCA", 8: "R-ACA", 9: "L-ACA",
    10: "R-Pcom", 11: "L-Pcom", 12: "Acom", 13: "3rd-A2",
}

EXPERIMENTS = {
    "DS104_baseline": {
        "pred_base": BASE / "results/zeroshot/DS104_baseline",
        "model": "nnUNetTrainer (baseline)",
    },
    "DS104_TCMN": {
        "pred_base": BASE / "results/zeroshot/DS104_TCMN",
        "model": "nnUNetTrainerTCMN",
    },
}

DATASETS = {
    "DS200_CTA_ISLES2024": {
        "gt_dir": BASE / "dataset/15692630/CTA_ISLES2024_TUM/cow_seg_labelsTr",
        "suffix": "_full_LPS_Mask",
        "modality": "CTA",
    },
    "DS201_MRA_IXI_HH": {
        "gt_dir": BASE / "dataset/15692630/MRA_IXI_HH/cow_seg_labelsTr",
        "suffix": "_full_LPS_Mask",
        "modality": "MRA",
    },
    "DS202_MRA_Lausanne": {
        "gt_dir": BASE / "dataset/15692630/MRA_Lausanne/cow_seg_labelsTr",
        "suffix": "_full_LPS_Mask",
        "modality": "MRA",
    },
}


def resample_label_to_target(label_nii, target_nii):
    """
    Resample a label volume to match the target volume's grid.
    Uses nearest-neighbor interpolation (appropriate for labels).
    """
    label_data = label_nii.get_fdata().astype(np.int32)
    target_shape = target_nii.shape[:3]

    # If shapes already match, no resampling needed
    if label_data.shape == target_shape and np.allclose(label_nii.affine, target_nii.affine):
        return label_data

    # Compute voxel coordinates in label space for each target voxel
    # target_voxel -> world -> label_voxel
    target_to_world = target_nii.affine
    world_to_label = np.linalg.inv(label_nii.affine)
    target_to_label = world_to_label @ target_to_world

    # Create coordinate grid in target space
    coords = np.mgrid[0:target_shape[0], 0:target_shape[1], 0:target_shape[2]]
    coords = coords.reshape(3, -1)  # (3, N)

    # Add homogeneous coordinate
    coords_h = np.vstack([coords, np.ones((1, coords.shape[1]))])

    # Transform to label space
    label_coords = target_to_label @ coords_h
    label_coords = label_coords[:3]  # (3, N)

    # Nearest-neighbor interpolation
    resampled = map_coordinates(
        label_data, label_coords, order=0, mode='constant', cval=0
    )
    resampled = resampled.reshape(target_shape).astype(np.int32)
    return resampled


def dice_score(pred: np.ndarray, gt: np.ndarray, label: int) -> float:
    pred_mask = (pred == label)
    gt_mask = (gt == label)
    intersection = np.sum(pred_mask & gt_mask)
    total = np.sum(pred_mask) + np.sum(gt_mask)
    if total == 0:
        return float('nan')
    return 2.0 * intersection / total


def compute_metrics_for_pair(pred_path: Path, gt_path: Path, labels: list) -> dict:
    pred_nii = nib.load(str(pred_path))
    gt_nii = nib.load(str(gt_path))

    pred_data = pred_nii.get_fdata().astype(np.int32)

    # Resample GT to prediction space
    gt_resampled = resample_label_to_target(gt_nii, pred_nii)

    results = {}
    for label in labels:
        results[label] = dice_score(pred_data, gt_resampled, label)
    return results


def main():
    all_results = {}

    for exp_name, exp_cfg in EXPERIMENTS.items():
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name} ({exp_cfg['model']})")
        print(f"{'='*60}")

        exp_results = {}
        for ds_name, ds_cfg in DATASETS.items():
            pred_dir = exp_cfg["pred_base"] / ds_name
            gt_dir = ds_cfg["gt_dir"]
            suffix = ds_cfg["suffix"]

            print(f"\n  --- {ds_name} ({ds_cfg['modality']}) ---")

            pred_files = sorted(pred_dir.glob("*.nii.gz"))
            if not pred_files:
                print(f"  No predictions found in {pred_dir}")
                continue

            labels = list(range(1, 14))
            per_subject = {}
            per_class_dice = defaultdict(list)

            for pred_path in pred_files:
                pred_stem = pred_path.stem.replace('.nii', '')
                gt_name = f"{pred_stem}{suffix}.nii.gz"
                gt_path = gt_dir / gt_name

                if not gt_path.exists():
                    print(f"  WARNING: GT not found for {pred_path.name}")
                    continue

                print(f"  Processing {pred_stem}...", end="", flush=True)
                scores = compute_metrics_for_pair(pred_path, gt_path, labels)
                per_subject[pred_stem] = scores

                # Compute this subject's mean dice for progress
                valid = [v for v in scores.values() if not np.isnan(v)]
                subj_mean = np.mean(valid) if valid else 0
                print(f" mean={subj_mean:.4f}")

                for label, dice in scores.items():
                    if not np.isnan(dice):
                        per_class_dice[label].append(dice)

            print(f"\n  Matched: {len(per_subject)} / {len(pred_files)} predictions")
            valid_dices = []
            for label in labels:
                dices = per_class_dice.get(label, [])
                if dices:
                    mean_dice = np.mean(dices)
                    valid_dices.append(mean_dice)
                    print(f"    Label {label:2d} ({TOPCOW_LABELS.get(label, '?'):8s}): "
                          f"Dice={mean_dice:.4f} ± {np.std(dices):.4f} (n={len(dices)})")

            if valid_dices:
                mean_all = np.mean(valid_dices)
                print(f"  >>> Mean Dice (all present classes): {mean_all:.4f}")
            else:
                mean_all = float('nan')

            exp_results[ds_name] = {
                "mean_dice": float(mean_all) if not np.isnan(mean_all) else None,
                "per_class": {
                    TOPCOW_LABELS.get(l, str(l)): float(np.mean(per_class_dice[l]))
                    for l in labels if per_class_dice.get(l)
                },
                "n_subjects": len(per_subject),
                "modality": ds_cfg["modality"],
            }

        all_results[exp_name] = exp_results

    out_path = BASE / "results/zeroshot/metrics_summary.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Experiment':<20} {'Dataset':<25} {'Mean Dice':>10} {'N':>5}")
    print("-" * 70)
    for exp_name, exp_res in all_results.items():
        for ds_name, ds_res in exp_res.items():
            dice_str = f"{ds_res['mean_dice']:.4f}" if ds_res['mean_dice'] else "N/A"
            print(f"{exp_name:<20} {ds_name:<25} {dice_str:>10} {ds_res['n_subjects']:>5}")


if __name__ == '__main__':
    main()
