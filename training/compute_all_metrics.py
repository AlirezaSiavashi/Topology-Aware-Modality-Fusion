"""
Compute Dice metrics for all zero-shot predictions.
Compares predictions against ground-truth labels in cross-domain datasets.
"""
import os
import sys
import json
import numpy as np
import nibabel as nib
from pathlib import Path

BASE = '/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset'
ZEROSHOT = os.path.join(BASE, 'results/zeroshot')
DATA = os.path.join(BASE, 'dataset/15692630')

# Label maps: 13-class TopCoW
TOPCOW_LABELS = {
    1: "BA", 2: "R-PCA", 3: "L-PCA", 4: "R-ICA", 5: "L-ICA",
    6: "R-MCA", 7: "L-MCA", 8: "R-ACA", 9: "L-ACA",
    10: "R-Pcom", 11: "L-Pcom", 12: "Acom", 13: "3rd-A2"
}

# GT label directories and filename mapping
DS_GT_INFO = {
    "DS200": {
        "gt_dir": os.path.join(DATA, "CTA_ISLES2024_TUM/cow_seg_labelsTr"),
        "pred_to_gt": lambda pred_name: pred_name.replace(".nii.gz", "_full_LPS_Mask.nii.gz"),
        "modality": "CTA",
    },
    "DS201": {
        "gt_dir": os.path.join(DATA, "MRA_IXI_HH/cow_seg_labelsTr"),
        "pred_to_gt": lambda pred_name: pred_name.replace(".nii.gz", "_full_LPS_Mask.nii.gz"),
        "modality": "MRA",
    },
    "DS202": {
        "gt_dir": os.path.join(DATA, "MRA_Lausanne/cow_seg_labelsTr"),
        "pred_to_gt": lambda pred_name: pred_name.replace(".nii.gz", "_full_LPS_Mask.nii.gz"),
        "modality": "MRA",
    },
}


def dice_score(pred, gt, label):
    p = (pred == label)
    g = (gt == label)
    if g.sum() == 0 and p.sum() == 0:
        return 1.0
    if g.sum() == 0 or p.sum() == 0:
        return 0.0
    intersection = (p & g).sum()
    return float(2.0 * intersection / (p.sum() + g.sum()))


def compute_metrics_for_pair(pred_path, gt_path, labels):
    pred_data = nib.load(str(pred_path)).get_fdata().astype(np.int32)
    gt_data = nib.load(str(gt_path)).get_fdata().astype(np.int32)

    if pred_data.shape != gt_data.shape:
        print(f"  WARNING: shape mismatch pred={pred_data.shape} gt={gt_data.shape}")
        # Try to resize gt to match pred (or vice versa)
        from scipy.ndimage import zoom
        scale = np.array(pred_data.shape) / np.array(gt_data.shape)
        if not np.allclose(scale, 1.0, atol=0.01):
            gt_data = zoom(gt_data.astype(float), scale, order=0).astype(np.int32)
            if gt_data.shape != pred_data.shape:
                print(f"  WARNING: Still mismatched after zoom, skipping")
                return None

    per_class = {}
    for label_id, label_name in labels.items():
        per_class[label_name] = dice_score(pred_data, gt_data, label_id)
    return per_class


def evaluate_folder(pred_dir, gt_info, labels):
    pred_files = sorted(Path(pred_dir).glob("*.nii.gz"))
    if not pred_files:
        return None

    gt_dir = gt_info["gt_dir"]
    pred_to_gt = gt_info["pred_to_gt"]

    results = []
    for pred_file in pred_files:
        gt_name = pred_to_gt(pred_file.name)
        gt_file = Path(gt_dir) / gt_name
        if not gt_file.exists():
            # Try direct match
            gt_file = Path(gt_dir) / pred_file.name
        if not gt_file.exists():
            print(f"  WARNING: No GT for {pred_file.name} (tried {gt_name})")
            continue

        metrics = compute_metrics_for_pair(pred_file, gt_file, labels)
        if metrics is not None:
            results.append(metrics)

    if not results:
        return None

    per_class_means = {}
    for label_name in labels.values():
        values = [r[label_name] for r in results]
        per_class_means[label_name] = float(np.mean(values))

    mean_dice = float(np.mean(list(per_class_means.values())))

    return {
        "mean_dice": mean_dice,
        "per_class": per_class_means,
        "n_subjects": len(results),
        "modality": gt_info["modality"],
    }


def main():
    all_results = {}

    for source_dir in sorted(Path(ZEROSHOT).iterdir()):
        if not source_dir.is_dir():
            continue
        source_name = source_dir.name
        all_results[source_name] = {}

        for target_dir in sorted(source_dir.iterdir()):
            if not target_dir.is_dir():
                continue
            target_name = target_dir.name

            pred_count = len(list(target_dir.glob("*.nii.gz")))
            if pred_count == 0:
                continue

            # Match target name to GT info
            gt_info = None
            for ds_key, ds_val in DS_GT_INFO.items():
                if ds_key in target_name:
                    gt_info = ds_val
                    break

            if gt_info is None:
                print(f"SKIP: No GT mapping for {source_name}/{target_name}")
                continue

            if not os.path.exists(gt_info["gt_dir"]):
                print(f"SKIP: GT dir missing: {gt_info['gt_dir']}")
                continue

            print(f"Evaluating {source_name} → {target_name} ({pred_count} cases)...")
            result = evaluate_folder(str(target_dir), gt_info, TOPCOW_LABELS)
            if result:
                all_results[source_name][target_name] = result
                print(f"  Mean Dice: {result['mean_dice']:.4f}")

    # Remove empty entries
    all_results = {k: v for k, v in all_results.items() if v}

    # Save
    out_path = os.path.join(ZEROSHOT, "metrics_summary.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved metrics to {out_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Source Model':<25} {'Target':<25} {'Mean Dice':>10} {'N':>5} {'Modality':>8}")
    print("-" * 90)
    for source, targets in sorted(all_results.items()):
        for target, metrics in sorted(targets.items()):
            print(f"{source:<25} {target:<25} {metrics['mean_dice']:>10.4f} {metrics['n_subjects']:>5} {metrics['modality']:>8}")
    print("=" * 90)


if __name__ == '__main__':
    main()
