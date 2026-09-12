"""
Zero-shot binary vessel evaluation for KiPA22.

Strategy:
  - Predictions are 13-class CoW segmentations (from TCMN or baseline)
  - GT labels are 2-class renal vessel (artery=1, vein=2)
  - Binarize both: any label > 0 = vessel foreground
  - Compute: binary Dice, binary clDice, HD95, Betti-0 error

Usage:
  python evaluate_binary_zeroshot.py \
      --pred_dir predictions/tcmn104/KiPA22 \
      --gt_dir   preprocessed/nnUNet_raw/Dataset111_KiPA22/labelsTr \
      --output   evaluation/results/KiPA22_TCMN_zeroshot \
      --num_workers 16
"""

import os
import argparse
import glob
import json
import numpy as np
import nibabel as nib
from multiprocessing import Pool
from skimage.morphology import skeletonize
from surface_distance import compute_surface_distances, compute_robust_hausdorff


def binary_dice(pred_bin, gt_bin):
    inter = (pred_bin & gt_bin).sum()
    denom = pred_bin.sum() + gt_bin.sum()
    if denom == 0:
        return float('nan')
    return float(2 * inter / denom)


def binary_cldice(pred_bin, gt_bin):
    """Centerline Dice: harmonic mean of skeleton recall and skeleton precision."""
    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return float('nan')
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return 0.0

    skel_pred = skeletonize(pred_bin)
    skel_gt   = skeletonize(gt_bin)

    # Sensitivity of gt skeleton covered by prediction
    if skel_gt.sum() > 0:
        tprec = float((skel_gt & pred_bin).sum()) / float(skel_gt.sum())
    else:
        tprec = 0.0

    # Sensitivity of pred skeleton covered by GT
    if skel_pred.sum() > 0:
        tsens = float((skel_pred & gt_bin).sum()) / float(skel_pred.sum())
    else:
        tsens = 0.0

    if tprec + tsens == 0:
        return 0.0
    return float(2 * tprec * tsens / (tprec + tsens))


def binary_hd95(pred_bin, gt_bin, spacing):
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return float('nan')
    surf = compute_surface_distances(gt_bin, pred_bin, spacing_mm=spacing)
    return float(compute_robust_hausdorff(surf, 95))


def binary_betti0(pred_bin, gt_bin):
    """
    Betti-0 error: |CC(pred) - CC(gt)| for binary masks.
    Uses connected component labeling.
    """
    from scipy.ndimage import label as scipy_label
    _, n_pred = scipy_label(pred_bin)
    _, n_gt   = scipy_label(gt_bin)
    if n_gt == 0:
        return float('nan')
    return float(abs(n_pred - n_gt))


def eval_case(args):
    pred_file, gt_file, case_id = args
    try:
        pred_nib = nib.load(pred_file)
        gt_nib   = nib.load(gt_file)

        pred = np.round(pred_nib.get_fdata()).astype(np.int32)
        gt   = np.round(gt_nib.get_fdata()).astype(np.int32)

        # Binarize both
        pred_bin = (pred > 0)
        gt_bin   = (gt > 0)

        spacing = tuple(float(s) for s in gt_nib.header.get_zooms()[:3])

        dice   = binary_dice(pred_bin, gt_bin)
        cldice = binary_cldice(pred_bin, gt_bin)
        hd95   = binary_hd95(pred_bin, gt_bin, spacing)
        betti0 = binary_betti0(pred_bin, gt_bin)

        return {
            'case_id': case_id,
            'Dice':    dice,
            'clDice':  cldice,
            'HD95':    hd95,
            'Betti0':  betti0,
        }
    except Exception as e:
        print(f"  ERROR on {case_id}: {e}")
        return {'case_id': case_id, 'Dice': float('nan'),
                'clDice': float('nan'), 'HD95': float('nan'), 'Betti0': float('nan')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir',    required=True)
    parser.add_argument('--gt_dir',      required=True)
    parser.add_argument('--output',      required=True)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    # Match predictions to GT
    pred_files = sorted(glob.glob(os.path.join(args.pred_dir, '*.nii.gz')))
    pred_files = [f for f in pred_files if not os.path.basename(f).startswith(('dataset', 'plans', 'predict'))]

    pairs = []
    for pf in pred_files:
        case_id = os.path.basename(pf).replace('.nii.gz', '')
        gt_file = os.path.join(args.gt_dir, f'{case_id}.nii.gz')
        if os.path.exists(gt_file):
            pairs.append((pf, gt_file, case_id))

    print(f"\nFound {len(pairs)} matched cases")
    print(f"Pred dir: {args.pred_dir}")
    print(f"GT dir:   {args.gt_dir}")
    print("=" * 60)

    with Pool(args.num_workers) as pool:
        results = pool.map(eval_case, pairs)

    # Print per-case
    print(f"\n{'Case':<30} {'Dice':>6} {'clDice':>7} {'HD95':>7} {'Betti0':>7}")
    print("-" * 60)
    for r in results:
        d = r['Dice']
        c = r['clDice']
        h = r['HD95']
        b = r['Betti0']
        print(f"  {r['case_id']:<28} {d:.4f}  {c:.4f}  {h:>6.1f}mm  {b:.2f}")

    # Summary
    metrics = ['Dice', 'clDice', 'HD95', 'Betti0']
    summary = {}
    print("\n" + "=" * 60)
    print("BINARY ZERO-SHOT SUMMARY")
    print("=" * 60)
    for m in metrics:
        vals = [r[m] for r in results if not np.isnan(r[m])]
        mean = float(np.mean(vals)) if vals else float('nan')
        std  = float(np.std(vals))  if vals else float('nan')
        summary[f'{m}_mean'] = mean
        summary[f'{m}_std']  = std
        unit = 'mm' if m == 'HD95' else ''
        print(f"  {m:<10}: {mean:.4f} ± {std:.4f} {unit}")

    # Save
    csv_path  = args.output + '.csv'
    json_path = args.output + '.summary.json'

    import csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['case_id', 'Dice', 'clDice', 'HD95', 'Betti0'])
        writer.writeheader()
        writer.writerows(results)

    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nPer-case CSV: {csv_path}")
    print(f"Summary JSON: {json_path}")


if __name__ == '__main__':
    main()
