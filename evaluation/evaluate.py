"""
Comprehensive Vessel Segmentation Evaluation Script
Computes: Dice, clDice, HD95, Betti-0 error per class
Usage:
    python evaluate.py --pred_dir <predictions> --gt_dir <ground_truth> \
                       --dataset_json <dataset.json> --output <results.csv> \
                       [--num_workers 8]
"""

import argparse
import csv
import json
import os
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import label as scipy_label
from skimage.morphology import skeletonize

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Metric implementations
# ─────────────────────────────────────────────────────────────────────────────

def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Volumetric Dice coefficient."""
    p = pred.astype(bool)
    g = gt.astype(bool)
    inter = (p & g).sum()
    denom = p.sum() + g.sum()
    if denom == 0:
        return float('nan')  # class absent in both pred and gt
    return float(2 * inter / denom)


def cldice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Centerline Dice (clDice).
    Skeletonizes both pred and gt, then measures overlap.
    Handles 3D volumes via skimage skeletonize (Lee 1994 thinning).
    Returns nan if gt is empty.
    """
    p = pred.astype(bool)
    g = gt.astype(bool)
    if g.sum() == 0:
        return float('nan')
    if p.sum() == 0:
        return 0.0

    skel_p = skeletonize(p).astype(bool)
    skel_g = skeletonize(g).astype(bool)

    # Topology precision: skeleton of pred covered by gt mask
    tp = (skel_p & g).sum()
    tprec = tp / (skel_p.sum() + 1e-8)

    # Topology sensitivity: skeleton of gt covered by pred mask
    ts = (skel_g & p).sum()
    tsens = ts / (skel_g.sum() + 1e-8)

    if tprec + tsens < 1e-8:
        return 0.0
    return float(2 * tprec * tsens / (tprec + tsens))


def hd95(pred: np.ndarray, gt: np.ndarray, voxel_spacing=(1.0, 1.0, 1.0)) -> float:
    """
    95th-percentile Hausdorff Distance in mm.
    Uses surface_distance library if available, otherwise falls back
    to scipy-based border extraction.
    """
    p = pred.astype(bool)
    g = gt.astype(bool)
    if g.sum() == 0 or p.sum() == 0:
        return float('nan')

    try:
        from surface_distance import (compute_surface_distances,
                                      compute_robust_hausdorff)
        sd = compute_surface_distances(g, p, voxel_spacing)
        return float(compute_robust_hausdorff(sd, 95))
    except Exception:
        # Fallback: sample surface points via erosion
        from scipy.ndimage import binary_erosion, distance_transform_edt
        border_p = p & ~binary_erosion(p)
        border_g = g & ~binary_erosion(g)
        if border_p.sum() == 0 or border_g.sum() == 0:
            return float('nan')
        # Distance from pred border to gt
        dt_g = distance_transform_edt(~g, sampling=voxel_spacing)
        dt_p = distance_transform_edt(~p, sampling=voxel_spacing)
        d1 = dt_g[border_p]
        d2 = dt_p[border_g]
        all_d = np.concatenate([d1, d2])
        return float(np.percentile(all_d, 95))


def betti0_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Betti-0 error: |#connected_components(pred) - #connected_components(gt)|
    Uses 26-connectivity (full 3D).
    Returns nan if gt is empty.
    """
    p = pred.astype(bool)
    g = gt.astype(bool)
    if g.sum() == 0:
        return float('nan')
    _, n_pred = scipy_label(p, structure=np.ones((3, 3, 3)))
    _, n_gt   = scipy_label(g, structure=np.ones((3, 3, 3)))
    return float(abs(n_pred - n_gt))


def betti1_error_approx(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Approximate Betti-1 (loops) error using gudhi cubical persistence.
    Falls back to nan if gudhi not available or computation fails.
    Downsamples to avoid OOM on large volumes.
    """
    try:
        import gudhi
        from skimage.transform import resize

        def count_loops(mask):
            # Downsample for speed (loops preserved under coarse downsampling)
            target = (64, 64, 64)
            m = resize(mask.astype(float), target, order=0, anti_aliasing=False)
            # Cubical complex on binary image
            cc = gudhi.CerseiFilteredCubicalComplex(
                dimensions=list(m.shape),
                top_dimensional_cells=m.flatten().tolist()
            )
            cc.compute_persistence()
            pairs = cc.persistence()
            # Count Betti-1 (dimension 1, infinite bars = actual loops)
            b1 = sum(1 for (dim, (b, d)) in pairs
                     if dim == 1 and d == float('inf'))
            return b1

        p = pred.astype(bool)
        g = gt.astype(bool)
        if g.sum() == 0:
            return float('nan')
        return float(abs(count_loops(p) - count_loops(g)))
    except Exception:
        return float('nan')


# ─────────────────────────────────────────────────────────────────────────────
# Per-case evaluation
# ─────────────────────────────────────────────────────────────────────────────

def get_voxel_spacing(nib_img):
    """Extract voxel spacing (mm) from nibabel image header."""
    hdr = nib_img.header
    zooms = hdr.get_zooms()
    return tuple(float(z) for z in zooms[:3])


def evaluate_case(args):
    """Evaluate one prediction vs ground-truth file. Returns dict of metrics."""
    pred_path, gt_path, label_map, compute_betti1 = args
    case_id = Path(pred_path).stem.replace('.nii', '')

    pred_img = nib.load(pred_path)
    gt_img   = nib.load(gt_path)

    pred_vol = np.round(pred_img.get_fdata()).astype(np.int32)
    gt_vol   = np.round(gt_img.get_fdata()).astype(np.int32)
    spacing  = get_voxel_spacing(gt_img)

    row = {'case_id': case_id}

    all_dice, all_cl, all_hd, all_b0 = [], [], [], []

    for label_id, label_name in label_map.items():
        p = (pred_vol == label_id)
        g = (gt_vol  == label_id)

        d  = dice_score(p, g)
        cl = cldice_score(p, g)
        h  = hd95(p, g, spacing)
        b0 = betti0_error(p, g)

        row[f'{label_name}_Dice']    = round(d,  4) if not np.isnan(d)  else 'nan'
        row[f'{label_name}_clDice']  = round(cl, 4) if not np.isnan(cl) else 'nan'
        row[f'{label_name}_HD95']    = round(h,  2) if not np.isnan(h)  else 'nan'
        row[f'{label_name}_Betti0']  = round(b0, 0) if not np.isnan(b0) else 'nan'

        if not np.isnan(d):  all_dice.append(d)
        if not np.isnan(cl): all_cl.append(cl)
        if not np.isnan(h):  all_hd.append(h)
        if not np.isnan(b0): all_b0.append(b0)

        if compute_betti1:
            b1 = betti1_error_approx(p, g)
            row[f'{label_name}_Betti1'] = round(b1, 0) if not np.isnan(b1) else 'nan'

    row['mean_Dice']   = round(float(np.mean(all_dice)),  4) if all_dice else 'nan'
    row['mean_clDice'] = round(float(np.mean(all_cl)),    4) if all_cl   else 'nan'
    row['mean_HD95']   = round(float(np.mean(all_hd)),    2) if all_hd   else 'nan'
    row['mean_Betti0'] = round(float(np.mean(all_b0)),    2) if all_b0   else 'nan'

    print(f"  {case_id}: Dice={row['mean_Dice']:.4f}  "
          f"clDice={row['mean_clDice']:.4f}  "
          f"HD95={row['mean_HD95']:.1f}mm  "
          f"Betti0={row['mean_Betti0']:.1f}")
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-level aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_results(rows, label_map, compute_betti1):
    """Compute mean ± std for each metric across all cases."""
    metrics = ['Dice', 'clDice', 'HD95', 'Betti0']
    if compute_betti1:
        metrics.append('Betti1')

    summary = {}
    for label_name in list(label_map.values()) + ['mean']:
        for m in metrics:
            key = f'{label_name}_{m}'
            vals = [r[key] for r in rows if key in r and r[key] != 'nan']
            if vals:
                summary[f'{key}_mean'] = round(float(np.mean(vals)), 4)
                summary[f'{key}_std']  = round(float(np.std(vals)),  4)
            else:
                summary[f'{key}_mean'] = 'nan'
                summary[f'{key}_std']  = 'nan'
    return summary


def print_summary_table(summary, label_map):
    """Print a formatted summary table to stdout."""
    header = f"\n{'Vessel':<14} {'Dice':>8} {'clDice':>8} {'HD95':>8} {'Betti0':>8}"
    print(header)
    print("-" * 50)

    all_dice_means = []
    all_cl_means   = []

    for label_name in label_map.values():
        d  = summary.get(f'{label_name}_Dice_mean',   'nan')
        cl = summary.get(f'{label_name}_clDice_mean', 'nan')
        h  = summary.get(f'{label_name}_HD95_mean',   'nan')
        b  = summary.get(f'{label_name}_Betti0_mean', 'nan')

        d_str  = f"{d:.4f}"  if d  != 'nan' else '  N/A '
        cl_str = f"{cl:.4f}" if cl != 'nan' else '  N/A '
        h_str  = f"{h:.2f}"  if h  != 'nan' else '  N/A '
        b_str  = f"{b:.2f}"  if b  != 'nan' else '  N/A '

        print(f"{label_name:<14} {d_str:>8} {cl_str:>8} {h_str:>8} {b_str:>8}")
        if d != 'nan':  all_dice_means.append(d)
        if cl != 'nan': all_cl_means.append(cl)

    print("-" * 50)
    mean_d  = summary.get('mean_Dice_mean',   'nan')
    mean_cl = summary.get('mean_clDice_mean', 'nan')
    mean_h  = summary.get('mean_HD95_mean',   'nan')
    mean_b  = summary.get('mean_Betti0_mean', 'nan')

    print(f"{'MEAN':<14} "
          f"{str(round(mean_d,4)) if mean_d!='nan' else 'N/A':>8} "
          f"{str(round(mean_cl,4)) if mean_cl!='nan' else 'N/A':>8} "
          f"{str(round(mean_h,2)) if mean_h!='nan' else 'N/A':>8} "
          f"{str(round(mean_b,2)) if mean_b!='nan' else 'N/A':>8}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def load_heldout_cases(split_file, fold):
    """Case IDs in the validation half of `fold` -- i.e. never trained on."""
    with open(split_file) as f:
        splits = json.load(f)
    if fold >= len(splits):
        raise SystemExit(f"fold {fold} not in {split_file} ({len(splits)} folds)")
    return set(splits[fold]["val"])


def find_pairs(pred_dir, gt_dir, allowed=None):
    """
    Find matching (pred, gt) file pairs.
    Handles: prediction files may have extra suffixes stripped by nnUNet.

    If `allowed` is given, cases outside it are skipped. Predicting over a whole
    dataset directory and scoring everything silently mixes training cases into
    the result -- on Dataset104 that inflated macro Dice by +0.055 (0.806
    reported vs 0.750 on genuinely held-out cases), because 101 of the 125
    scored volumes had been in the training set.
    """
    pred_files = sorted(Path(pred_dir).glob("*.nii.gz"))
    pairs = []
    skipped = 0
    for pf in pred_files:
        # nnUNet strips _0000 suffix when saving predictions
        stem = pf.name.replace(".nii.gz", "")
        if allowed is not None and stem not in allowed:
            skipped += 1
            continue
        gt_file = Path(gt_dir) / f"{stem}.nii.gz"
        if gt_file.exists():
            pairs.append((str(pf), str(gt_file)))
        else:
            print(f"  WARNING: no GT found for {pf.name}")
    if allowed is not None:
        print(f"  held-out filter: kept {len(pairs)}, skipped {skipped} "
              f"(seen in training)")
        if not pairs:
            raise SystemExit(
                "no predictions matched the held-out set -- wrong fold or "
                "wrong split file?")
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Vessel segmentation evaluation")
    parser.add_argument("--pred_dir",     required=True,
                        help="Directory with predicted .nii.gz files")
    parser.add_argument("--gt_dir",       required=True,
                        help="Directory with ground-truth .nii.gz files")
    parser.add_argument("--dataset_json", required=True,
                        help="Path to dataset.json with label map")
    parser.add_argument("--output",       required=True,
                        help="Output CSV path (e.g. results.csv)")
    parser.add_argument("--num_workers",  type=int, default=8,
                        help="Parallel workers (default: 8)")
    parser.add_argument("--betti1",       action="store_true",
                        help="Also compute Betti-1 (loops) — slow, needs gudhi")
    parser.add_argument("--no_hd95",      action="store_true",
                        help="Skip HD95 computation (faster)")
    parser.add_argument("--split_file",   default=None,
                        help="splits_final.json; restrict scoring to the "
                             "held-out cases of --fold. Use this for any "
                             "in-domain number, or training cases leak in.")
    parser.add_argument("--fold",         type=int, default=0,
                        help="Fold whose val split defines the held-out set")
    parser.add_argument("--allow_full_set", action="store_true",
                        help="Score every prediction found, with no held-out "
                             "filter. Only correct for a truly external test "
                             "set (zero-shot), never for the training dataset.")
    args = parser.parse_args()

    if args.split_file is None and not args.allow_full_set:
        raise SystemExit(
            "refusing to score without --split_file: pass --split_file/--fold "
            "to restrict to held-out cases, or --allow_full_set if this is an "
            "external zero-shot dataset the model never trained on.")

    # Load label map
    with open(args.dataset_json) as f:
        ds = json.load(f)
    label_map = {int(v): k for k, v in ds['labels'].items()
                 if k != 'background'}
    label_map = dict(sorted(label_map.items()))

    print(f"\n{'='*60}")
    print(f"Evaluating: {args.pred_dir}")
    print(f"GT:         {args.gt_dir}")
    print(f"Labels:     {list(label_map.values())}")
    print(f"{'='*60}\n")

    # Find pairs
    allowed = (load_heldout_cases(args.split_file, args.fold)
               if args.split_file else None)
    pairs = find_pairs(args.pred_dir, args.gt_dir, allowed=allowed)
    if not pairs:
        print("ERROR: No matching prediction/GT pairs found.")
        return
    print(f"Found {len(pairs)} cases\n")

    # Build args list for parallel workers
    worker_args = [(p, g, label_map, args.betti1) for p, g in pairs]

    # Run evaluation
    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            rows = pool.map(evaluate_case, worker_args)
    else:
        rows = [evaluate_case(a) for a in worker_args]

    # Aggregate
    summary = aggregate_results(rows, label_map, args.betti1)

    # Print summary table
    print_summary_table(summary, label_map)

    # Save per-case CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_keys = list(rows[0].keys())
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)

    # Save summary JSON
    summary_path = output_path.with_suffix('.summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nPer-case results saved to:  {output_path}")
    print(f"Summary stats saved to:     {summary_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
