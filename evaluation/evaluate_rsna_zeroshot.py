"""
evaluate_rsna_zeroshot.py
=========================
Zero-shot evaluation of our TopCoW-trained models on the RSNA 2025
Intracranial Aneurysm Detection dataset.

Strategy
--------
- RSNA Dataset001 has 178 labeled cases (74 CTA, 41 MRA, rest MRI T1/T2)
- Our models are trained on TopCoW (13-class CoW segmentation)
- Both datasets share the same Circle of Willis anatomy but use DIFFERENT
  label indices. We remap predictions before computing metrics.

Label remapping  (TopCoW pred → RSNA GT)
-----------------------------------------
TopCoW → RSNA
  1 BA          → 2  (Basilar Tip)
  2 R-PCA       → 1  (Other Posterior Circulation — closest match)
  3 L-PCA       → 1  (Other Posterior Circulation — closest match)
  4 R-ICA       → 5  (Right Infraclinoid ICA) + 7 (Right Supraclinoid ICA)
  5 R-MCA       → 9  (Right Middle Cerebral Artery)
  6 L-ICA       → 6  (Left Infraclinoid ICA) + 8 (Left Supraclinoid ICA)
  7 L-MCA       → 10 (Left Middle Cerebral Artery)
  8 R-Pcom      → 3  (Right Posterior Communicating Artery)
  9 L-Pcom      → 4  (Left Posterior Communicating Artery)
 10 Acom        → 13 (Anterior Communicating Artery)
 11 R-ACA       → 11 (Right Anterior Cerebral Artery)
 12 L-ACA       → 12 (Left Anterior Cerebral Artery)
 13 3rd-A2      → 11 (Right Anterior Cerebral Artery — ACA variant)

For the GT side, RSNA infra/supraclinoid ICA is merged into a single ICA
class (matching TopCoW's single R-ICA / L-ICA classes).

Evaluation approach
-------------------
1. BINARY (merged): binarize both pred and GT → Dice, clDice, HD95, Betti-0
   Best for overall vessel detection quality across both CTA and MRA.

2. PER-CLASS (remapped): evaluate each matched vessel class separately.
   Shows which vessel classes benefit from topology training.

Models evaluated
----------------
  E1: DS102 nnUNetTrainer (CTA baseline, no topology)     → tested on CTA
  E2: DS102 nnUNetTrainerTopologyVessel (CTA + topology)  → tested on CTA
  E3: DS103 nnUNetTrainer (MRA baseline, no topology)     → tested on MRA
  (E4: DS103 TCMN available if trained — skipped here)

Usage
-----
  python evaluate_rsna_zeroshot.py --model cta_baseline  --split cta
  python evaluate_rsna_zeroshot.py --model cta_topology  --split cta
  python evaluate_rsna_zeroshot.py --model mra_baseline  --split mra

  # Run all at once:
  python evaluate_rsna_zeroshot.py --run_all
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import nibabel as nib

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

RSNA_BASE = Path("/scratch/siyavash/Alireza_thesis/external_dataset"
                 "/rsna-intracranial-aneurysm-detection"
                 "/rsna2025_1st_place/data")

BRAIN_BASE = Path("/scratch/siyavash/Alireza_thesis/external_dataset"
                  "/rsna-intracranial-aneurysm-detection"
                  "/brain_external_dataset")

NNUNET_RESULTS = BRAIN_BASE / "results" / "nnUNet"
RSNA_IMAGES_TR = RSNA_BASE / "nnUNet" / "nnUNet_raw" / "Dataset001_VesselSegmentation" / "imagesTr"
RSNA_LABELS_TR = RSNA_BASE / "nnUNet" / "nnUNet_raw" / "Dataset001_VesselSegmentation" / "labelsTr"
RSNA_TRAIN_CSV = RSNA_BASE / "train.csv"

EVAL_OUTPUT = BRAIN_BASE / "evaluation" / "results" / "RSNA_zeroshot"
EVAL_OUTPUT.mkdir(parents=True, exist_ok=True)

VENV_PYTHON = ("/scratch/siyavash/Alireza_thesis/external_dataset"
               "/rsna-intracranial-aneurysm-detection/nnunet_venv/bin/python3.9")

# ─────────────────────────────────────────────────────────────────────────────
# Label mapping
# ─────────────────────────────────────────────────────────────────────────────

# Maps TopCoW label index → RSNA label index
# Multiple TopCoW labels can map to the same RSNA label
TOPCOW_TO_RSNA = {
    1:  2,   # BA         → Basilar Tip
    2:  1,   # R-PCA      → Other Posterior Circulation
    3:  1,   # L-PCA      → Other Posterior Circulation
    4:  5,   # R-ICA      → Right Infraclinoid ICA (infra+supra merged into infra)
    5:  9,   # R-MCA      → Right MCA
    6:  6,   # L-ICA      → Left Infraclinoid ICA
    7:  10,  # L-MCA      → Left MCA
    8:  3,   # R-Pcom     → Right Posterior Communicating
    9:  4,   # L-Pcom     → Left Posterior Communicating
    10: 13,  # Acom       → Anterior Communicating
    11: 11,  # R-ACA      → Right Anterior Cerebral
    12: 12,  # L-ACA      → Left Anterior Cerebral
    13: 11,  # 3rd-A2     → Right Anterior Cerebral (ACA variant)
}

# For GT: merge RSNA infra+supra ICA into single class
# RSNA 5 (R-infra) + 7 (R-supra) → merged as 5
# RSNA 6 (L-infra) + 8 (L-supra) → merged as 6
RSNA_GT_MERGE = {
    7: 5,   # R-supra ICA → R-ICA
    8: 6,   # L-supra ICA → L-ICA
}

# Classes to evaluate per-class (RSNA label IDs, after merging)
EVAL_CLASSES = {
    1:  'Other_Post_Circ',
    2:  'BA',
    3:  'R_Pcom',
    4:  'L_Pcom',
    5:  'R_ICA',
    6:  'L_ICA',
    9:  'R_MCA',
    10: 'L_MCA',
    11: 'R_ACA',
    12: 'L_ACA',
    13: 'Acom',
}

# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────

MODELS = {
    'cta_baseline': {
        'dataset_id': '102',
        'trainer':    'nnUNetTrainer',
        'modality':   'CTA',
        'desc':       'DS102 CTA baseline (no topology)',
    },
    'cta_topology': {
        'dataset_id': '102',
        'trainer':    'nnUNetTrainerTopologyVessel',
        'modality':   'CTA',
        'desc':       'DS102 CTA + topology losses (cbDice+SDF+skel)',
    },
    'mra_baseline': {
        'dataset_id': '103',
        'trainer':    'nnUNetTrainer',
        'modality':   'MRA',
        'desc':       'DS103 MRA baseline (no topology)',
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Metric functions (same as evaluate.py)
# ─────────────────────────────────────────────────────────────────────────────

def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred.astype(bool)
    g = gt.astype(bool)
    inter = (p & g).sum()
    denom = p.sum() + g.sum()
    if denom == 0:
        return float('nan')
    return float(2 * inter / denom)


def cldice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    from skimage.morphology import skeletonize
    p = pred.astype(bool)
    g = gt.astype(bool)
    if g.sum() == 0:
        return float('nan')
    if p.sum() == 0:
        return 0.0
    skel_p = skeletonize(p).astype(bool)
    skel_g = skeletonize(g).astype(bool)
    tprec = (skel_p & g).sum() / (skel_p.sum() + 1e-8)
    tsens = (skel_g & p).sum() / (skel_g.sum() + 1e-8)
    if tprec + tsens < 1e-8:
        return 0.0
    return float(2 * tprec * tsens / (tprec + tsens))


def hd95(pred: np.ndarray, gt: np.ndarray, spacing=(1., 1., 1.)) -> float:
    from surface_distance import compute_surface_distances, compute_robust_hausdorff
    p = pred.astype(bool)
    g = gt.astype(bool)
    if g.sum() == 0 or p.sum() == 0:
        return float('nan')
    try:
        sd = compute_surface_distances(g, p, spacing)
        return float(compute_robust_hausdorff(sd, 95))
    except Exception:
        return float('nan')


def betti0_error(pred: np.ndarray, gt: np.ndarray) -> float:
    from scipy.ndimage import label as scipy_label
    p = pred.astype(bool)
    g = gt.astype(bool)
    if g.sum() == 0:
        return float('nan')
    _, n_pred = scipy_label(p, structure=np.ones((3, 3, 3)))
    _, n_gt   = scipy_label(g, structure=np.ones((3, 3, 3)))
    return float(abs(n_pred - n_gt))


# ─────────────────────────────────────────────────────────────────────────────
# Prediction → RSNA label space conversion
# ─────────────────────────────────────────────────────────────────────────────

def remap_topcow_pred_to_rsna(pred: np.ndarray) -> np.ndarray:
    """
    Convert TopCoW 13-class prediction to RSNA label space.
    Applies TOPCOW_TO_RSNA mapping.
    """
    out = np.zeros_like(pred)
    for tc_label, rsna_label in TOPCOW_TO_RSNA.items():
        out[pred == tc_label] = rsna_label
    return out


def remap_rsna_gt_merge_ica(gt: np.ndarray) -> np.ndarray:
    """
    Merge RSNA infra/supra ICA classes into single ICA per side.
    7 → 5 (R-supra → R-ICA), 8 → 6 (L-supra → L-ICA)
    """
    out = gt.copy()
    for src, dst in RSNA_GT_MERGE.items():
        out[gt == src] = dst
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Run nnUNet inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(model_key: str, case_ids: list, tmp_dir: Path) -> Path:
    """
    Run nnUNetv2_predict on the given cases.
    Returns path to directory containing predictions.
    """
    cfg = MODELS[model_key]
    ds_id  = cfg['dataset_id']
    trainer = cfg['trainer']

    pred_dir = tmp_dir / f"pred_{model_key}"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Build input dir with symlinks to requested cases
    inp_dir = tmp_dir / f"inp_{model_key}"
    inp_dir.mkdir(parents=True, exist_ok=True)

    for uid in case_ids:
        src = RSNA_IMAGES_TR / f"{uid}_0000.nii.gz"
        dst = inp_dir / f"{uid}_0000.nii.gz"
        if src.exists() and not dst.exists():
            dst.symlink_to(src)

    env = os.environ.copy()
    env['nnUNet_raw']          = str(BRAIN_BASE / "preprocessed" / "nnUNet_raw")
    env['nnUNet_preprocessed'] = str(BRAIN_BASE / "preprocessed" / "nnUNet_preprocessed")
    env['nnUNet_results']      = str(NNUNET_RESULTS)
    env['CUDA_VISIBLE_DEVICES'] = '0'

    cmd = [
        "nnUNetv2_predict",
        "-d", ds_id,
        "-i", str(inp_dir),
        "-o", str(pred_dir),
        "-f", "0",
        "-tr", trainer,
        "-c", "3d_fullres",
        "--save_probabilities",
    ]
    # Remove --save_probabilities for faster runs
    cmd = [c for c in cmd if c != "--save_probabilities"]

    print(f"\n[Inference] Model: {model_key} | Cases: {len(case_ids)}")
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=False)
    if result.returncode != 0:
        print(f"  WARNING: inference returned code {result.returncode}")

    return pred_dir


# ─────────────────────────────────────────────────────────────────────────────
# Per-case evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_case(args):
    uid, pred_dir, compute_perclass = args
    pred_file = pred_dir / f"{uid}.nii.gz"
    gt_file   = RSNA_LABELS_TR / f"{uid}.nii.gz"

    if not pred_file.exists():
        print(f"  MISSING pred: {uid}")
        return None
    if not gt_file.exists():
        print(f"  MISSING gt:   {uid}")
        return None

    pred_nib = nib.load(pred_file)
    gt_nib   = nib.load(gt_file)
    spacing  = tuple(float(s) for s in gt_nib.header.get_zooms()[:3])

    pred_raw = np.round(pred_nib.get_fdata()).astype(np.int32)
    gt_raw   = np.round(gt_nib.get_fdata()).astype(np.int32)

    # Remap
    pred_rsna = remap_topcow_pred_to_rsna(pred_raw)
    gt_rsna   = remap_rsna_gt_merge_ica(gt_raw)

    row = {'case_id': uid}

    # ── Binary evaluation ──────────────────────────────────────────────────
    pred_bin = (pred_rsna > 0)
    gt_bin   = (gt_rsna > 0)

    row['bin_Dice']   = dice_score(pred_bin, gt_bin)
    row['bin_clDice'] = cldice_score(pred_bin, gt_bin)
    row['bin_HD95']   = hd95(pred_bin, gt_bin, spacing)
    row['bin_Betti0'] = betti0_error(pred_bin, gt_bin)

    print(f"  {uid[:40]}  Dice={row['bin_Dice']:.4f}  "
          f"clDice={row['bin_clDice']:.4f}  HD95={row['bin_HD95']:.1f}mm  "
          f"Betti0={row['bin_Betti0']:.1f}")

    if not compute_perclass:
        return row

    # ── Per-class evaluation ───────────────────────────────────────────────
    for cls_id, cls_name in EVAL_CLASSES.items():
        p = (pred_rsna == cls_id)
        g = (gt_rsna   == cls_id)
        d  = dice_score(p, g)
        cl = cldice_score(p, g)
        h  = hd95(p, g, spacing)
        b  = betti0_error(p, g)
        row[f'{cls_name}_Dice']   = round(d,  4) if not np.isnan(d)  else 'nan'
        row[f'{cls_name}_clDice'] = round(cl, 4) if not np.isnan(cl) else 'nan'
        row[f'{cls_name}_HD95']   = round(h,  2) if not np.isnan(h)  else 'nan'
        row[f'{cls_name}_Betti0'] = round(b,  1) if not np.isnan(b)  else 'nan'

    return row


# ─────────────────────────────────────────────────────────────────────────────
# Summary printing & saving
# ─────────────────────────────────────────────────────────────────────────────

def summarise(rows: list, model_key: str, split: str, pred_dir: Path):
    valid = [r for r in rows if r is not None]
    if not valid:
        print("No valid cases.")
        return

    out_prefix = EVAL_OUTPUT / f"{model_key}_{split}"

    # ── Binary summary ──────────────────────────────────────────────────────
    bin_metrics = ['bin_Dice', 'bin_clDice', 'bin_HD95', 'bin_Betti0']
    print(f"\n{'='*60}")
    print(f"BINARY ZERO-SHOT SUMMARY  model={model_key}  split={split}  n={len(valid)}")
    print(f"{'='*60}")
    summary = {}
    for m in bin_metrics:
        vals = [r[m] for r in valid if m in r and not np.isnan(r[m])]
        mean = float(np.mean(vals)) if vals else float('nan')
        std  = float(np.std(vals))  if vals else float('nan')
        summary[f'{m}_mean'] = round(mean, 4)
        summary[f'{m}_std']  = round(std,  4)
        unit = 'mm' if 'HD95' in m else ''
        print(f"  {m:<15}: {mean:.4f} ± {std:.4f} {unit}")

    # ── Per-class summary ──────────────────────────────────────────────────
    has_perclass = any(f'{list(EVAL_CLASSES.values())[0]}_Dice' in r for r in valid)
    if has_perclass:
        print(f"\n{'Vessel':<22} {'Dice':>7} {'clDice':>8} {'HD95':>8} {'Betti0':>8}")
        print('-' * 58)
        all_dice, all_cl, all_hd = [], [], []
        for cls_id, cls_name in EVAL_CLASSES.items():
            d_vals  = [r.get(f'{cls_name}_Dice')   for r in valid]
            cl_vals = [r.get(f'{cls_name}_clDice')  for r in valid]
            h_vals  = [r.get(f'{cls_name}_HD95')    for r in valid]
            d_vals  = [v for v in d_vals  if v is not None and v != 'nan']
            cl_vals = [v for v in cl_vals if v is not None and v != 'nan']
            h_vals  = [v for v in h_vals  if v is not None and v != 'nan']
            md  = float(np.mean(d_vals))  if d_vals  else float('nan')
            mcl = float(np.mean(cl_vals)) if cl_vals else float('nan')
            mh  = float(np.mean(h_vals))  if h_vals  else float('nan')
            print(f"  {cls_name:<20} {md:>7.4f} {mcl:>8.4f} {mh:>8.2f}")
            if not np.isnan(md):  all_dice.append(md)
            if not np.isnan(mcl): all_cl.append(mcl)
            if not np.isnan(mh):  all_hd.append(mh)
            summary[f'{cls_name}_Dice_mean']  = round(md,  4)
            summary[f'{cls_name}_clDice_mean'] = round(mcl, 4)
            summary[f'{cls_name}_HD95_mean']   = round(mh,  2)

        print('-' * 58)
        print(f"  {'MEAN':<20} {np.mean(all_dice):>7.4f} "
              f"{np.mean(all_cl):>8.4f} {np.mean(all_hd):>8.2f}")

    # Save CSV
    csv_path  = str(out_prefix) + '.csv'
    json_path = str(out_prefix) + '.summary.json'
    all_keys  = list(valid[0].keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(valid)

    with open(json_path, 'w') as f:
        json.dump({'model': model_key, 'split': split, 'n_cases': len(valid),
                   **summary}, f, indent=2)

    print(f"\nCSV:  {csv_path}")
    print(f"JSON: {json_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def get_case_ids_for_split(split: str) -> list:
    """Return case UIDs filtered by modality split (cta / mra / all)."""
    import csv as _csv
    modality_filter = {'cta': 'CTA', 'mra': 'MRA',
                       'mra_t2': 'MRI T2', 'mra_t1': 'MRI T1post'}
    with open(RSNA_TRAIN_CSV) as f:
        reader = _csv.DictReader(f)
        rows = list(reader)

    label_ids = {p.name.replace('.nii.gz', '')
                 for p in RSNA_LABELS_TR.iterdir() if p.suffix == '.gz'}

    if split == 'all':
        keep_modalities = {'CTA', 'MRA'}
    else:
        keep_modalities = {modality_filter.get(split, split.upper())}

    return [r['SeriesInstanceUID'] for r in rows
            if r['SeriesInstanceUID'] in label_ids
            and r['Modality'] in keep_modalities]


def run_model(model_key: str, split: str, skip_inference: bool,
              pred_dir_override: str | None, num_workers: int,
              perclass: bool):
    print(f"\n{'#'*60}")
    print(f"# MODEL: {model_key}  SPLIT: {split}")
    print(f"# {MODELS[model_key]['desc']}")
    print(f"{'#'*60}")

    case_ids = get_case_ids_for_split(split)
    print(f"Cases for split '{split}': {len(case_ids)}")

    if pred_dir_override:
        pred_dir = Path(pred_dir_override)
    elif skip_inference:
        pred_dir = EVAL_OUTPUT / f"pred_{model_key}_{split}"
        print(f"Skipping inference, using: {pred_dir}")
    else:
        with tempfile.TemporaryDirectory(prefix='rsna_zeroshot_') as tmp:
            pred_dir_tmp = run_inference(model_key, case_ids, Path(tmp))
            # Copy predictions to a persistent location
            pred_dir = EVAL_OUTPUT / f"pred_{model_key}_{split}"
            if pred_dir.exists():
                shutil.rmtree(pred_dir)
            shutil.copytree(pred_dir_tmp, pred_dir)

    # Evaluate
    worker_args = [(uid, pred_dir, perclass) for uid in case_ids]
    if num_workers > 1:
        with Pool(num_workers) as pool:
            rows = pool.map(evaluate_case, worker_args)
    else:
        rows = [evaluate_case(a) for a in worker_args]

    summarise(rows, model_key, split, pred_dir)


def main():
    parser = argparse.ArgumentParser(description="RSNA zero-shot evaluation")
    parser.add_argument('--model',    choices=list(MODELS.keys()),
                        help='Model to evaluate')
    parser.add_argument('--split',    default='cta',
                        choices=['cta', 'mra', 'all'],
                        help='Modality split to evaluate on')
    parser.add_argument('--run_all',  action='store_true',
                        help='Run all model+split combinations')
    parser.add_argument('--skip_inference', action='store_true',
                        help='Skip inference, use existing predictions')
    parser.add_argument('--pred_dir', default=None,
                        help='Override prediction directory')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--perclass',    action='store_true',
                        help='Also compute per-class metrics (slower)')
    args = parser.parse_args()

    if args.run_all:
        pairs = [
            ('cta_baseline', 'cta'),
            ('cta_topology',  'cta'),
            ('mra_baseline', 'mra'),
        ]
        for model_key, split in pairs:
            run_model(model_key, split,
                      args.skip_inference, None,
                      args.num_workers, args.perclass)
    elif args.model:
        run_model(args.model, args.split,
                  args.skip_inference, args.pred_dir,
                  args.num_workers, args.perclass)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
