"""
analyze_zeroshot.py
====================
Quantitative analysis of zero-shot predictions (no ground truth).

Metrics computed per model per dataset:
  1. Mean predicted vessel volume (voxels) — proxy for coverage
  2. Class detection rate — fraction of 13 classes predicted in each case
  3. Per-class detection frequency — which classes each model finds
  4. Volume coefficient of variation — consistency across cases

Usage:
    python3 analyze_zeroshot.py
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
from collections import defaultdict

BASE = Path("/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset/results/zeroshot_eval")

MODELS   = ["E0", "E1", "TCMN"]
DATASETS = ["IXI", "LAU"]
NUM_CLASSES = 13  # classes 1-13

def analyze_folder(folder: Path):
    """Return per-case stats from a prediction folder."""
    files = sorted(folder.glob("*.nii.gz"))
    if not files:
        return None

    volumes, class_counts, class_presence = [], [], defaultdict(int)
    for f in files:
        arr = nib.load(f).get_fdata().astype(np.int32)
        total_vessel = int((arr > 0).sum())
        volumes.append(total_vessel)

        detected = set(np.unique(arr)) - {0}
        class_counts.append(len(detected))
        for c in detected:
            class_presence[c] += 1

    n = len(files)
    return {
        "n_cases":     n,
        "vol_mean":    np.mean(volumes),
        "vol_std":     np.std(volumes),
        "vol_cv":      np.std(volumes) / (np.mean(volumes) + 1e-6),
        "cls_mean":    np.mean(class_counts),
        "cls_std":     np.std(class_counts),
        "class_freq":  {c: class_presence[c] / n for c in range(1, NUM_CLASSES + 1)},
    }

print("=" * 70)
print("ZERO-SHOT PREDICTION ANALYSIS — IXI & Lausanne MRA")
print("=" * 70)

# Per-dataset per-model summary
for ds in DATASETS:
    ds_name = "IXI HH" if ds == "IXI" else "Lausanne"
    print(f"\n{'─'*70}")
    print(f"Dataset: {ds_name}")
    print(f"{'─'*70}")
    print(f"{'Model':<8} {'Cases':>6} {'Vol mean':>10} {'Vol CV':>8} {'Classes':>10}")
    print(f"{'─'*42}")

    results = {}
    for model in MODELS:
        folder = BASE / f"{model}_{ds}"
        if not folder.exists():
            print(f"{model:<8}  NOT DONE YET")
            continue
        stats = analyze_folder(folder)
        if stats is None:
            print(f"{model:<8}  EMPTY")
            continue
        results[model] = stats
        print(f"{model:<8} {stats['n_cases']:>6} {stats['vol_mean']:>10.0f} "
              f"{stats['vol_cv']:>8.3f} {stats['cls_mean']:>8.1f}±{stats['cls_std']:.1f}")

    # Class detection frequency comparison
    if len(results) >= 2:
        print(f"\nClass detection frequency (fraction of cases where class detected):")
        print(f"{'Class':<8}", end="")
        for m in MODELS:
            if m in results:
                print(f"{m:>10}", end="")
        print()
        for c in range(1, NUM_CLASSES + 1):
            print(f"  {c:<6}", end="")
            for m in MODELS:
                if m in results:
                    freq = results[m]["class_freq"].get(c, 0.0)
                    print(f"{freq:>10.2f}", end="")
            print()

        # TCMN vs E1 improvement
        if "TCMN" in results and "E1" in results:
            print(f"\nTCMN vs E1 class detection delta:")
            improvements = []
            for c in range(1, NUM_CLASSES + 1):
                delta = results["TCMN"]["class_freq"].get(c, 0) - results["E1"]["class_freq"].get(c, 0)
                improvements.append((c, delta))
                if abs(delta) > 0.05:
                    sign = "+" if delta > 0 else ""
                    print(f"  Class {c}: {sign}{delta:.2f}")
            avg_delta = np.mean([d for _, d in improvements])
            print(f"  Average delta: {avg_delta:+.3f}")

print(f"\n{'='*70}")
print("Summary: TCMN should show higher class detection on MRA zero-shot")
print("if domain conditioning generalizes better to unseen MRA scanners.")
print("=" * 70)
