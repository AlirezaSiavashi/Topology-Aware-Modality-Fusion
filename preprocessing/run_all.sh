#!/usr/bin/env bash
# =============================================================================
# run_all.sh
# Full preprocessing pipeline for all vessel datasets.
#
# Usage:
#   bash run_all.sh [OPTIONS]
#
# Options:
#   --tasks   brain|external|all    (default: all)
#   --workers N                     (default: 16)
#   --skip-aux                      skip SDF/skeleton/curvature generation
#   --skip-vesselness               skip Frangi vesselness maps
#   --stats-only                    only run dataset_stats.py, no preprocessing
#   --dry-run                       print commands without running them
#
# Environment variables (override paths without editing config.py):
#   VESSEL_DATASET_ROOT  – raw data root
#   VESSEL_OUTPUT_ROOT   – preprocessed output root
#   MSD_HEPATIC_ROOT     – MSD Task08 hepatic vessel folder
#   KIPA22_ROOT          – KiPA22 renal CTA folder
#   ASOCA_ROOT           – ASOCA coronary artery folder
#   DRIVE_ROOT           – DRIVE retinal vessel folder
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASKS="all"
# Default 3 workers: safe for 16 GB SLURM memory limit with aux-map generation.
# Each MRA worker peaks ~3-4 GB (volume + skeleton graph + spline buffers).
# Use --workers 8 if running --skip-aux (image+label only, ~0.5 GB/worker).
WORKERS=3
SKIP_AUX=""
SKIP_VESS=""
STATS_ONLY=0
DRY_RUN=0
VENV_PYTHON="/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/aneur/bin/python3"
PYTHON="${PYTHON:-$VENV_PYTHON}"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks)       TASKS="$2";      shift 2 ;;
        --workers)     WORKERS="$2";    shift 2 ;;
        --skip-aux)    SKIP_AUX="--skip-aux";   shift ;;
        --skip-vesselness) SKIP_VESS="--skip-vesselness"; shift ;;
        --stats-only)  STATS_ONLY=1;    shift ;;
        --dry-run)     DRY_RUN=1;       shift ;;
        *)             echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Helper: run or echo ───────────────────────────────────────────────────────
run() {
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] $*"
    else
        echo "$(date '+%H:%M:%S')  >>  $*"
        "$@"
    fi
}

# ── Check Python and dependencies ─────────────────────────────────────────────
echo "=========================================================="
echo "  Vessel Preprocessing Pipeline"
echo "  Script dir : $SCRIPT_DIR"
echo "  Python     : $PYTHON"
echo "  Tasks      : $TASKS"
echo "  Workers    : $WORKERS"
echo "  Skip aux   : ${SKIP_AUX:-no}"
echo "  Skip vess  : ${SKIP_VESS:-no}"
echo "=========================================================="

# Quick dependency check
$PYTHON -c "
import sys
missing = []
for pkg in ['SimpleITK', 'numpy', 'scipy', 'skimage', 'networkx']:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)
if missing:
    print('MISSING packages:', missing)
    print('Install with:')
    print('  pip install SimpleITK numpy scipy scikit-image networkx')
    sys.exit(1)
else:
    print('All required packages found.')
"

# ── Step 0: Inspect raw data ──────────────────────────────────────────────────
echo ""
echo "── Step 0: Raw dataset statistics ──────────────────────────"
run "$PYTHON" "$SCRIPT_DIR/dataset_stats.py" --mode raw

if [[ $STATS_ONLY -eq 1 ]]; then
    echo "Stats-only mode: done."
    exit 0
fi

# ── Step 1: Brain datasets ────────────────────────────────────────────────────
if [[ "$TASKS" == "all" || "$TASKS" == "brain" ]]; then
    echo ""
    echo "── Step 1: Brain datasets ───────────────────────────────────"

    # TopBrain only (fast; use for initial sanity check)
    echo ""
    echo "  1a. TopBrain (25 CT + 25 MR) ..."
    run "$PYTHON" "$SCRIPT_DIR/preprocess_brain.py" \
        --tasks topbrain \
        --workers "$WORKERS" \
        $SKIP_AUX $SKIP_VESS

    # TopCoW (125 CT + 125 MR — larger, main training set)
    echo ""
    echo "  1b. TopCoW (125 CT + 125 MR) ..."
    run "$PYTHON" "$SCRIPT_DIR/preprocess_brain.py" \
        --tasks topcow \
        --workers "$WORKERS" \
        $SKIP_AUX $SKIP_VESS

    # Cross-domain zero-shot test sets
    echo ""
    echo "  1c. Cross-domain subsets (ISLES, IXI, Lausanne) ..."
    run "$PYTHON" "$SCRIPT_DIR/preprocess_brain.py" \
        --tasks crossdomain \
        --workers "$WORKERS" \
        $SKIP_AUX $SKIP_VESS

    # ITKTubeTK unlabelled MRA
    echo ""
    echo "  1d. ITKTubeTK unlabelled MRA ..."
    run "$PYTHON" "$SCRIPT_DIR/preprocess_brain.py" \
        --tasks itktubetk \
        --workers "$WORKERS" \
        $SKIP_VESS   # no aux (no labels)
fi

# ── Step 2: External datasets ─────────────────────────────────────────────────
if [[ "$TASKS" == "all" || "$TASKS" == "external" ]]; then
    echo ""
    echo "── Step 2: External datasets ────────────────────────────────"
    echo "  (skipped if root paths not set — see config.py)"

    run "$PYTHON" "$SCRIPT_DIR/preprocess_external.py" \
        --tasks all \
        --workers "$WORKERS" \
        $SKIP_AUX

    # Print download instructions if any external dataset was skipped
    "$PYTHON" "$SCRIPT_DIR/preprocess_external.py" --show-downloads 2>/dev/null || true
fi

# ── Step 3: Inspect preprocessed outputs ─────────────────────────────────────
echo ""
echo "── Step 3: Preprocessed dataset statistics ──────────────────"
run "$PYTHON" "$SCRIPT_DIR/dataset_stats.py" --mode preprocessed

echo ""
echo "=========================================================="
echo "  Preprocessing complete."
echo "  Output: ${VESSEL_OUTPUT_ROOT:-<see config.py>}"
echo "=========================================================="
