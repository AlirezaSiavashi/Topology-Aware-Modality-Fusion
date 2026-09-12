#!/bin/bash
# Run the M3-CoW ablation matrix sequentially on one GPU.
#   GPU=0 ARMS="nnUNetTrainerM3CoW nnUNetTrainerM3CoW_NoStateLoss" ./run_m3cow_matrix.sh
#
# Each arm runs to completion before the next starts, so two GPUs = two lanes.
# Suggested split (the two headline arms first, they answer the central claim):
#   lane A (GPU 0): M3CoW, NoComplex, NoTrapezoid, CTonly, Adapt_n5
#   lane B (GPU 1): NoStateLoss, NoComplexNoGeom, NoAlign, NoDyn, NoTransformer
set -euo pipefail
GPU=${GPU:-0}; FOLD=${FOLD:-0}; DATASET=${DATASET:-104}
ARMS=${ARMS:-"nnUNetTrainerM3CoW nnUNetTrainerM3CoW_NoStateLoss"}
BASE="/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset"
cd "$BASE"
for A in $ARMS; do
    echo "=== [$(date +%H:%M:%S)] starting $A on GPU $GPU ==="
    TRAINER="$A" GPU="$GPU" FOLD="$FOLD" DATASET="$DATASET" ./run_m3cow.sh
    PID=$(pgrep -n -f "run_training $DATASET 3d_fullres $FOLD -tr $A") || true
    [ -n "${PID:-}" ] && while kill -0 "$PID" 2>/dev/null; do sleep 60; done
    echo "=== [$(date +%H:%M:%S)] finished $A ==="
    tail -4 "logs/${A}_f${FOLD}.log" || true
done
echo "=== matrix lane done $(date) ==="
