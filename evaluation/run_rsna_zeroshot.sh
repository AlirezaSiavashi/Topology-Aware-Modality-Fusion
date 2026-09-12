#!/bin/bash
# Run zero-shot evaluation on RSNA 2025 dataset
# Tests: CTA baseline, CTA+topology, MRA baseline

source /scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/nnunet_venv/bin/activate

BASE=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset

export nnUNet_raw="${BASE}/preprocessed/nnUNet_raw"
export nnUNet_preprocessed="${BASE}/preprocessed/nnUNet_preprocessed"
export nnUNet_results="${BASE}/results/nnUNet"
export PYTHONPATH="${BASE}/training:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0

EVAL_SCRIPT="${BASE}/evaluation/evaluate_rsna_zeroshot.py"
PRED_BASE="${BASE}/evaluation/results/RSNA_zeroshot"

echo "============================================================"
echo "RSNA Zero-Shot Evaluation"
echo "Started: $(date)"
echo "============================================================"

# ── Step 1: Run inference for each model ─────────────────────────────────────

# CTA models → evaluated on RSNA CTA cases (74 cases)
for MODEL in cta_baseline cta_topology; do
    PRED_DIR="${PRED_BASE}/pred_${MODEL}_cta"
    if [ -d "$PRED_DIR" ] && [ "$(ls -A $PRED_DIR/*.nii.gz 2>/dev/null | wc -l)" -gt 60 ]; then
        echo "[$MODEL] Predictions already exist ($PRED_DIR), skipping inference."
    else
        echo "[$MODEL] Running inference on RSNA CTA cases..."
        python "$EVAL_SCRIPT" --model "$MODEL" --split cta \
               --num_workers 8 --perclass 2>&1 | tee "${PRED_BASE}/${MODEL}_cta.log"
    fi
done

# MRA model → evaluated on RSNA MRA cases (41 cases)
MODEL=mra_baseline
PRED_DIR="${PRED_BASE}/pred_${MODEL}_mra"
if [ -d "$PRED_DIR" ] && [ "$(ls -A $PRED_DIR/*.nii.gz 2>/dev/null | wc -l)" -gt 35 ]; then
    echo "[$MODEL] Predictions already exist ($PRED_DIR), skipping inference."
else
    echo "[$MODEL] Running inference on RSNA MRA cases..."
    python "$EVAL_SCRIPT" --model "$MODEL" --split mra \
           --num_workers 8 --perclass 2>&1 | tee "${PRED_BASE}/${MODEL}_mra.log"
fi

echo ""
echo "============================================================"
echo "All done: $(date)"
echo "Results in: $PRED_BASE"
echo "============================================================"
