#!/bin/bash
# Run DS104 baseline (nnUNetTrainer, no TCMN) inference on DS103 MR and DS102 CT
# This is the CRITICAL missing ablation: isolates TCMN contribution

source /scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/nnunet_venv/bin/activate

BASE=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset
EVAL_DIR="${BASE}/evaluation/results"

export nnUNet_raw="${BASE}/preprocessed/nnUNet_raw"
export nnUNet_preprocessed="${BASE}/preprocessed/nnUNet_preprocessed"
export nnUNet_results="${BASE}/results/nnUNet"
export PYTHONPATH="${BASE}/training:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=1

echo "============================================================"
echo "DS104 Baseline Evaluation (no TCMN)"
echo "Started: $(date)"
echo "============================================================"

# --- Step 1: DS104 baseline → DS103 MR (CTA→MRA zero-shot) ---
PRED_DIR="${EVAL_DIR}/E_joint_baseline_DS104_2_DS103"
if [ -d "$PRED_DIR" ] && [ "$(ls -A $PRED_DIR/*.nii.gz 2>/dev/null | wc -l)" -gt 100 ]; then
    echo "[DS104→DS103] Predictions exist, skipping inference."
else
    echo "[DS104→DS103] Running inference..."
    mkdir -p "$PRED_DIR"
    nnUNetv2_predict \
        -d 104 \
        -i "${nnUNet_raw}/Dataset103_TopCoW_MR/imagesTr" \
        -o "$PRED_DIR" \
        -f 0 \
        -tr nnUNetTrainer \
        -c 3d_fullres
fi

echo "[DS104→DS103] Running evaluation..."
python "${BASE}/evaluation/evaluate.py" \
    --pred_dir "$PRED_DIR" \
    --gt_dir "${nnUNet_raw}/Dataset103_TopCoW_MR/labelsTr" \
    --dataset_json "${nnUNet_raw}/Dataset103_TopCoW_MR/dataset.json" \
    --output "${EVAL_DIR}/E_joint_baseline_DS104_2_DS103" \
    --num_workers 8 2>&1

echo ""

# --- Step 2: DS104 baseline → DS102 CT (MRA→CTA zero-shot) ---
PRED_DIR2="${EVAL_DIR}/E_joint_baseline_DS104_2_DS102"
if [ -d "$PRED_DIR2" ] && [ "$(ls -A $PRED_DIR2/*.nii.gz 2>/dev/null | wc -l)" -gt 100 ]; then
    echo "[DS104→DS102] Predictions exist, skipping inference."
else
    echo "[DS104→DS102] Running inference..."
    mkdir -p "$PRED_DIR2"
    nnUNetv2_predict \
        -d 104 \
        -i "${nnUNet_raw}/Dataset102_TopCoW_CT/imagesTr" \
        -o "$PRED_DIR2" \
        -f 0 \
        -tr nnUNetTrainer \
        -c 3d_fullres
fi

echo "[DS104→DS102] Running evaluation..."
python "${BASE}/evaluation/evaluate.py" \
    --pred_dir "$PRED_DIR2" \
    --gt_dir "${nnUNet_raw}/Dataset102_TopCoW_CT/labelsTr" \
    --dataset_json "${nnUNet_raw}/Dataset102_TopCoW_CT/dataset.json" \
    --output "${EVAL_DIR}/E_joint_baseline_DS104_2_DS102" \
    --num_workers 8 2>&1

echo ""
echo "============================================================"
echo "All done: $(date)"
echo "============================================================"
