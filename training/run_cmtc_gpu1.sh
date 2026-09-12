#!/bin/bash
# Run CMTC training directly on GPU 1 (no SLURM)

source /scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/nnunet_venv/bin/activate

BASE=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset

export nnUNet_raw="${BASE}/preprocessed/nnUNet_raw"
export nnUNet_preprocessed="${BASE}/preprocessed/nnUNet_preprocessed"
export nnUNet_results="${BASE}/results/nnUNet"
export PYTHONPATH="${BASE}/training:${PYTHONPATH}"
export nnUNet_n_proc_DA=4
export CUDA_VISIBLE_DEVICES=1

CKPT_DIR="${BASE}/results/nnUNet/Dataset104_TopCoW_joint/nnUNetTrainerCMTC__nnUNetPlans__3d_fullres/fold_0"
TARGET_EPOCH=1000
LOG="${BASE}/training/cmtc_gpu1_$(date +%Y%m%d_%H%M%S).log"

echo "=== CMTC Training on GPU 1 started: $(date) ===" | tee "$LOG"
echo "=== GPU: $(nvidia-smi -i 1 --query-gpu=name --format=csv,noheader 2>/dev/null) ===" | tee -a "$LOG"

while true; do
    CURRENT_EPOCH=$(python3 -c "
import torch
ckpt = torch.load('${CKPT_DIR}/checkpoint_latest.pth', map_location='cpu', weights_only=False)
print(ckpt['current_epoch'])
" 2>/dev/null)

    echo "=== $(date): Starting from epoch ${CURRENT_EPOCH} ===" | tee -a "$LOG"

    if [ -n "$CURRENT_EPOCH" ] && [ "$CURRENT_EPOCH" -ge "$TARGET_EPOCH" ]; then
        echo "=== Reached target epoch ${TARGET_EPOCH}. Done! ===" | tee -a "$LOG"
        break
    fi

    python3 ${BASE}/training/run_cmtc_debug.py 2>&1 | tee -a "$LOG"
    EXIT_CODE=${PIPESTATUS[0]}

    echo "=== $(date): Training exited with code ${EXIT_CODE} ===" | tee -a "$LOG"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "=== Training completed normally ===" | tee -a "$LOG"
        break
    fi

    sleep 5
done

echo "=== CMTC Training finished: $(date) ===" | tee -a "$LOG"
