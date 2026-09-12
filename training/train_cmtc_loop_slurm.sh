#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=120G
#SBATCH --time=0
#SBATCH --job-name=cmtc104
#SBATCH --output=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset/training/slurm_%j.log
#SBATCH --error=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset/training/slurm_%j.log

source /scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/nnunet_venv/bin/activate

BASE=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset

export nnUNet_raw="${BASE}/preprocessed/nnUNet_raw"
export nnUNet_preprocessed="${BASE}/preprocessed/nnUNet_preprocessed"
export nnUNet_results="${BASE}/results/nnUNet"
export PYTHONPATH="${BASE}/training:${PYTHONPATH}"
export nnUNet_n_proc_DA=4

CKPT_DIR="${BASE}/results/nnUNet/Dataset104_TopCoW_joint/nnUNetTrainerCMTC__nnUNetPlans__3d_fullres/fold_0"
TARGET_EPOCH=1000

echo "=== Job started: $(date) ==="
echo "=== Node: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="

while true; do
    # Get current epoch
    CURRENT_EPOCH=$(python3 -c "
import torch
ckpt = torch.load('${CKPT_DIR}/checkpoint_latest.pth', map_location='cpu', weights_only=False)
print(ckpt['current_epoch'])
" 2>/dev/null)

    echo "=== $(date): Starting from epoch ${CURRENT_EPOCH} ==="

    if [ -n "$CURRENT_EPOCH" ] && [ "$CURRENT_EPOCH" -ge "$TARGET_EPOCH" ]; then
        echo "=== Reached target epoch ${TARGET_EPOCH}. Done! ==="
        break
    fi

    # Run training with faulthandler for stack trace on segfault
    python ${BASE}/training/run_cmtc_debug.py
    EXIT_CODE=$?

    echo "=== $(date): Training exited with code ${EXIT_CODE} ==="

    if [ $EXIT_CODE -eq 0 ]; then
        echo "=== Training completed normally ==="
        break
    fi

    # Brief pause before restart
    sleep 5
done

echo "=== Job finished: $(date) ==="
