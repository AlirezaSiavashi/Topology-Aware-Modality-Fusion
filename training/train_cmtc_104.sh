#!/bin/bash
# Train DS104 (joint CTA+MRA) with CMTC + TCMN + cbDice + SDF + Skeleton
# GPU 1 (PCIE A100 40GB)

source /scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/nnunet_venv/bin/activate

BASE=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset

export nnUNet_raw="${BASE}/preprocessed/nnUNet_raw"
export nnUNet_preprocessed="${BASE}/preprocessed/nnUNet_preprocessed"
export nnUNet_results="${BASE}/results/nnUNet"
export PYTHONPATH="${BASE}/training:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0

echo "=== CMTC training started: $(date) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="

nnUNetv2_train 104 3d_fullres 0 -tr nnUNetTrainerCMTC --c

echo "=== CMTC training finished: $(date) ==="
