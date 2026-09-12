#!/bin/bash
# Train DS104 with TCMN normalization ONLY (no topology losses)
# Ablation: isolates the contribution of TCMN from topology losses
# GPU 1 (student)

source /scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/nnunet_venv/bin/activate

BASE=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset

export nnUNet_raw="${BASE}/preprocessed/nnUNet_raw"
export nnUNet_preprocessed="${BASE}/preprocessed/nnUNet_preprocessed"
export nnUNet_results="${BASE}/results/nnUNet"
export PYTHONPATH="${BASE}/training:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=1

echo "=== TCMN_NoTopo training started: $(date) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 1) ==="

nnUNetv2_train 104 3d_fullres 0 -tr nnUNetTrainerTCMN_NoTopo

echo "=== TCMN_NoTopo training finished: $(date) ==="
