#!/bin/bash
#SBATCH --job-name=nnunet_tv
#SBATCH --partition=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=192:00:00
#SBATCH --output=logs/slurm_nnunet_%j.log
#SBATCH --error=logs/slurm_nnunet_%j.log

# Usage: sbatch --export=FOLD=1 submit_nnunet_fold.sh
# FOLD must be set via --export (1, 2, 3, or 4)

set -euo pipefail

FOLD=${FOLD:-1}

VENV="/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/nnunet_venv"
BASE="/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset"
PY="$VENV/bin/python3.9"

export nnUNet_raw="${BASE}/preprocessed/nnUNet_raw"
export nnUNet_preprocessed="${BASE}/preprocessed/nnUNet_preprocessed"
export nnUNet_results="${BASE}/results/nnUNet"
export PATH="$VENV/bin:$PATH"
export PIP_NO_USER_INSTALL=1

mkdir -p logs

echo "=== Job started: $(date) ==="
echo "=== Node: $SLURMD_NODENAME, GPU: $CUDA_VISIBLE_DEVICES ==="
echo "=== Training Dataset104 fold ${FOLD} trainer=nnUNetTrainerTopologyVessel ==="

"$PY" -m nnunetv2.run.run_training 104 3d_fullres "${FOLD}" \
    -tr nnUNetTrainerTopologyVessel --npz

echo "=== Fold ${FOLD} finished: $(date) ==="
