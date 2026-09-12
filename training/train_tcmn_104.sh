#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=120G
#SBATCH --time=0
#SBATCH --job-name=tcmn104
#SBATCH --output=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset/training/slurm_%j.log
#SBATCH --error=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset/training/slurm_%j.log

source /scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/nnunet_venv/bin/activate

BASE=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset

export nnUNet_raw="${BASE}/preprocessed/nnUNet_raw"
export nnUNet_preprocessed="${BASE}/preprocessed/nnUNet_preprocessed"
export nnUNet_results="${BASE}/results/nnUNet"
export PYTHONPATH="${BASE}/training:${PYTHONPATH}"

echo "=== Job started: $(date) ==="
echo "=== Node: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="

# DS104: joint CT+MR, 250 subjects, TCMN + topology losses
python -m nnunetv2.run.run_training 104 3d_fullres 0 \
  -tr nnUNetTrainerTCMN \
  --c \
  > ${BASE}/training/tcmn104.log 2>&1

echo "=== Job finished: $(date) ==="
