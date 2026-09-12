#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --job-name=val_tcmn104
#SBATCH --output=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset/training/slurm_scripts/val_tcmn104_%j.log
#SBATCH --error=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset/training/slurm_scripts/val_tcmn104_%j.log

source /scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/nnunet_venv/bin/activate

BASE=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset
export nnUNet_raw="${BASE}/preprocessed/nnUNet_raw"
export nnUNet_preprocessed="${BASE}/preprocessed/nnUNet_preprocessed"
export nnUNet_results="${BASE}/results/nnUNet"
export PYTHONPATH="${BASE}/training:${PYTHONPATH}"

echo "=== Validation: TCMN DS104 TopCoW_joint ==="
echo "=== $(date) ==="

nnUNetv2_train 104 3d_fullres 0 -tr nnUNetTrainerTCMN --val

echo "=== Done: $(date) ==="
