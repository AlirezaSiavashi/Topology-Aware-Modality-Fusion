#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --mem=80G
#SBATCH --time=0
#SBATCH --job-name=preprocess
#SBATCH --output=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset/training/slurm_preprocess_%j.log

BASE=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection

source ${BASE}/nnunet_venv/bin/activate

export nnUNet_raw="${BASE}/brain_external_dataset/preprocessed/nnUNet_raw"
export nnUNet_preprocessed="${BASE}/brain_external_dataset/preprocessed/nnUNet_preprocessed"
export nnUNet_results="${BASE}/brain_external_dataset/results/nnUNet"

echo "=== Job started: $(date) ==="
echo "=== Node: $(hostname) ==="

# Plan + preprocess datasets 101, 102, 103, 104, 110
# -c 3d_fullres  : only prepare the 3d_fullres configuration
# -np 16         : use 16 parallel workers (server has 96 CPUs)
nnUNetv2_plan_and_preprocess -d 101 102 103 104 110 -c 3d_fullres -np 16

echo "=== Job finished: $(date) ==="
