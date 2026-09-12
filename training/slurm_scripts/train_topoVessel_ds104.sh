#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=120G
#SBATCH --time=0
#SBATCH --job-name=tv104
#SBATCH --output=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset/training/slurm_scripts/tv104_%j.log
#SBATCH --error=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset/training/slurm_scripts/tv104_%j.log

source /scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/nnunet_venv/bin/activate

BASE=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset
export nnUNet_raw="${BASE}/preprocessed/nnUNet_raw"
export nnUNet_preprocessed="${BASE}/preprocessed/nnUNet_preprocessed"
export nnUNet_results="${BASE}/results/nnUNet"
export PYTHONPATH="${BASE}/training:${PYTHONPATH}"
export nnUNet_n_proc_DA=4

echo "=== Training: TopologyVessel DS104 TopCoW_joint (E1 ablation) ==="
echo "=== $(date) ==="

# E1: nnUNet baseline + cbDice + SDF + skeleton on joint CTA+MRA
nnUNetv2_train 104 3d_fullres 0 -tr nnUNetTrainerTopologyVessel

echo "=== Done: $(date) ==="
