#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --mem=80G
#SBATCH --time=0
#SBATCH --job-name=preprocess_topcow
#SBATCH --output=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset/training/slurm_preprocess_%j.log

source /scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/aneur/bin/activate

export nnUNet_raw=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset/raw/nnUNet_raw
export nnUNet_preprocessed=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset/preprocessed/nnUNet_preprocessed
export nnUNet_results=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset/results/nnUNet

nnUNetv2_preprocess -d 101 102 103 104 110 -c 3d_fullres -np 16
