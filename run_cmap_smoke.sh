#!/bin/bash
BASE="/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset"
VENV="/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/nnunet_venv"
export nnUNet_raw="${BASE}/preprocessed/nnUNet_raw"
export nnUNet_preprocessed="${BASE}/preprocessed/nnUNet_preprocessed"
export nnUNet_results="${BASE}/results/nnUNet"
export CUDA_VISIBLE_DEVICES=1
cd "$BASE"
"$VENV/bin/python3.9" -m nnunetv2.run.run_training 104 3d_fullres 0 -tr nnUNetTrainerCMAP_2epochs --npz
