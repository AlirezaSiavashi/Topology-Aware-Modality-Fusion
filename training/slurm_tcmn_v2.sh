#!/bin/bash
#SBATCH --job-name=tcmn_v2
#SBATCH --partition=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=UNLIMITED
#SBATCH --output=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset/logs/tcmn_v2_%j.out
#SBATCH --error=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset/logs/tcmn_v2_%j.err

BASE=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection
export nnUNet_raw=$BASE/brain_external_dataset/preprocessed/nnUNet_raw
export nnUNet_preprocessed=$BASE/brain_external_dataset/preprocessed/nnUNet_preprocessed
export nnUNet_results=$BASE/brain_external_dataset/results/nnUNet
export nnUNet_n_proc_DA=8

PYTHON=$BASE/nnunet_venv/bin/python3.9
NNUNET=$BASE/nnunet_venv/bin/nnUNetv2_train

FOLD_DIR=$nnUNet_results/Dataset104_TopCoW_joint/nnUNetTrainerTCMN__nnUNetPlans__3d_fullres/fold_0

# Backup old TCMN checkpoints (broken run — no aux alignment)
if [ -f "$FOLD_DIR/checkpoint_final.pth" ] && [ ! -d "$FOLD_DIR/backup_v1_broken" ]; then
    mkdir -p "$FOLD_DIR/backup_v1_broken"
    cp "$FOLD_DIR/checkpoint_final.pth" "$FOLD_DIR/backup_v1_broken/" 2>/dev/null
    cp "$FOLD_DIR/checkpoint_best.pth"  "$FOLD_DIR/backup_v1_broken/" 2>/dev/null
    echo "Backed up old TCMN checkpoints (broken aux alignment) to backup_v1_broken/"
fi

# Resume if checkpoint_latest exists, otherwise start fresh
if [ -f "$FOLD_DIR/checkpoint_latest.pth" ]; then
    echo "Resuming TCMN v2 from checkpoint_latest.pth"
    RESUME_FLAG="--c"
else
    # Remove old final/best so nnUNet starts fresh
    rm -f "$FOLD_DIR/checkpoint_final.pth" "$FOLD_DIR/checkpoint_best.pth"
    RESUME_FLAG=""
    echo "Starting fresh TCMN v2 training"
fi

echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES"
echo "Trainer: nnUNetTrainerTCMN | Dataset: 104 | Fold: 0"

$NNUNET 104 3d_fullres 0 -tr nnUNetTrainerTCMN $RESUME_FLAG
