#!/bin/bash
# Launch TCMN v2 training on GPU 1 free MIG partition
# Fixes: (1) residual FiLM init, (2) aux loss weights reduced to 0.1/0.05/0.05

export CUDA_VISIBLE_DEVICES=MIG-50e7ea79-5c28-5c08-8703-7df708730fd8

BASE=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection
export nnUNet_raw=$BASE/brain_external_dataset/preprocessed/nnUNet_raw
export nnUNet_preprocessed=$BASE/brain_external_dataset/preprocessed/nnUNet_preprocessed
export nnUNet_results=$BASE/brain_external_dataset/results/nnUNet
export nnUNet_n_proc_DA=4

PYTHON=$BASE/nnunet_venv/bin/python3.9
NNUNET=$BASE/nnunet_venv/bin/nnUNetv2_train

LOG_DIR=$BASE/brain_external_dataset/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/tcmn_v2_$(date +%Y%m%d_%H%M%S).log"

echo "Starting TCMN v2 training on MIG device: $CUDA_VISIBLE_DEVICES"
echo "Log: $LOG"
echo "Changes: residual FiLM init, lambda_topo=0.1, lambda_sdf=0.05, lambda_skel=0.05"

# Move old TCMN checkpoint to backup so training starts fresh
OLD_CKPT=$nnUNet_results/Dataset104_TopCoW_joint/nnUNetTrainerTCMN__nnUNetPlans__3d_fullres/fold_0
if [ -f "$OLD_CKPT/checkpoint_final.pth" ]; then
    mkdir -p "$OLD_CKPT/backup_v1"
    mv "$OLD_CKPT/checkpoint_final.pth" "$OLD_CKPT/backup_v1/" 2>/dev/null
    mv "$OLD_CKPT/checkpoint_best.pth" "$OLD_CKPT/backup_v1/" 2>/dev/null
    mv "$OLD_CKPT/checkpoint_latest.pth" "$OLD_CKPT/backup_v1/" 2>/dev/null
    echo "Moved old v1 checkpoints to backup_v1/"
fi

nohup $NNUNET 104 3d_fullres 0 -tr nnUNetTrainerTCMN > "$LOG" 2>&1 &
PID=$!
echo "Training PID: $PID"
echo $PID > "$LOG_DIR/tcmn_v2.pid"
echo "Tail log with: tail -f $LOG"
