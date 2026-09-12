#!/bin/bash
# Launch E1 v2 (TopologyVessel) with FIXED on-the-fly aux map computation
# Fix: aux targets now derived from batch['target'] instead of center-cropped external files
# Runs on GPU 1 MIG partition 1 (free 18GB)

export CUDA_VISIBLE_DEVICES=MIG-7acbdec9-a39b-5bf1-a5b1-f02688006c7c

BASE=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection
export nnUNet_raw=$BASE/brain_external_dataset/preprocessed/nnUNet_raw
export nnUNet_preprocessed=$BASE/brain_external_dataset/preprocessed/nnUNet_preprocessed
export nnUNet_results=$BASE/brain_external_dataset/results/nnUNet
export nnUNet_n_proc_DA=4

PYTHON=$BASE/nnunet_venv/bin/python3.9
NNUNET=$BASE/nnunet_venv/bin/nnUNetv2_train

LOG_DIR=$BASE/brain_external_dataset/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/e1_v2_$(date +%Y%m%d_%H%M%S).log"

echo "Starting E1 v2 (TopologyVessel fixed) on MIG: $CUDA_VISIBLE_DEVICES" | tee "$LOG"
echo "Fix: on-the-fly aux computation from batch target (aligned)" | tee -a "$LOG"

OLD=$nnUNet_results/Dataset104_TopCoW_joint/nnUNetTrainerTopologyVessel__nnUNetPlans__3d_fullres/fold_0

# If checkpoint_latest exists → resume; otherwise start fresh
if [ -f "$OLD/checkpoint_latest.pth" ]; then
    echo "Resuming from checkpoint_latest.pth" | tee -a "$LOG"
    RESUME_FLAG="--c"
else
    # Only backup/delete on a true fresh start
    if [ -f "$OLD/checkpoint_final.pth" ] && [ ! -d "$OLD/backup_v1" ]; then
        mkdir -p "$OLD/backup_v1"
        cp "$OLD/checkpoint_final.pth" "$OLD/backup_v1/" 2>/dev/null
        cp "$OLD/checkpoint_best.pth" "$OLD/backup_v1/" 2>/dev/null
        echo "Backed up old E1 checkpoints to backup_v1/" | tee -a "$LOG"
    fi
    rm -f "$OLD/checkpoint_final.pth" "$OLD/checkpoint_best.pth" "$OLD/checkpoint_latest.pth"
    RESUME_FLAG=""
    echo "Starting fresh training" | tee -a "$LOG"
fi

nohup $NNUNET 104 3d_fullres 0 -tr nnUNetTrainerTopologyVessel $RESUME_FLAG >> "$LOG" 2>&1 &
PID=$!
echo $PID > "$LOG_DIR/e1_v2.pid"
echo "Training PID: $PID" | tee -a "$LOG"
echo "Log: $LOG"
