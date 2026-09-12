#!/bin/bash
# Run final inference + evaluation for E1 (TopologyVessel) using checkpoint_final.pth
# Saves predictions to validation_final/ to avoid overwriting existing mid-training predictions

BASE=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection
export nnUNet_raw=$BASE/brain_external_dataset/preprocessed/nnUNet_raw
export nnUNet_preprocessed=$BASE/brain_external_dataset/preprocessed/nnUNet_preprocessed
export nnUNet_results=$BASE/brain_external_dataset/results/nnUNet

PYTHON=$BASE/nnunet_venv/bin/python3.9
LOG=$BASE/brain_external_dataset/logs/e1_final_inference_$(date +%Y%m%d_%H%M%S).log

# Use GPU 0 — 33GB free
export CUDA_VISIBLE_DEVICES=0

DATASET=104
TRAINER=nnUNetTrainerTopologyVessel
FOLD=0
PRED_OUT=$nnUNet_results/Dataset104_TopCoW_joint/${TRAINER}__nnUNetPlans__3d_fullres/fold_${FOLD}/validation_final

echo "Running E1 final inference → $PRED_OUT" | tee "$LOG"
echo "Checkpoint: checkpoint_final.pth" | tee -a "$LOG"

$BASE/nnunet_venv/bin/nnUNetv2_predict \
    -i  $nnUNet_raw/Dataset104_TopCoW_joint/imagesTr \
    -o  $PRED_OUT \
    -d  $DATASET \
    -c  3d_fullres \
    -f  $FOLD \
    -tr $TRAINER \
    -chk checkpoint_final.pth \
    2>&1 | tee -a "$LOG"

echo "Inference done. Running evaluation..." | tee -a "$LOG"

$BASE/nnunet_venv/bin/nnUNetv2_evaluate_folder \
    $nnUNet_preprocessed/Dataset104_TopCoW_joint/gt_segmentations \
    $PRED_OUT \
    -djfile $nnUNet_results/Dataset104_TopCoW_joint/${TRAINER}__nnUNetPlans__3d_fullres/fold_${FOLD}/validation/dataset.json \
    -pfile  $nnUNet_results/Dataset104_TopCoW_joint/${TRAINER}__nnUNetPlans__3d_fullres/fold_${FOLD}/validation/plans.json \
    2>&1 | tee -a "$LOG"

echo "Done. Results in $PRED_OUT/summary.json" | tee -a "$LOG"
