#!/bin/bash
set -e

BASE=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection
export nnUNet_raw=$BASE/brain_external_dataset/preprocessed/nnUNet_raw
export nnUNet_preprocessed=$BASE/brain_external_dataset/preprocessed/nnUNet_preprocessed
export nnUNet_results=$BASE/brain_external_dataset/results/nnUNet

NNPRED=$BASE/nnunet_venv/bin/nnUNetv2_predict
NNEVAL=$BASE/nnunet_venv/bin/nnUNetv2_evaluate_folder
RAW=$BASE/brain_external_dataset/preprocessed/nnUNet_raw/Dataset104_TopCoW_joint
PRED=$BASE/brain_external_dataset/results/clean_eval
PFILE=$BASE/brain_external_dataset/results/nnUNet/Dataset104_TopCoW_joint/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation/predict_from_raw_data_args.json

echo "=== Predicting E0 ==="
mkdir -p $PRED/pred_E0
$NNPRED -i $PRED/val_images -o $PRED/pred_E0 \
    -d 104 -c 3d_fullres -f 0 \
    -tr nnUNetTrainer \
    -chk checkpoint_best.pth --disable_tta

echo "=== Predicting E1 ==="
mkdir -p $PRED/pred_E1
$NNPRED -i $PRED/val_images -o $PRED/pred_E1 \
    -d 104 -c 3d_fullres -f 0 \
    -tr nnUNetTrainerTopologyVessel \
    -chk checkpoint_best.pth --disable_tta

echo "=== Predicting TCMN ==="
mkdir -p $PRED/pred_TCMN
$NNPRED -i $PRED/val_images -o $PRED/pred_TCMN \
    -d 104 -c 3d_fullres -f 0 \
    -tr nnUNetTrainerTCMN \
    -chk checkpoint_best.pth --disable_tta

echo "=== Evaluating E0 ==="
$NNEVAL $PRED/pred_E0 $PRED/val_labels \
    -djfile $RAW/dataset.json -pfile $PFILE

echo "=== Evaluating E1 ==="
$NNEVAL $PRED/pred_E1 $PRED/val_labels \
    -djfile $RAW/dataset.json -pfile $PFILE

echo "=== Evaluating TCMN ==="
$NNEVAL $PRED/pred_TCMN $PRED/val_labels \
    -djfile $RAW/dataset.json -pfile $PFILE

echo "ALL DONE"
