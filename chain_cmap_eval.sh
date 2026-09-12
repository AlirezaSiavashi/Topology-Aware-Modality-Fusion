#!/bin/bash
cd /scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset
PY=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/nnunet_venv/bin/python3.9
D=results/nnUNet/Dataset104_TopCoW_joint/nnUNetTrainerCMAP__nnUNetPlans__3d_fullres/fold_0
# wait for training AND final validation to finish
until [ "$(ls $D/validation/*.nii.gz 2>/dev/null | wc -l)" -ge 50 ]; do sleep 60; done
sleep 30
echo "=== CMAP held-out evaluation ==="
$PY evaluation/evaluate.py --pred_dir "$D/validation" \
  --gt_dir preprocessed/nnUNet_raw/Dataset104_TopCoW_joint/labelsTr \
  --dataset_json preprocessed/nnUNet_preprocessed/Dataset104_TopCoW_joint/dataset.json \
  --split_file preprocessed/nnUNet_preprocessed/Dataset104_TopCoW_joint/splits_final.json --fold 0 \
  --output evaluation/results/CLEAN/cmap_DS104_heldout.csv --num_workers 16
# GPU 1 is free once CMAP exits; run its zero-shot there
TRAINER=nnUNetTrainerCMAP GPU=1 TAG=cmap ./run_zeroshot_baseline.sh
TAG=cmap ./eval_zeroshot_chain.sh
