#!/bin/bash
set -uo pipefail
cd /scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset
VENV=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/nnunet_venv
for TAG in nogeom pairedhalf; do
  echo "=== evaluating $TAG $(date) ==="
  "$VENV/bin/python3.9" evaluation/evaluate.py \
    --pred_dir "evaluation/results/CLEAN/${TAG}_heldout" \
    --gt_dir preprocessed/nnUNet_raw/Dataset104_TopCoW_joint/labelsTr \
    --dataset_json preprocessed/nnUNet_raw/Dataset104_TopCoW_joint/dataset.json \
    --split_file preprocessed/nnUNet_preprocessed/Dataset104_TopCoW_joint/splits_final.json \
    --fold 0 --output "evaluation/results/CLEAN/${TAG}_DS104_heldout" --num_workers 4
  echo "  -> exit $?"
done
echo "=== evals done $(date) ==="
