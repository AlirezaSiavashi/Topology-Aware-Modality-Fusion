#!/bin/bash
BASE=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset
source /scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/nnunet_venv/bin/activate
export PYTHONPATH="${BASE}/training:${PYTHONPATH}"

echo "=== [E4] MR baseline in-domain (DS103 → DS103) ==="
python3 ${BASE}/evaluation/evaluate.py \
  --pred_dir ${BASE}/predictions/baseline103/DS103_indomain \
  --gt_dir ${BASE}/preprocessed/nnUNet_raw/Dataset103_TopCoW_MR/labelsTr \
  --dataset_json ${BASE}/preprocessed/nnUNet_raw/Dataset103_TopCoW_MR/dataset.json \
  --output ${BASE}/evaluation/results/E4_MRbaseline_DS103_indomain \
  --num_workers 16

echo ""
echo "=== [E4x] MR baseline cross-domain (DS103 model → DS102 CTA) ==="
python3 ${BASE}/evaluation/evaluate.py \
  --pred_dir ${BASE}/predictions/baseline103/DS102_crossdomain \
  --gt_dir ${BASE}/preprocessed/nnUNet_raw/Dataset102_TopCoW_CT/labelsTr \
  --dataset_json ${BASE}/preprocessed/nnUNet_raw/Dataset102_TopCoW_CT/dataset.json \
  --output ${BASE}/evaluation/results/E4x_MRbaseline_DS102_crossdomain \
  --num_workers 16

echo "=== ALL DONE ==="
