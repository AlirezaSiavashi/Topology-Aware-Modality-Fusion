#!/bin/bash
BASE=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset
PY=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/nnunet_venv/bin/python3.9
cd $BASE
TAG=${TAG:-baseline}
declare -A SRC=(
 [200_CrossDomain_CTA_ISLES2024]=CTA_ISLES2024_TUM
 [201_CrossDomain_MRA_IXI_HH]=MRA_IXI_HH
 [202_CrossDomain_MRA_Lausanne]=MRA_Lausanne )
until ! pgrep -f "nnUNetv2_predict" >/dev/null; do sleep 30; done
for DS in "${!SRC[@]}"; do
  P="evaluation/results/CLEAN/zeroshot_${TAG}_${DS}"
  G="evaluation/results/CLEAN/gt_${DS}"
  echo "================ $DS ================"
  $PY evaluation/prepare_zeroshot_gt.py --pred_dir "$P" \
      --gt_src "dataset/15692630/${SRC[$DS]}/cow_seg_labelsTr" \
      --out_dir "$G" \
      --dataset_json preprocessed/nnUNet_preprocessed/Dataset104_TopCoW_joint/dataset.json
  $PY evaluation/evaluate.py --pred_dir "$P" --gt_dir "$G" \
      --dataset_json preprocessed/nnUNet_preprocessed/Dataset104_TopCoW_joint/dataset.json \
      --allow_full_set --no_hd95 --num_workers 16 \
      --output "evaluation/results/CLEAN/zeroshot_${TAG}_${DS}.csv" 2>&1 | tail -20
done
