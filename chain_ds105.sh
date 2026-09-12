#!/bin/bash
set -euo pipefail
BASE="/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset"
VENV="/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/nnunet_venv"
export nnUNet_raw="${BASE}/preprocessed/nnUNet_raw" nnUNet_preprocessed="${BASE}/preprocessed/nnUNet_preprocessed" nnUNet_results="${BASE}/results/nnUNet"
export CUDA_VISIBLE_DEVICES=0
cd "$BASE"

# 1. wait for the dataset build
until [ "$(ls preprocessed/nnUNet_raw/Dataset105_TopCoW_ctwin/imagesTr/*.nii.gz 2>/dev/null | wc -l)" -ge 250 ]; do sleep 30; done
echo "=== DS105 built: $(ls preprocessed/nnUNet_raw/Dataset105_TopCoW_ctwin/imagesTr/*.nii.gz | wc -l) images ==="

# 2. plan + preprocess
"$VENV/bin/nnUNetv2_plan_and_preprocess" -d 105 --verify_dataset_integrity -np 16 -c 3d_fullres

# 3. reuse Dataset104's patient-disjoint splits verbatim (identical case ids),
#    so the comparison differs only in CTA intensity mapping
cp preprocessed/nnUNet_preprocessed/Dataset104_TopCoW_joint/splits_final.json \
   preprocessed/nnUNet_preprocessed/Dataset105_TopCoW_ctwin/splits_final.json
"$VENV/bin/python3.9" - <<'PY'
import json
f=json.load(open("preprocessed/nnUNet_preprocessed/Dataset105_TopCoW_ctwin/splits_final.json"))[0]
tr={c.split("_")[-1] for c in f["train"]}; va={c.split("_")[-1] for c in f["val"]}
assert not (tr & va), f"ABORT: {len(tr&va)} subjects leak"
print(f"split check OK: train {len(f['train'])} val {len(f['val'])}, 0 leaked")
PY

# 4. train
"$VENV/bin/python3.9" -m nnunetv2.run.run_training 105 3d_fullres 0 -tr nnUNetTrainer --npz
echo "=== DS105 training done $(date) ==="
