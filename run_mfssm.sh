#!/bin/bash
set -euo pipefail
TRAINER=${TRAINER:?}; GPU=${GPU:-0}; FOLD=${FOLD:-0}; DATASET=${DATASET:-104}
BASE="/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset"
VENV="/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/nnunet_venv"
export nnUNet_raw="${BASE}/preprocessed/nnUNet_raw" nnUNet_preprocessed="${BASE}/preprocessed/nnUNet_preprocessed" nnUNet_results="${BASE}/results/nnUNet"
export CUDA_VISIBLE_DEVICES="${GPU}"
cd "$BASE"; mkdir -p logs
DEST="${VENV}/lib64/python3.9/site-packages/nnunetv2/training/nnUNetTrainer/variants/loss"
cp training/mfssm.py training/mfssm_net.py training/nnUNetTrainerMFSSM.py "${DEST}/"
rm -rf "${DEST}/__pycache__"
"$VENV/bin/python3.9" - "$DATASET" "$FOLD" <<'PY'
import glob,json,sys
ds,fold=sys.argv[1],int(sys.argv[2])
f=json.load(open(glob.glob(f"preprocessed/nnUNet_preprocessed/Dataset{ds}_*/splits_final.json")[0]))[fold]
tr={c.split("_")[-1] for c in f["train"]}; va={c.split("_")[-1] for c in f["val"]}
assert not (tr&va), f"ABORT: {len(tr&va)} subjects leak"
print(f"split OK: train {len(f['train'])} val {len(f['val'])}, 0 leaked")
PY
nohup "$VENV/bin/python3.9" -m nnunetv2.run.run_training "$DATASET" 3d_fullres "$FOLD" -tr "$TRAINER" --npz \
  > "logs/${TRAINER}_f${FOLD}.log" 2>&1 &
echo "$TRAINER on GPU $GPU pid $!"
