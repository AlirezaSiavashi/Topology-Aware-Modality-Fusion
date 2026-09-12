#!/bin/bash
# Launch an M3-CoW training on a chosen local GPU.
#   TRAINER=nnUNetTrainerM3CoW GPU=0 FOLD=0 ./run_m3cow.sh
set -euo pipefail
TRAINER=${TRAINER:-nnUNetTrainerM3CoW}; GPU=${GPU:-0}; FOLD=${FOLD:-0}
DATASET=${DATASET:-104}; CONFIG=${CONFIG:-3d_fullres}
BASE="/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset"
VENV="/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/nnunet_venv"
PY="$VENV/bin/python3.9"
export nnUNet_raw="${BASE}/preprocessed/nnUNet_raw" \
       nnUNet_preprocessed="${BASE}/preprocessed/nnUNet_preprocessed" \
       nnUNet_results="${BASE}/results/nnUNet"
export CUDA_VISIBLE_DEVICES="${GPU}"
cd "$BASE"; mkdir -p logs

# Keep the deployed copy in sync with the working tree -- easy to forget, and
# silently trains stale code if you do.
DEST="${VENV}/lib64/python3.9/site-packages/nnunetv2/training/nnUNetTrainer/variants/loss"
for f in mamba3_ssm.py state_losses.py anatomy_transformer.py mfssm3_net.py \
         nnUNetTrainerM3CoW.py; do
    cp "training/${f}" "${DEST}/"
done
rm -rf "${DEST}/__pycache__"

# Refuse to train on a split that is not patient-disjoint (TopCoW ct_i / mr_i
# are the same subject).
"$PY" - "$DATASET" "$FOLD" <<'PYEOF'
import glob, json, sys
ds, fold = sys.argv[1], int(sys.argv[2])
hits = glob.glob(f"preprocessed/nnUNet_preprocessed/Dataset{ds}_*/splits_final.json")
if not hits: sys.exit(f"no splits_final.json for Dataset{ds}")
f = json.load(open(hits[0]))[fold]
tr = {c.split("_")[-1] for c in f["train"]}; va = {c.split("_")[-1] for c in f["val"]}
if tr & va: sys.exit(f"ABORT: {len(tr & va)} subjects in both train and val of fold {fold}")
print(f"split check OK: fold {fold} train {len(f['train'])} val {len(f['val'])}, 0 leaked")
PYEOF

LOG="logs/${TRAINER}_f${FOLD}.log"
echo "=== ${TRAINER} dataset ${DATASET} fold ${FOLD} on GPU ${GPU} -> ${LOG} ==="
nohup "$PY" -m nnunetv2.run.run_training "$DATASET" "$CONFIG" "$FOLD" \
    -tr "$TRAINER" --npz > "$LOG" 2>&1 &
echo "pid $!"
