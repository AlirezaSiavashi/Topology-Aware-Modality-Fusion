#!/bin/bash
# Launch a CMAP (or any) nnU-Net training on a chosen local GPU via nohup.
# The SLURM path (submit_nnunet.sh) is for the queue; this is for driving the
# second GPU on willi directly.
#
# Usage:
#   TRAINER=nnUNetTrainerCMAP GPU=1 FOLD=0 ./run_cmap.sh
#   TRAINER=nnUNetTrainerCMAP_NoHyperbolic GPU=1 ./run_cmap.sh
#
# Writes logs/<TRAINER>_f<FOLD>.log and prints the pid.

set -euo pipefail

TRAINER=${TRAINER:-nnUNetTrainerCMAP}
GPU=${GPU:-1}
FOLD=${FOLD:-0}
DATASET=${DATASET:-104}
CONFIG=${CONFIG:-3d_fullres}

BASE="/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset"
VENV="/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/nnunet_venv"
PY="$VENV/bin/python3.9"

export nnUNet_raw="${BASE}/preprocessed/nnUNet_raw"
export nnUNet_preprocessed="${BASE}/preprocessed/nnUNet_preprocessed"
export nnUNet_results="${BASE}/results/nnUNet"
export CUDA_VISIBLE_DEVICES="${GPU}"

cd "$BASE"
mkdir -p logs

# Keep the deployed copy in sync with the working tree -- easy to forget, and
# silently trains stale code if you do.
DEST="${VENV}/lib64/python3.9/site-packages/nnunetv2/training/nnUNetTrainer/variants/loss"
for f in hyperbolic.py cmap.py cmap_net.py nnUNetTrainerCMAP.py; do
    cp "training/${f}" "${DEST}/"
done
rm -rf "${DEST}/__pycache__"

# Refuse to train on a split that is not patient-disjoint (TopCoW ct_i / mr_i
# are the same subject).
"$PY" - "$DATASET" "$FOLD" <<'PYEOF'
import glob, json, sys
ds, fold = sys.argv[1], int(sys.argv[2])
hits = glob.glob(f"preprocessed/nnUNet_preprocessed/Dataset{ds}_*/splits_final.json")
if not hits:
    sys.exit(f"no splits_final.json for Dataset{ds}")
f = json.load(open(hits[0]))[fold]
tr = {c.split("_")[-1] for c in f["train"]}
va = {c.split("_")[-1] for c in f["val"]}
if tr & va:
    sys.exit(f"ABORT: {len(tr & va)} subjects in both train and val of fold {fold}")
print(f"split check OK: fold {fold} train {len(f['train'])} val {len(f['val'])}, 0 leaked")
PYEOF

LOG="logs/${TRAINER}_f${FOLD}.log"
echo "=== ${TRAINER} dataset ${DATASET} fold ${FOLD} on GPU ${GPU} -> ${LOG} ==="
nohup "$PY" -m nnunetv2.run.run_training "$DATASET" "$CONFIG" "$FOLD" \
    -tr "$TRAINER" --npz > "$LOG" 2>&1 &
echo "pid $!"
