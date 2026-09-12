#!/bin/bash
#SBATCH --job-name=nnunet
#SBATCH --partition=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=192:00:00
#SBATCH --output=logs/slurm_%x_%j.log
#SBATCH --error=logs/slurm_%x_%j.log

# Generalised nnU-Net training launcher.
#
# Usage:
#   sbatch --export=DATASET=104,FOLD=0,TRAINER=nnUNetTrainer \
#          --job-name=ds104_clean_f0 submit_nnunet.sh
#
# Supersedes submit_nnunet_fold.sh, which hardcoded dataset 104 +
# nnUNetTrainerTopologyVessel.

set -euo pipefail

DATASET=${DATASET:-104}
FOLD=${FOLD:-0}
TRAINER=${TRAINER:-nnUNetTrainer}
CONFIG=${CONFIG:-3d_fullres}

VENV="/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/nnunet_venv"
BASE="/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset"
PY="$VENV/bin/python3.9"

export nnUNet_raw="${BASE}/preprocessed/nnUNet_raw"
export nnUNet_preprocessed="${BASE}/preprocessed/nnUNet_preprocessed"
export nnUNet_results="${BASE}/results/nnUNet"
export PATH="$VENV/bin:$PATH"
export PIP_NO_USER_INSTALL=1

cd "$BASE"
mkdir -p logs

echo "=== Job started: $(date) ==="
echo "=== Node: ${SLURMD_NODENAME:-?}, GPU: ${CUDA_VISIBLE_DEVICES:-?} ==="
echo "=== Dataset${DATASET} ${CONFIG} fold ${FOLD} trainer=${TRAINER} ==="

# Fail loudly if the split file is not patient-disjoint (TopCoW ct_i / mr_i are
# the same subject; see preprocessing/make_patient_disjoint_splits.py).
"$PY" - "$DATASET" "$FOLD" <<'PYEOF'
import json, sys, glob
ds, fold = sys.argv[1], int(sys.argv[2])
hits = glob.glob(f"preprocessed/nnUNet_preprocessed/Dataset{ds}_*/splits_final.json")
if not hits:
    sys.exit(f"no splits_final.json for Dataset{ds}")
f = json.load(open(hits[0]))[fold]
tr = {c.split("_")[-1] for c in f["train"]}
va = {c.split("_")[-1] for c in f["val"]}
if tr & va:
    sys.exit(f"ABORT: {len(tr & va)} subjects in both train and val of fold {fold}")
print(f"split check OK: fold {fold} train {len(f['train'])} val {len(f['val'])}, 0 leaked subjects")
PYEOF

"$PY" -m nnunetv2.run.run_training "$DATASET" "$CONFIG" "$FOLD" -tr "$TRAINER" --npz

echo "=== Finished: $(date) ==="
