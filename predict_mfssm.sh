#!/bin/bash
# Two-pass inference for MF-SSM models.
#
# nnU-Net's built-in post-training validation predicts the whole mixed val set
# in one call and never passes modality, so it would route every MRA case
# through the CT operator. Its summary.json is therefore INVALID for MF-SSM
# models and must be ignored. This script predicts each modality separately
# with MFSSM_MODALITY set correctly, which is what every reported number uses.
#
# Usage: TRAINER=nnUNetTrainerMFSSM GPU=0 ./predict_mfssm.sh
set -euo pipefail
TRAINER=${TRAINER:?}; GPU=${GPU:-0}; DATASET=${DATASET:-104}; FOLD=${FOLD:-0}
BASE="/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset"
VENV="/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/nnunet_venv"
export nnUNet_raw="${BASE}/preprocessed/nnUNet_raw" nnUNet_preprocessed="${BASE}/preprocessed/nnUNet_preprocessed" nnUNet_results="${BASE}/results/nnUNet"
export CUDA_VISIBLE_DEVICES="${GPU}"
cd "$BASE"
OUT="evaluation/results/CLEAN/mfssm_${TRAINER}_heldout"; mkdir -p "$OUT"
"$VENV/bin/python3.9" - "$DATASET" "$FOLD" <<'PY'
import json,glob,shutil,sys
from pathlib import Path
ds,fold=sys.argv[1],int(sys.argv[2])
sp=json.load(open(glob.glob(f"preprocessed/nnUNet_preprocessed/Dataset{ds}_*/splits_final.json")[0]))[fold]
raw=Path(glob.glob(f"preprocessed/nnUNet_raw/Dataset{ds}_*")[0])/"imagesTr"
for mod in ("ct","mr"):
    d=Path(f"/tmp/mfssm_in_{mod}"); shutil.rmtree(d,ignore_errors=True); d.mkdir(parents=True)
    n=0
    for c in sp["val"]:
        if f"_{mod}_" in c:
            f=raw/f"{c}_0000.nii.gz"
            if f.exists(): shutil.copy2(f,d/f.name); n+=1
    print(f"{mod}: {n} held-out cases -> {d}")
PY
for M in 0 1; do
  [ "$M" = 0 ] && MOD=ct || MOD=mr
  echo "=== predicting $MOD with MFSSM_MODALITY=$M ==="
  MFSSM_MODALITY=$M "$VENV/bin/nnUNetv2_predict" -i /tmp/mfssm_in_${MOD} -o "$OUT" \
    -d "$DATASET" -c 3d_fullres -f "$FOLD" -tr "$TRAINER" -chk checkpoint_final.pth \
    --disable_tta -npp 4 -nps 4
done
echo "=== done: $(ls $OUT/*.nii.gz | wc -l) predictions in $OUT ==="
