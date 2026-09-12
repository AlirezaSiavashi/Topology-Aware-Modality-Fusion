#!/bin/bash
# Held-out evaluation of M3-CoW on the project's own pipeline, with the CORRECT
# modality operator per case (nnU-Net's internal validation ran everything as CT).
set -euo pipefail
BASE="/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset"
VENV="/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/nnunet_venv"
TRAINER=${TRAINER:-nnUNetTrainerM3CoW}; GPU=${GPU:-0}; TAG=${TAG:-m3cow}
export nnUNet_raw="${BASE}/preprocessed/nnUNet_raw" \
       nnUNet_preprocessed="${BASE}/preprocessed/nnUNet_preprocessed" \
       nnUNet_results="${BASE}/results/nnUNet"
export CUDA_VISIBLE_DEVICES="${GPU}"
cd "$BASE"
OUT="evaluation/results/CLEAN/${TAG}_heldout"
rm -rf "$OUT"; mkdir -p "$OUT"
for M in ct mr; do
    export MFSSM_MODALITY=$([ "$M" = "mr" ] && echo 1 || echo 0)
    echo "=== predicting ${M} (MFSSM_MODALITY=${MFSSM_MODALITY}) ==="
    "$VENV/bin/nnUNetv2_predict" -i "evaluation/results/CLEAN/heldout_imgs_${M}" \
        -o "$OUT" -d 104 -c 3d_fullres -f 0 -tr "$TRAINER" \
        -chk checkpoint_final.pth --disable_tta -npp 4 -nps 4
done
"$VENV/bin/python3.9" evaluation/evaluate.py \
    --pred_dir "$OUT" \
    --gt_dir preprocessed/nnUNet_raw/Dataset104_TopCoW_joint/labelsTr \
    --dataset_json preprocessed/nnUNet_raw/Dataset104_TopCoW_joint/dataset.json \
    --split_file preprocessed/nnUNet_preprocessed/Dataset104_TopCoW_joint/splits_final.json \
    --fold 0 --output "evaluation/results/CLEAN/${TAG}_DS104_heldout" --num_workers 16
