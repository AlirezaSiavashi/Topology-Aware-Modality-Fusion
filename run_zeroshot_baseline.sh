#!/bin/bash
# Zero-shot inference of a trained DS104 model on the three external sets.
# Usage: TRAINER=nnUNetTrainer GPU=0 ./run_zeroshot_baseline.sh
set -euo pipefail
TRAINER=${TRAINER:-nnUNetTrainer}
GPU=${GPU:-0}
TAG=${TAG:-baseline}
BASE="/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset"
VENV="/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/nnunet_venv"
export nnUNet_raw="${BASE}/preprocessed/nnUNet_raw"
export nnUNet_preprocessed="${BASE}/preprocessed/nnUNet_preprocessed"
export nnUNet_results="${BASE}/results/nnUNet"
export CUDA_VISIBLE_DEVICES="${GPU}"
cd "$BASE"
for DS in 200_CrossDomain_CTA_ISLES2024 201_CrossDomain_MRA_IXI_HH 202_CrossDomain_MRA_Lausanne; do
    OUT="evaluation/results/CLEAN/zeroshot_${TAG}_${DS}"
    mkdir -p "$OUT"
    echo "=== ${DS} -> ${OUT} ($(date +%H:%M:%S)) ==="
    "$VENV/bin/nnUNetv2_predict" \
        -i "preprocessed/nnUNet_raw/Dataset${DS}/imagesTr" \
        -o "$OUT" -d 104 -c 3d_fullres -f 0 -tr "$TRAINER" \
        -chk checkpoint_final.pth --disable_tta -npp 8 -nps 8
done
echo "=== zero-shot inference done $(date) ==="
