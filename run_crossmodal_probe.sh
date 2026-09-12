#!/bin/bash
# Cross-modality transfer probe with the existing single-modality models
# (trained on the ORIGINAL CT windowing). Answers: does a model trained on one
# modality work on the other at all?
BASE="/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset"
VENV="/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/nnunet_venv"
export nnUNet_raw="${BASE}/preprocessed/nnUNet_raw" nnUNet_preprocessed="${BASE}/preprocessed/nnUNet_preprocessed" nnUNet_results="${BASE}/results/nnUNet"
cd "$BASE"
# CT-trained model (d=102) applied to held-out MRA, on GPU 0
CUDA_VISIBLE_DEVICES=0 "$VENV/bin/nnUNetv2_predict" \
  -i evaluation/results/CLEAN/heldout_imgs_mr -o evaluation/results/CLEAN/xmod_CTmodel_on_MRA \
  -d 102 -c 3d_fullres -f 0 -tr nnUNetTrainer -chk checkpoint_final.pth --disable_tta -npp 4 -nps 4 &
P1=$!
# MR-trained model (d=103) applied to held-out CTA, on GPU 1
CUDA_VISIBLE_DEVICES=1 "$VENV/bin/nnUNetv2_predict" \
  -i evaluation/results/CLEAN/heldout_imgs_ct -o evaluation/results/CLEAN/xmod_MRmodel_on_CTA \
  -d 103 -c 3d_fullres -f 0 -tr nnUNetTrainer -chk checkpoint_final.pth --disable_tta -npp 4 -nps 4 &
P2=$!
wait $P1 $P2
echo "=== cross-modality inference done $(date) ==="
