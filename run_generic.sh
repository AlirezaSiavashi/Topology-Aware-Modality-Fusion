#!/bin/bash
set -euo pipefail
TRAINER=${TRAINER:?}; GPU=${GPU:-1}; FOLD=${FOLD:-0}; DATASET=${DATASET:-104}
BASE="/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset"
VENV="/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/nnunet_venv"
export nnUNet_raw="${BASE}/preprocessed/nnUNet_raw" nnUNet_preprocessed="${BASE}/preprocessed/nnUNet_preprocessed" nnUNet_results="${BASE}/results/nnUNet"
export CUDA_VISIBLE_DEVICES="${GPU}"
cd "$BASE"; mkdir -p logs
DEST="${VENV}/lib64/python3.9/site-packages/nnunetv2/training/nnUNetTrainer/variants/loss"
# Copy ONLY importable trainer modules. nnU-Net's recursive_find_python_class
# imports every module in this tree, so any analysis/test script copied here
# executes its top-level code on import -- that once made inference appear to
# hang at 200% CPU with no GPU use.
for f in mfssm.py mfssm_net.py cmap.py cmap_net.py hyperbolic.py tcmn.py \
         network_wrapper.py loss_cbdice.py loss_cmtc.py nnUNetTrainer*.py; do
    [ -f "training/$f" ] && cp "training/$f" "${DEST}/"
done
rm -rf "${DEST}/__pycache__"
nohup "$VENV/bin/python3.9" -m nnunetv2.run.run_training "$DATASET" 3d_fullres "$FOLD" -tr "$TRAINER" --npz \
  > "logs/${TRAINER}_f${FOLD}.log" 2>&1 &
echo "$TRAINER on GPU $GPU pid $!"
