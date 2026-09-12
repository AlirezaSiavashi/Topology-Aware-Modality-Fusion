#!/bin/bash
set -e

BASE=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection
export nnUNet_raw=$BASE/brain_external_dataset/preprocessed/nnUNet_raw
export nnUNet_preprocessed=$BASE/brain_external_dataset/preprocessed/nnUNet_preprocessed
export nnUNet_results=$BASE/brain_external_dataset/results/nnUNet

NNPRED=$BASE/nnunet_venv/bin/nnUNetv2_predict
OUT_BASE=$BASE/brain_external_dataset/results/zeroshot_eval

IXI=$BASE/brain_external_dataset/preprocessed/nnUNet_raw/Dataset201_CrossDomain_MRA_IXI_HH/imagesTr
LAU=$BASE/brain_external_dataset/preprocessed/nnUNet_raw/Dataset202_CrossDomain_MRA_Lausanne/imagesTr

predict() {
    local TRAINER=$1 DATASET=$2 LABEL=$3 INPUT=$4
    local OUT=$OUT_BASE/${LABEL}_${DATASET}
    mkdir -p "$OUT"
    echo ""
    echo ">>> $LABEL on $DATASET ($INPUT)"
    $NNPRED -i "$INPUT" -o "$OUT" \
        -d 104 -c 3d_fullres -f 0 \
        -tr "$TRAINER" \
        -chk checkpoint_best.pth \
        --disable_tta
    echo "Done: $OUT"
}

mkdir -p "$OUT_BASE"

# E0 on both datasets
predict nnUNetTrainer                IXI   E0 "$IXI"
predict nnUNetTrainer                LAU   E0 "$LAU"

# E1 on both datasets
predict nnUNetTrainerTopologyVessel  IXI   E1 "$IXI"
predict nnUNetTrainerTopologyVessel  LAU   E1 "$LAU"

# TCMN on both datasets
predict nnUNetTrainerTCMN            IXI   TCMN "$IXI"
predict nnUNetTrainerTCMN            LAU   TCMN "$LAU"

echo ""
echo "=== ALL ZERO-SHOT PREDICTIONS DONE ==="
echo "Results in: $OUT_BASE"
ls -la "$OUT_BASE"
