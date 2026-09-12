#!/bin/bash
# Run validation + zero-shot inference directly on GPU 0 (no SLURM)
# Shares GPU with CMTC training (which uses ~12GB, leaving ~28GB free)

source /scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/nnunet_venv/bin/activate

BASE=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset
export nnUNet_raw="${BASE}/preprocessed/nnUNet_raw"
export nnUNet_preprocessed="${BASE}/preprocessed/nnUNet_preprocessed"
export nnUNet_results="${BASE}/results/nnUNet"
export PYTHONPATH="${BASE}/training:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0

LOG="${BASE}/training/val_zeroshot_$(date +%Y%m%d_%H%M%S).log"

echo "=== Val + Zero-shot started: $(date) ===" | tee "$LOG"

# ── 1. Validation: TopologyVessel DS102 ──
echo "=== [1/4] Validation: TopologyVessel DS102 ===" | tee -a "$LOG"
nnUNetv2_train 102 3d_fullres 0 -tr nnUNetTrainerTopologyVessel --val 2>&1 | tee -a "$LOG"
echo "=== [1/4] Done: $(date) ===" | tee -a "$LOG"

# ── 2. Validation: TCMN DS104 ──
echo "=== [2/4] Validation: TCMN DS104 ===" | tee -a "$LOG"
nnUNetv2_train 104 3d_fullres 0 -tr nnUNetTrainerTCMN --val 2>&1 | tee -a "$LOG"
echo "=== [2/4] Done: $(date) ===" | tee -a "$LOG"

# ── 3. Zero-shot: DS104 baseline → cross-domain ──
echo "=== [3/4] Zero-shot: DS104 baseline ===" | tee -a "$LOG"
OUTBASE="${BASE}/results/zeroshot/DS104_baseline"

mkdir -p "${OUTBASE}/DS200_CTA_ISLES2024"
nnUNetv2_predict \
    -i "${BASE}/preprocessed/nnUNet_raw/Dataset200_CrossDomain_CTA_ISLES2024/imagesTr" \
    -o "${OUTBASE}/DS200_CTA_ISLES2024" \
    -d 104 -c 3d_fullres -tr nnUNetTrainer -f 0 \
    --disable_tta --continue_prediction 2>&1 | tee -a "$LOG"

mkdir -p "${OUTBASE}/DS201_MRA_IXI_HH"
nnUNetv2_predict \
    -i "${BASE}/preprocessed/nnUNet_raw/Dataset201_CrossDomain_MRA_IXI_HH/imagesTr" \
    -o "${OUTBASE}/DS201_MRA_IXI_HH" \
    -d 104 -c 3d_fullres -tr nnUNetTrainer -f 0 \
    --disable_tta --continue_prediction 2>&1 | tee -a "$LOG"

mkdir -p "${OUTBASE}/DS202_MRA_Lausanne"
nnUNetv2_predict \
    -i "${BASE}/preprocessed/nnUNet_raw/Dataset202_CrossDomain_MRA_Lausanne/imagesTr" \
    -o "${OUTBASE}/DS202_MRA_Lausanne" \
    -d 104 -c 3d_fullres -tr nnUNetTrainer -f 0 \
    --disable_tta --continue_prediction 2>&1 | tee -a "$LOG"

echo "=== [3/4] Done: $(date) ===" | tee -a "$LOG"

# ── 4. Zero-shot: DS104 TCMN → cross-domain ──
echo "=== [4/4] Zero-shot: DS104 TCMN ===" | tee -a "$LOG"
OUTBASE="${BASE}/results/zeroshot/DS104_TCMN"

mkdir -p "${OUTBASE}/DS200_CTA_ISLES2024"
nnUNetv2_predict \
    -i "${BASE}/preprocessed/nnUNet_raw/Dataset200_CrossDomain_CTA_ISLES2024/imagesTr" \
    -o "${OUTBASE}/DS200_CTA_ISLES2024" \
    -d 104 -c 3d_fullres -tr nnUNetTrainerTCMN -f 0 \
    --disable_tta --continue_prediction 2>&1 | tee -a "$LOG"

mkdir -p "${OUTBASE}/DS201_MRA_IXI_HH"
nnUNetv2_predict \
    -i "${BASE}/preprocessed/nnUNet_raw/Dataset201_CrossDomain_MRA_IXI_HH/imagesTr" \
    -o "${OUTBASE}/DS201_MRA_IXI_HH" \
    -d 104 -c 3d_fullres -tr nnUNetTrainerTCMN -f 0 \
    --disable_tta --continue_prediction 2>&1 | tee -a "$LOG"

mkdir -p "${OUTBASE}/DS202_MRA_Lausanne"
nnUNetv2_predict \
    -i "${BASE}/preprocessed/nnUNet_raw/Dataset202_CrossDomain_MRA_Lausanne/imagesTr" \
    -o "${OUTBASE}/DS202_MRA_Lausanne" \
    -d 104 -c 3d_fullres -tr nnUNetTrainerTCMN -f 0 \
    --disable_tta --continue_prediction 2>&1 | tee -a "$LOG"

echo "=== [4/4] Done: $(date) ===" | tee -a "$LOG"
echo "=== All val + zero-shot finished: $(date) ===" | tee -a "$LOG"
