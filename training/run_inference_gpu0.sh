#!/bin/bash
# Run inference tasks directly on GPU 0
# Uses predict_custom.py for trainers with TCMN (strict=False loading)

source /scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/nnunet_venv/bin/activate

BASE=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset
export nnUNet_raw="${BASE}/preprocessed/nnUNet_raw"
export nnUNet_preprocessed="${BASE}/preprocessed/nnUNet_preprocessed"
export nnUNet_results="${BASE}/results/nnUNet"
export PYTHONPATH="${BASE}/training:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0

PREDICT="${BASE}/training/predict_custom.py"

LOG="${BASE}/training/inference_gpu0_$(date +%Y%m%d_%H%M%S).log"
echo "=== Inference started: $(date) ===" | tee "$LOG"

# ── 1. TCMN DS104 validation ──────────────────────────────────────────────────
echo "=== [1/5] TCMN DS104 validation ===" | tee -a "$LOG"
python3 "${BASE}/training/run_tcmn_val.py" 2>&1 | tee -a "$LOG"
echo "=== [1/5] Done: $(date) ===" | tee -a "$LOG"

# ── 2. Zero-shot: DS104 TCMN → DS200/201/202 ─────────────────────────────────
echo "=== [2/5] Zero-shot: DS104 TCMN → cross-domain ===" | tee -a "$LOG"
OUTBASE="${BASE}/results/zeroshot/DS104_TCMN_v2"

for ds in "Dataset200_CrossDomain_CTA_ISLES2024:DS200" \
          "Dataset201_CrossDomain_MRA_IXI_HH:DS201" \
          "Dataset202_CrossDomain_MRA_Lausanne:DS202"; do
    INPUT=$(echo $ds | cut -d: -f1)
    OUTPUT=$(echo $ds | cut -d: -f2)
    mkdir -p "${OUTBASE}/${OUTPUT}"
    echo "  Predicting ${OUTPUT}..." | tee -a "$LOG"
    python3 "${PREDICT}" \
        -i "${nnUNet_raw}/${INPUT}/imagesTr" \
        -o "${OUTBASE}/${OUTPUT}" \
        -d 104 -c 3d_fullres -tr nnUNetTrainerTCMN -f 0 \
        --checkpoint checkpoint_final.pth \
        --disable_tta 2>&1 | tee -a "$LOG"
done
echo "=== [2/5] Done: $(date) ===" | tee -a "$LOG"

# ── 3. Zero-shot: DS102 TopologyVessel → DS200 (CTA only) ────────────────────
echo "=== [3/5] Zero-shot: DS102 TopologyVessel → DS200 ===" | tee -a "$LOG"
OUTBASE="${BASE}/results/zeroshot/DS102_TopologyVessel"
mkdir -p "${OUTBASE}/DS200"
python3 "${PREDICT}" \
    -i "${nnUNet_raw}/Dataset200_CrossDomain_CTA_ISLES2024/imagesTr" \
    -o "${OUTBASE}/DS200" \
    -d 102 -c 3d_fullres -tr nnUNetTrainerTopologyVessel -f 0 \
    --checkpoint checkpoint_final.pth \
    --disable_tta --continue_prediction 2>&1 | tee -a "$LOG"
echo "=== [3/5] Done: $(date) ===" | tee -a "$LOG"

# ── 4. Zero-shot: DS102 baseline → DS200/201/202 ─────────────────────────────
echo "=== [4/5] Zero-shot: DS102 baseline → cross-domain ===" | tee -a "$LOG"
OUTBASE="${BASE}/results/zeroshot/DS102_baseline"

for ds in "Dataset200_CrossDomain_CTA_ISLES2024:DS200" \
          "Dataset201_CrossDomain_MRA_IXI_HH:DS201" \
          "Dataset202_CrossDomain_MRA_Lausanne:DS202"; do
    INPUT=$(echo $ds | cut -d: -f1)
    OUTPUT=$(echo $ds | cut -d: -f2)
    mkdir -p "${OUTBASE}/${OUTPUT}"
    echo "  Predicting ${OUTPUT}..." | tee -a "$LOG"
    nnUNetv2_predict \
        -i "${nnUNet_raw}/${INPUT}/imagesTr" \
        -o "${OUTBASE}/${OUTPUT}" \
        -d 102 -c 3d_fullres -f 0 \
        --disable_tta --continue_prediction 2>&1 | tee -a "$LOG"
done
echo "=== [4/5] Done: $(date) ===" | tee -a "$LOG"

# ── 5. Zero-shot: DS103 baseline → DS200/201/202 ─────────────────────────────
echo "=== [5/5] Zero-shot: DS103 baseline → cross-domain ===" | tee -a "$LOG"
OUTBASE="${BASE}/results/zeroshot/DS103_baseline"

for ds in "Dataset200_CrossDomain_CTA_ISLES2024:DS200" \
          "Dataset201_CrossDomain_MRA_IXI_HH:DS201" \
          "Dataset202_CrossDomain_MRA_Lausanne:DS202"; do
    INPUT=$(echo $ds | cut -d: -f1)
    OUTPUT=$(echo $ds | cut -d: -f2)
    mkdir -p "${OUTBASE}/${OUTPUT}"
    echo "  Predicting ${OUTPUT}..." | tee -a "$LOG"
    nnUNetv2_predict \
        -i "${nnUNet_raw}/${INPUT}/imagesTr" \
        -o "${OUTBASE}/${OUTPUT}" \
        -d 103 -c 3d_fullres -f 0 \
        --disable_tta --continue_prediction 2>&1 | tee -a "$LOG"
done
echo "=== [5/5] Done: $(date) ===" | tee -a "$LOG"

echo "=== All inference finished: $(date) ===" | tee -a "$LOG"
