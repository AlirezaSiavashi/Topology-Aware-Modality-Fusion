#!/bin/bash
# Run all missing inference tasks on GPU 0
# Both GPUs busy with training, but GPU 0 has ~28GB free

source /scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/nnunet_venv/bin/activate

BASE=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset
export nnUNet_raw="${BASE}/preprocessed/nnUNet_raw"
export nnUNet_preprocessed="${BASE}/preprocessed/nnUNet_preprocessed"
export nnUNet_results="${BASE}/results/nnUNet"
export PYTHONPATH="${BASE}/training:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0

DS200="${BASE}/preprocessed/nnUNet_raw/Dataset200_CrossDomain_CTA_ISLES2024/imagesTr"
DS201="${BASE}/preprocessed/nnUNet_raw/Dataset201_CrossDomain_MRA_IXI_HH/imagesTr"
DS202="${BASE}/preprocessed/nnUNet_raw/Dataset202_CrossDomain_MRA_Lausanne/imagesTr"
ZEROSHOT="${BASE}/results/zeroshot"

LOG="${BASE}/training/all_inference_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== All inference started: $(date) ==="

# ── Task 1: TCMN DS104 validation ──────────────────────────────────────
echo ""
echo "=== Task 1/7: TCMN DS104 validation ==="
VAL_DIR="${nnUNet_results}/Dataset104_TopCoW_joint/nnUNetTrainerTCMN__nnUNetPlans__3d_fullres/fold_0/validation/summary.json"
if [ -f "$VAL_DIR" ]; then
    echo "SKIP: Already exists"
else
    python3 ${BASE}/training/run_tcmn_val.py
fi

# ── Task 2: TCMN zero-shot → DS200 ─────────────────────────────────────
echo ""
echo "=== Task 2/7: TCMN zero-shot → DS200 (CTA ISLES2024) ==="
OUT="${ZEROSHOT}/DS104_TCMN_v2/DS200_CTA_ISLES2024"
if [ "$(find "$OUT" -name '*.nii.gz' 2>/dev/null | wc -l)" -ge 26 ]; then
    echo "SKIP: Already have 26+ predictions"
else
    python3 ${BASE}/training/run_tcmn_predict.py -i "$DS200" -o "$OUT"
fi

# ── Task 3: TCMN zero-shot → DS201 ─────────────────────────────────────
echo ""
echo "=== Task 3/7: TCMN zero-shot → DS201 (MRA IXI HH) ==="
OUT="${ZEROSHOT}/DS104_TCMN_v2/DS201_MRA_IXI_HH"
if [ "$(find "$OUT" -name '*.nii.gz' 2>/dev/null | wc -l)" -ge 20 ]; then
    echo "SKIP: Already have 20+ predictions"
else
    python3 ${BASE}/training/run_tcmn_predict.py -i "$DS201" -o "$OUT"
fi

# ── Task 4: TCMN zero-shot → DS202 ─────────────────────────────────────
echo ""
echo "=== Task 4/7: TCMN zero-shot → DS202 (MRA Lausanne) ==="
OUT="${ZEROSHOT}/DS104_TCMN_v2/DS202_MRA_Lausanne"
if [ "$(find "$OUT" -name '*.nii.gz' 2>/dev/null | wc -l)" -ge 20 ]; then
    echo "SKIP: Already have 20+ predictions"
else
    python3 ${BASE}/training/run_tcmn_predict.py -i "$DS202" -o "$OUT"
fi

# ── Task 5: DS102 baseline zero-shot → DS200 ───────────────────────────
echo ""
echo "=== Task 5/7: DS102 baseline → DS200 (CTA ISLES2024) ==="
OUT="${ZEROSHOT}/DS102_baseline/DS200_CTA_ISLES2024"
if [ "$(find "$OUT" -name '*.nii.gz' 2>/dev/null | wc -l)" -ge 26 ]; then
    echo "SKIP: Already have 26+ predictions"
else
    mkdir -p "$OUT"
    nnUNetv2_predict -i "$DS200" -o "$OUT" \
        -d 102 -c 3d_fullres -tr nnUNetTrainer -f 0 \
        --disable_tta --continue_prediction
fi

# ── Task 6: DS103 baseline zero-shot → DS201 ───────────────────────────
echo ""
echo "=== Task 6/7: DS103 baseline → DS201 (MRA IXI HH) ==="
OUT="${ZEROSHOT}/DS103_baseline/DS201_MRA_IXI_HH"
if [ "$(find "$OUT" -name '*.nii.gz' 2>/dev/null | wc -l)" -ge 20 ]; then
    echo "SKIP: Already have 20+ predictions"
else
    mkdir -p "$OUT"
    nnUNetv2_predict -i "$DS201" -o "$OUT" \
        -d 103 -c 3d_fullres -tr nnUNetTrainer -f 0 \
        --disable_tta --continue_prediction
fi

# ── Task 7: DS103 baseline zero-shot → DS202 ───────────────────────────
echo ""
echo "=== Task 7/7: DS103 baseline → DS202 (MRA Lausanne) ==="
OUT="${ZEROSHOT}/DS103_baseline/DS202_MRA_Lausanne"
if [ "$(find "$OUT" -name '*.nii.gz' 2>/dev/null | wc -l)" -ge 20 ]; then
    echo "SKIP: Already have 20+ predictions"
else
    mkdir -p "$OUT"
    nnUNetv2_predict -i "$DS202" -o "$OUT" \
        -d 103 -c 3d_fullres -tr nnUNetTrainer -f 0 \
        --disable_tta --continue_prediction
fi

echo ""
echo "=== All inference finished: $(date) ==="
