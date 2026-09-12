#!/bin/bash
# Run TCMN-specific tasks that can't use nnUNetv2_predict/train --val
# (strict=False and multiprocessing issues)

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

LOG="${BASE}/training/tcmn_tasks_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== TCMN tasks started: $(date) ==="

# ── Task 1: TCMN zero-shot → DS200 (CTA) ───────────────────────────────
echo ""
echo "=== Task 1/4: TCMN → DS200 (CTA ISLES2024) ==="
OUT="${ZEROSHOT}/DS104_TCMN_v2/DS200_CTA_ISLES2024"
if [ "$(find "$OUT" -name '*.nii.gz' 2>/dev/null | wc -l)" -ge 26 ]; then
    echo "SKIP: Already have 26+ predictions"
else
    python3 ${BASE}/training/run_tcmn_predict.py -i "$DS200" -o "$OUT" -m 0
fi

# ── Task 2: TCMN zero-shot → DS201 (MRA) ───────────────────────────────
echo ""
echo "=== Task 2/4: TCMN → DS201 (MRA IXI HH) ==="
OUT="${ZEROSHOT}/DS104_TCMN_v2/DS201_MRA_IXI_HH"
if [ "$(find "$OUT" -name '*.nii.gz' 2>/dev/null | wc -l)" -ge 20 ]; then
    echo "SKIP: Already have 20+ predictions"
else
    python3 ${BASE}/training/run_tcmn_predict.py -i "$DS201" -o "$OUT" -m 1
fi

# ── Task 3: TCMN zero-shot → DS202 (MRA) ───────────────────────────────
echo ""
echo "=== Task 3/4: TCMN → DS202 (MRA Lausanne) ==="
OUT="${ZEROSHOT}/DS104_TCMN_v2/DS202_MRA_Lausanne"
if [ "$(find "$OUT" -name '*.nii.gz' 2>/dev/null | wc -l)" -ge 20 ]; then
    echo "SKIP: Already have 20+ predictions"
else
    python3 ${BASE}/training/run_tcmn_predict.py -i "$DS202" -o "$OUT" -m 1
fi

# ── Task 4: TCMN DS104 validation ──────────────────────────────────────
echo ""
echo "=== Task 4/4: TCMN DS104 validation ==="
VAL_SUMMARY="${nnUNet_results}/Dataset104_TopCoW_joint/nnUNetTrainerTCMN__nnUNetPlans__3d_fullres/fold_0/validation/summary.json"
if [ -f "$VAL_SUMMARY" ]; then
    echo "SKIP: Already exists"
else
    python3 ${BASE}/training/run_tcmn_val.py
fi

echo ""
echo "=== TCMN tasks finished: $(date) ==="
