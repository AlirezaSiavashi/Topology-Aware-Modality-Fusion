#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --job-name=zs_base104
#SBATCH --output=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset/training/slurm_scripts/zs_base104_%j.log
#SBATCH --error=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset/training/slurm_scripts/zs_base104_%j.log

source /scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/nnunet_venv/bin/activate

BASE=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset
export nnUNet_raw="${BASE}/preprocessed/nnUNet_raw"
export nnUNet_preprocessed="${BASE}/preprocessed/nnUNet_preprocessed"
export nnUNet_results="${BASE}/results/nnUNet"
export PYTHONPATH="${BASE}/training:${PYTHONPATH}"

echo "=== Zero-shot inference: DS104 baseline → cross-domain ==="
echo "=== $(date) ==="

OUTBASE="${BASE}/results/zeroshot/DS104_baseline"

# DS200: CTA ISLES2024 (26 cases)
echo "--- Predicting DS200 CTA_ISLES2024 ---"
mkdir -p "${OUTBASE}/DS200_CTA_ISLES2024"
nnUNetv2_predict \
    -i "${BASE}/preprocessed/nnUNet_raw/Dataset200_CrossDomain_CTA_ISLES2024/imagesTr" \
    -o "${OUTBASE}/DS200_CTA_ISLES2024" \
    -d 104 -c 3d_fullres -tr nnUNetTrainer -f 0 \
    --disable_tta --continue_prediction

# DS201: MRA IXI_HH (20 cases)
echo "--- Predicting DS201 MRA_IXI_HH ---"
mkdir -p "${OUTBASE}/DS201_MRA_IXI_HH"
nnUNetv2_predict \
    -i "${BASE}/preprocessed/nnUNet_raw/Dataset201_CrossDomain_MRA_IXI_HH/imagesTr" \
    -o "${OUTBASE}/DS201_MRA_IXI_HH" \
    -d 104 -c 3d_fullres -tr nnUNetTrainer -f 0 \
    --disable_tta --continue_prediction

# DS202: MRA Lausanne (20 cases)
echo "--- Predicting DS202 MRA_Lausanne ---"
mkdir -p "${OUTBASE}/DS202_MRA_Lausanne"
nnUNetv2_predict \
    -i "${BASE}/preprocessed/nnUNet_raw/Dataset202_CrossDomain_MRA_Lausanne/imagesTr" \
    -o "${OUTBASE}/DS202_MRA_Lausanne" \
    -d 104 -c 3d_fullres -tr nnUNetTrainer -f 0 \
    --disable_tta --continue_prediction

echo "=== Done: $(date) ==="
