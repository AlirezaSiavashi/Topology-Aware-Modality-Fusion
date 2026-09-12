#!/usr/bin/env bash
# ============================================================
# nnunet_env.sh  –  run nnUNetv2 commands
#
# Usage:
#   bash nnunet_env.sh plan 104          # fingerprint + plan + preprocess Dataset104
#   bash nnunet_env.sh plan 100 101      # multiple datasets
#   bash nnunet_env.sh train 104 0       # train Dataset104 fold 0
#   bash nnunet_env.sh help              # show this message
#
# Or source to set env vars and use CLI directly:
#   source nnunet_env.sh
#   nnUNetv2_plan_and_preprocess -d 104 --verify_dataset_integrity -np 16 -c 3d_fullres
# ============================================================

VENV="/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/nnunet_venv"
BASE="/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset"
PY="$VENV/bin/python3.9"

export nnUNet_raw="${BASE}/preprocessed/nnUNet_raw"
export nnUNet_preprocessed="${BASE}/preprocessed/nnUNet_preprocessed"
export nnUNet_results="${BASE}/results/nnUNet"
export PATH="$VENV/bin:$PATH"
export PIP_NO_USER_INSTALL=1

mkdir -p "${nnUNet_preprocessed}" "${nnUNet_results}"

if [[ "${1}" == "plan" ]]; then
    shift
    echo "Running plan_and_preprocess on datasets: $@ ..."
    "$PY" -m nnunetv2.experiment_planning.plan_and_preprocess_entrypoints \
        -d "$@" --verify_dataset_integrity -np 16 -c 3d_fullres
elif [[ "${1}" == "train" && -n "${2}" ]]; then
    DATASET="${2}"
    FOLD="${3:-0}"
    TRAINER="${4:-nnUNetTrainer}"
    echo "Training Dataset${DATASET} fold ${FOLD} trainer=${TRAINER} ..."
    "$PY" -m nnunetv2.run.run_training "${DATASET}" 3d_fullres "${FOLD}" -tr "${TRAINER}" --npz
else
    echo "nnUNet_raw          = ${nnUNet_raw}"
    echo "nnUNet_preprocessed = ${nnUNet_preprocessed}"
    echo "nnUNet_results      = ${nnUNet_results}"
    echo "Python              = $PY"
    echo ""
    echo "Usage:"
    echo "  bash nnunet_env.sh plan 104          # plan+preprocess Dataset104 (TopCoW joint)"
    echo "  bash nnunet_env.sh plan 100 101      # TopBrain CT + MR"
    echo "  bash nnunet_env.sh train 104 0       # train Dataset104 fold 0"
fi
