#!/bin/bash
#SBATCH --partition=mitarb
#SBATCH --gres=gpu:mitarb:1
#SBATCH --mem=120G
#SBATCH --time=0
#SBATCH --job-name=parallel102
#SBATCH --output=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset/training/slurm_%j.log
#SBATCH --error=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset/training/slurm_%j.log

source /scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/nnunet_venv/bin/activate

BASE=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset

export nnUNet_raw="${BASE}/preprocessed/nnUNet_raw"
export nnUNet_preprocessed="${BASE}/preprocessed/nnUNet_preprocessed"
export nnUNet_results="${BASE}/results/nnUNet"
export PYTHONPATH="${BASE}/training:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0

echo "=== Job started: $(date) ==="
echo "=== Node: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="

# Baseline — plain nnUNetTrainer, resume if checkpoint exists
python -m nnunetv2.run.run_training 102 3d_fullres 0 \
  --c \
  > ${BASE}/training/baseline102.log 2>&1 &
PID_BASELINE=$!
echo "Baseline PID: ${PID_BASELINE}"

# Small delay to avoid simultaneous data loading
sleep 30

# Topology — cbDice + SDF + skeleton heads, fresh start
python -m nnunetv2.run.run_training 102 3d_fullres 0 \
  -tr nnUNetTrainerTopologyVessel \
  > ${BASE}/training/topology102.log 2>&1 &
PID_TOPOLOGY=$!
echo "Topology PID: ${PID_TOPOLOGY}"

echo "Both jobs running. Waiting for both to finish..."
wait ${PID_BASELINE} ${PID_TOPOLOGY}

echo "=== Job finished: $(date) ==="
