#!/bin/bash
# _PairedHalf crashed at epoch 0 with "background workers no longer alive" while
# three trainings spawned workers at once. Stagger the start rather than repeat it.
set -euo pipefail
cd /scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset
echo "[retry] delaying 12 min to clear worker-spawn contention $(date)"
sleep 720
echo "[retry] launching _PairedHalf on GPU 1 $(date)"
TRAINER=nnUNetTrainerM3CoW_PairedHalf DATASET=104 GPU=1 FOLD=0 ./run_m3cow.sh
sleep 300
P=$(pgrep -f "run_training 104 3d_fullres 0 -tr nnUNetTrainerM3CoW_PairedHalf" | head -1)
if [ -z "$P" ]; then echo "[retry] FAILED to start again"; else echo "[retry] alive, pid $P"; fi
