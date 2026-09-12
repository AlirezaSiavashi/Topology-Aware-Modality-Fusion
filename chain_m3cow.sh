#!/bin/bash
# Wait for a running training to exit, then launch the next one on that GPU.
#   WAIT_PID=12345 TRAINER=nnUNetTrainerM3CoW_Unpaired GPU=0 ./chain_m3cow.sh
#
# Waits on the PID rather than on a log line: a run that CRASHES still frees the
# GPU, and we want the queued job to start in that case too.
set -euo pipefail
: "${WAIT_PID:?set WAIT_PID}"; : "${TRAINER:?set TRAINER}"; GPU=${GPU:-0}
BASE="/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset"
cd "$BASE"; mkdir -p logs
echo "[chain] waiting for pid ${WAIT_PID} to exit, then ${TRAINER} on GPU ${GPU} ($(date))"
while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 60; done
echo "[chain] pid ${WAIT_PID} gone at $(date); launching ${TRAINER}"
sleep 30                      # let CUDA memory actually be released
TRAINER="${TRAINER}" DATASET="${DATASET:-104}" GPU="${GPU}" FOLD="${FOLD:-0}" ./run_m3cow.sh
