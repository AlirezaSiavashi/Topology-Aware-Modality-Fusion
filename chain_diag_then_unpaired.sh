#!/bin/bash
# GPU0 queue: wait for _NoStateLoss -> short _NoComplexNoState diagnostic -> _Unpaired
#
# The diagnostic keeps num_epochs=1000 so its LR schedule is IDENTICAL to the runs
# it is being compared against -- a shortened schedule decays LR fast and could
# suppress the very divergence we are testing for. We just stop it early instead.
set -euo pipefail
BASE="/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset"
cd "$BASE"; mkdir -p logs
WAIT_PID=2949492; DIAG_EPOCHS=40
echo "[q] waiting for _NoStateLoss (pid ${WAIT_PID}) $(date)"
while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 60; done
echo "[q] free at $(date); launching diagnostic _NoComplexNoState"
sleep 30
TRAINER=nnUNetTrainerM3CoW_NoComplexNoState DATASET=104 GPU=0 FOLD=0 ./run_m3cow.sh
sleep 20
DPID=$(pgrep -f "run_training 104 3d_fullres 0 -tr nnUNetTrainerM3CoW_NoComplexNoState" | head -1)
echo "[q] diagnostic pid ${DPID}; stopping after ${DIAG_EPOCHS} epochs"
L="results/nnUNet/Dataset104_TopCoW_joint/nnUNetTrainerM3CoW_NoComplexNoState__nnUNetPlans__3d_fullres/fold_0"
while kill -0 "${DPID}" 2>/dev/null; do
  f=$(ls -t ${L}/training_log_*.txt 2>/dev/null | head -1)
  [ -n "$f" ] && [ "$(grep -c 'Epoch time' "$f" 2>/dev/null)" -ge "${DIAG_EPOCHS}" ] && break
  sleep 60
done
echo "[q] diagnostic reached ${DIAG_EPOCHS} epochs (or exited) at $(date); verdict:"
f=$(ls -t ${L}/training_log_*.txt 2>/dev/null | head -1)
grep -o "train_loss [-0-9.a-z]*" "$f" 2>/dev/null | tail -3
kill -9 "${DPID}" 2>/dev/null || true
pgrep -P "${DPID}" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 45
echo "[q] launching _Unpaired $(date)"
TRAINER=nnUNetTrainerM3CoW_Unpaired DATASET=104 GPU=0 FOLD=0 ./run_m3cow.sh
