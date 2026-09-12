#!/bin/bash
# 5-fold CV of M3CoW vs nnU-Net baseline (folds 1-4; fold 0 already done).
#
# Why: on the held-out eval the fold-0 gain over baseline is +0.0199 at p=0.097.
# Seeds cannot fix that -- they shrink variance but do not grow the effect.
# Folds do: n goes 50 -> 250 paired cases, ~2.2x the power, and 5-fold is the
# standard nnU-Net protocol reviewers already expect.
#
# Jobs are ordered in MATCHED PAIRS so that if this is interrupted, every
# completed fold still has both arms and is usable.
#
# Concurrency 3: each job holds ~38 GB RSS and the box has ~134 GB free.
# Overcommitting is what OOM-killed _ThetaLow at epoch 708.
set -uo pipefail
cd /scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset
DEVICES=(0 MIG-7acbdec9-a39b-5bf1-a5b1-f02688006c7c MIG-50e7ea79-5c28-5c08-8703-7df708730fd8)
JOBS=(
  "nnUNetTrainerM3CoW 1" "nnUNetTrainer 1"
  "nnUNetTrainerM3CoW 2" "nnUNetTrainer 2"
  "nnUNetTrainerM3CoW 3" "nnUNetTrainer 3"
  "nnUNetTrainerM3CoW 4" "nnUNetTrainer 4"
)
MIN_FREE_GB=45
declare -A PID_OF
i=0
while [ $i -lt ${#JOBS[@]} ] || [ ${#PID_OF[@]} -gt 0 ]; do
  # reap finished
  for d in "${!PID_OF[@]}"; do
    if ! kill -0 "${PID_OF[$d]}" 2>/dev/null; then
      echo "[sched] device $d freed (pid ${PID_OF[$d]}) $(date)"
      unset "PID_OF[$d]"
    fi
  done
  # launch while a device is free, jobs remain, and RAM allows
  for d in "${DEVICES[@]}"; do
    [ $i -ge ${#JOBS[@]} ] && break
    [ -n "${PID_OF[$d]:-}" ] && continue
    FREE=$(free -g | awk '/^Mem:/{print $7}')
    if [ "$FREE" -lt "$MIN_FREE_GB" ]; then
      echo "[sched] only ${FREE}GB free, need ${MIN_FREE_GB}GB -- holding $(date)"
      break
    fi
    set -- ${JOBS[$i]}
    TR=$1; FOLD=$2
    echo "[sched] launching $TR fold $FOLD on $d  (${FREE}GB free) $(date)"
    TRAINER="$TR" DATASET=104 GPU="$d" FOLD="$FOLD" ./run_m3cow.sh >> logs/sched_5fold_launch.log 2>&1
    sleep 25
    P=$(pgrep -f "run_training 104 3d_fullres ${FOLD} -tr ${TR}( |$)" | head -1)
    if [ -z "$P" ]; then echo "[sched] FAILED to start $TR fold $FOLD"; else PID_OF[$d]=$P; echo "[sched]   pid $P"; fi
    i=$((i+1))
    sleep 120     # stagger worker spawn; simultaneous spawn crashed _PairedHalf before
  done
  sleep 120
done
echo "[sched] ALL 8 RUNS COMPLETE $(date)"
