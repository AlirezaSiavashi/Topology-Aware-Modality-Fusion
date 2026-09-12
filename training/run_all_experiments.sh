#!/bin/bash
#
# Master experiment launcher for TMI paper ablations.
# Submits SLURM jobs for all remaining experiments.
#
# Current state (2026-03-24):
#   GPU 0: CMTC training running (job 340)
#   GPU 1: MIG slice available (~20GB)
#
# Usage: bash run_all_experiments.sh
#

BASE=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset
TRAIN_DIR=${BASE}/training
SCRIPTS_DIR=${TRAIN_DIR}/slurm_scripts

mkdir -p ${SCRIPTS_DIR}

echo "=== TMI Experiment Launcher ==="
echo "=== $(date) ==="
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: Validation runs for completed models (fast, ~30min each)
# ──────────────────────────────────────────────────────────────────────────────

echo "--- Phase 1: Validation runs ---"

# 1a. TopologyVessel DS102 validation
JOB1=$(sbatch --parsable ${SCRIPTS_DIR}/val_topoVessel_ds102.sh)
echo "Submitted val_topoVessel_ds102: Job ${JOB1}"

# 1b. TCMN DS104 validation
JOB2=$(sbatch --parsable ${SCRIPTS_DIR}/val_tcmn_ds104.sh)
echo "Submitted val_tcmn_ds104: Job ${JOB2}"

# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: Cross-domain zero-shot inference (after val, parallel is fine)
# ──────────────────────────────────────────────────────────────────────────────

echo ""
echo "--- Phase 2: Cross-domain zero-shot inference ---"

# 2a. DS104 baseline → DS200/201/202
JOB3=$(sbatch --parsable --dependency=afterany:${JOB1}:${JOB2} ${SCRIPTS_DIR}/zeroshot_baseline_ds104.sh)
echo "Submitted zeroshot_baseline_ds104: Job ${JOB3}"

# 2b. DS104 TCMN → DS200/201/202
JOB4=$(sbatch --parsable --dependency=afterany:${JOB3} ${SCRIPTS_DIR}/zeroshot_tcmn_ds104.sh)
echo "Submitted zeroshot_tcmn_ds104: Job ${JOB4}"

# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: Training new models on GPU 1 (sequential — one at a time)
# ──────────────────────────────────────────────────────────────────────────────

echo ""
echo "--- Phase 3: New training runs ---"

# 3a. TopologyVessel on DS104 joint (E1 ablation)
JOB5=$(sbatch --parsable --dependency=afterany:${JOB4} ${SCRIPTS_DIR}/train_topoVessel_ds104.sh)
echo "Submitted train_topoVessel_ds104: Job ${JOB5}"

# 3b. MSD Task08 hepatic vessel baseline (DS110)
JOB6=$(sbatch --parsable --dependency=afterany:${JOB5} ${SCRIPTS_DIR}/train_baseline_ds110.sh)
echo "Submitted train_baseline_ds110: Job ${JOB6}"

# 3c. KiPA22 baseline (DS111)
JOB7=$(sbatch --parsable --dependency=afterany:${JOB6} ${SCRIPTS_DIR}/train_baseline_ds111.sh)
echo "Submitted train_baseline_ds111: Job ${JOB7}"

echo ""
echo "=== All jobs submitted ==="
echo "Monitor: squeue -u \$USER"
