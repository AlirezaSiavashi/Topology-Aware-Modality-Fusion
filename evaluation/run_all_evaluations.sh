#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Run all evaluations for the TMI paper
# Usage: bash run_all_evaluations.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

BASE=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset
EVAL_DIR=${BASE}/evaluation
RESULTS_DIR=${EVAL_DIR}/results

source /scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/nnunet_venv/bin/activate

export nnUNet_raw="${BASE}/preprocessed/nnUNet_raw"
export nnUNet_results="${BASE}/results/nnUNet"
export PYTHONPATH="${BASE}/training:${PYTHONPATH}"

mkdir -p ${RESULTS_DIR}

SCRIPT="${EVAL_DIR}/evaluate.py"
WORKERS=8

echo "========================================================"
echo "  TMI Paper Evaluation Suite"
echo "  $(date)"
echo "========================================================"

# ─────────────────────────────────────────────────────────────────────────────
# E0: Baseline DS102 — in-domain CTA validation
# Ground truth is inside the nnUNet validation folder
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[E0] Baseline CTA — in-domain validation (DS102)"
python3 ${SCRIPT} \
  --pred_dir   ${nnUNet_results}/Dataset102_TopCoW_CT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation \
  --gt_dir     ${nnUNet_raw}/Dataset102_TopCoW_CT/labelsTr \
  --dataset_json ${nnUNet_raw}/Dataset102_TopCoW_CT/dataset.json \
  --output     ${RESULTS_DIR}/E0_baseline_DS102_indomain.csv \
  --num_workers ${WORKERS} \
  2>&1 | tee ${RESULTS_DIR}/E0_baseline_DS102_indomain.log

# ─────────────────────────────────────────────────────────────────────────────
# E0 cross-domain: Baseline CTA → MRA (DS103)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[E0-xdomain] Baseline CTA → MRA zero-shot (DS102 model → DS103 test)"
python3 ${SCRIPT} \
  --pred_dir   ${BASE}/predictions/baseline102/DS103 \
  --gt_dir     ${nnUNet_raw}/Dataset103_TopCoW_MR/labelsTr \
  --dataset_json ${nnUNet_raw}/Dataset103_TopCoW_MR/dataset.json \
  --output     ${RESULTS_DIR}/E0_baseline_CTA2MRA_crossdomain.csv \
  --num_workers ${WORKERS} \
  2>&1 | tee ${RESULTS_DIR}/E0_baseline_CTA2MRA_crossdomain.log

# ─────────────────────────────────────────────────────────────────────────────
# E3: TCMN v1 → MRA (DS104 model → DS103)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[E3] TCMN joint → MRA (DS104 model → DS103 test)"
python3 ${SCRIPT} \
  --pred_dir   ${BASE}/predictions/tcmn104/DS103 \
  --gt_dir     ${nnUNet_raw}/Dataset103_TopCoW_MR/labelsTr \
  --dataset_json ${nnUNet_raw}/Dataset103_TopCoW_MR/dataset.json \
  --output     ${RESULTS_DIR}/E3_TCMN_DS104_2_DS103.csv \
  --num_workers ${WORKERS} \
  2>&1 | tee ${RESULTS_DIR}/E3_TCMN_DS104_2_DS103.log

# ─────────────────────────────────────────────────────────────────────────────
# E1: Topology DS102 — add this block when topology training finishes
# Uncomment when predictions/topology102/DS103/ exists
# ─────────────────────────────────────────────────────────────────────────────
# echo ""
# echo "[E1] Topology DS102 → MRA cross-domain"
# python3 ${SCRIPT} \
#   --pred_dir   ${BASE}/predictions/topology102/DS103 \
#   --gt_dir     ${nnUNet_raw}/Dataset103_TopCoW_MR/labelsTr \
#   --dataset_json ${nnUNet_raw}/Dataset103_TopCoW_MR/dataset.json \
#   --output     ${RESULTS_DIR}/E1_topology_DS102_2_DS103.csv \
#   --num_workers ${WORKERS} \
#   2>&1 | tee ${RESULTS_DIR}/E1_topology_DS102_2_DS103.log

# ─────────────────────────────────────────────────────────────────────────────
# E1+E3: TCMN v2 (topology + TCMN) — uncomment when training finishes
# ─────────────────────────────────────────────────────────────────────────────
# echo ""
# echo "[E1+E3] TCMN v2 (topology+TCMN) → MRA"
# python3 ${SCRIPT} \
#   --pred_dir   ${BASE}/predictions/tcmn104_v2/DS103 \
#   --gt_dir     ${nnUNet_raw}/Dataset103_TopCoW_MR/labelsTr \
#   --dataset_json ${nnUNet_raw}/Dataset103_TopCoW_MR/dataset.json \
#   --output     ${RESULTS_DIR}/E4_TCMNv2_DS104_2_DS103.csv \
#   --num_workers ${WORKERS} \
#   2>&1 | tee ${RESULTS_DIR}/E4_TCMNv2_DS104_2_DS103.log

# ─────────────────────────────────────────────────────────────────────────────
# Print final summary table
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  FINAL SUMMARY (TMI Table)"
echo "========================================================"
python3 - <<'PYEOF'
import json, glob, os

results_dir = "/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset/evaluation/results"

experiments = [
    ("E0_baseline_DS102_indomain",      "E0  Baseline CTA (in-domain)  "),
    ("E0_baseline_CTA2MRA_crossdomain", "E0  Baseline CTA→MRA (0-shot) "),
    ("E3_TCMN_DS104_2_DS103",           "E3  TCMN joint→MRA (0-shot)   "),
    ("E1_topology_DS102_2_DS103",       "E1  Topology CTA→MRA (0-shot) "),
    ("E4_TCMNv2_DS104_2_DS103",         "E1+E3 TCMN-v2→MRA (0-shot)   "),
]

print(f"\n{'Experiment':<35} {'Dice':>7} {'clDice':>8} {'HD95':>8} {'Betti0':>8}")
print("-" * 70)

for fname, label in experiments:
    path = os.path.join(results_dir, f"{fname}.summary.json")
    if not os.path.exists(path):
        print(f"{label} {'(not yet run)':>35}")
        continue
    with open(path) as f:
        s = json.load(f)
    d  = s.get("mean_Dice_mean",   "nan")
    cl = s.get("mean_clDice_mean", "nan")
    h  = s.get("mean_HD95_mean",   "nan")
    b  = s.get("mean_Betti0_mean", "nan")
    print(f"{label} "
          f"{str(round(d,4)) if d!='nan' else 'N/A':>7} "
          f"{str(round(cl,4)) if cl!='nan' else 'N/A':>8} "
          f"{str(round(h,2)) if h!='nan' else 'N/A':>8} "
          f"{str(round(b,2)) if b!='nan' else 'N/A':>8}")

print("-" * 70)
PYEOF

echo ""
echo "All evaluation results saved to: ${RESULTS_DIR}/"
echo "Done: $(date)"
