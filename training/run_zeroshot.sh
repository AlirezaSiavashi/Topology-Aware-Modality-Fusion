#!/bin/bash
# Zero-shot inference: E0 and E1 (trained on DS104 TopCoW) on 3 unseen datasets
# Datasets: ISLES2024 CTA, IXI MRA, Lausanne MRA
# No fine-tuning — direct inference with TopCoW-trained checkpoints

BASE=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection
export nnUNet_raw=$BASE/brain_external_dataset/preprocessed/nnUNet_raw
export nnUNet_preprocessed=$BASE/brain_external_dataset/preprocessed/nnUNet_preprocessed
export nnUNet_results=$BASE/brain_external_dataset/results/nnUNet
export CUDA_VISIBLE_DEVICES=0

PYTHON=$BASE/nnunet_venv/bin/python3.9
PREDICT=$BASE/nnunet_venv/bin/nnUNetv2_predict
LOG_DIR=$BASE/brain_external_dataset/logs
ZEROSHOT_DIR=$BASE/brain_external_dataset/results/zeroshot
mkdir -p "$LOG_DIR" "$ZEROSHOT_DIR"

RAW_DATA=$BASE/brain_external_dataset/dataset/15692630

# ── Step 1: prepare nnUNet-format input folders ──────────────────────────────
echo "=== Preparing zero-shot input folders ==="

prepare_input() {
    local src_dir=$1
    local dst_dir=$2
    local suffix=$3   # _0000 channel suffix for nnUNet
    mkdir -p "$dst_dir"
    for f in "$src_dir"/*.nii.gz; do
        base=$(basename "$f" .nii.gz)
        # nnUNet expects filename_0000.nii.gz for single-channel input
        ln -sf "$f" "$dst_dir/${base}_0000.nii.gz" 2>/dev/null || \
        cp "$f" "$dst_dir/${base}_0000.nii.gz"
    done
    echo "  Prepared $(ls $dst_dir | wc -l) cases in $dst_dir"
}

prepare_input "$RAW_DATA/CTA_ISLES2024_TUM/imagesTr"  "$ZEROSHOT_DIR/input_ISLES2024"  "_0000"
prepare_input "$RAW_DATA/MRA_IXI_HH/imagesTr"         "$ZEROSHOT_DIR/input_IXI_HH"    "_0000"
prepare_input "$RAW_DATA/MRA_Lausanne/imagesTr"        "$ZEROSHOT_DIR/input_Lausanne"  "_0000"

# ── Step 2: run inference for each model × dataset ───────────────────────────

run_predict() {
    local model_name=$1
    local trainer=$2
    local chk=$3
    local input_dir=$4
    local dataset_tag=$5
    local out_dir=$ZEROSHOT_DIR/${model_name}_${dataset_tag}
    local log=$LOG_DIR/zeroshot_${model_name}_${dataset_tag}.log

    mkdir -p "$out_dir"
    echo ""
    echo "=== ${model_name} → ${dataset_tag} ==="
    echo "    Input:  $input_dir ($(ls $input_dir | wc -l) files)"
    echo "    Output: $out_dir"
    echo "    Log:    $log"

    $PREDICT \
        -i  "$input_dir" \
        -o  "$out_dir" \
        -d  104 \
        -c  3d_fullres \
        -f  0 \
        -tr "$trainer" \
        -chk "$chk" \
        --disable_tta \
        2>&1 | tee "$log"

    echo "  → Inference done for ${model_name} on ${dataset_tag}"
}

# E0 baseline
run_predict "E0_baseline"      "nnUNetTrainer"              "checkpoint_final.pth" \
    "$ZEROSHOT_DIR/input_ISLES2024"  "ISLES2024_CTA"
run_predict "E0_baseline"      "nnUNetTrainer"              "checkpoint_final.pth" \
    "$ZEROSHOT_DIR/input_IXI_HH"    "IXI_MRA"
run_predict "E0_baseline"      "nnUNetTrainer"              "checkpoint_final.pth" \
    "$ZEROSHOT_DIR/input_Lausanne"  "Lausanne_MRA"

# E1 TopologyVessel
run_predict "E1_topology"      "nnUNetTrainerTopologyVessel" "checkpoint_final.pth" \
    "$ZEROSHOT_DIR/input_ISLES2024"  "ISLES2024_CTA"
run_predict "E1_topology"      "nnUNetTrainerTopologyVessel" "checkpoint_final.pth" \
    "$ZEROSHOT_DIR/input_IXI_HH"    "IXI_MRA"
run_predict "E1_topology"      "nnUNetTrainerTopologyVessel" "checkpoint_final.pth" \
    "$ZEROSHOT_DIR/input_Lausanne"  "Lausanne_MRA"

echo ""
echo "=== All inference done. Running evaluation... ==="

# ── Step 3: evaluate against ground truth ────────────────────────────────────

evaluate() {
    local pred_dir=$1
    local gt_dir=$2
    local dataset_json=$3
    local plans_json=$4
    local tag=$5
    local log=$LOG_DIR/zeroshot_eval_${tag}.log

    echo ""
    echo "=== Evaluating: $tag ==="
    $BASE/nnunet_venv/bin/nnUNetv2_evaluate_folder \
        "$gt_dir" \
        "$pred_dir" \
        -djfile "$dataset_json" \
        -pfile  "$plans_json" \
        2>&1 | tee "$log"
}

DS_JSON=$nnUNet_results/Dataset104_TopCoW_joint/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation/dataset.json
PLANS_JSON=$nnUNet_results/Dataset104_TopCoW_joint/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation/plans.json

evaluate "$ZEROSHOT_DIR/E0_baseline_ISLES2024_CTA"  \
    "$RAW_DATA/CTA_ISLES2024_TUM/cow_seg_labelsTr"  \
    "$DS_JSON" "$PLANS_JSON"  "E0_ISLES2024"

evaluate "$ZEROSHOT_DIR/E0_baseline_IXI_MRA"       \
    "$RAW_DATA/MRA_IXI_HH/cow_seg_labelsTr"         \
    "$DS_JSON" "$PLANS_JSON"  "E0_IXI"

evaluate "$ZEROSHOT_DIR/E0_baseline_Lausanne_MRA"  \
    "$RAW_DATA/MRA_Lausanne/cow_seg_labelsTr"        \
    "$DS_JSON" "$PLANS_JSON"  "E0_Lausanne"

evaluate "$ZEROSHOT_DIR/E1_topology_ISLES2024_CTA" \
    "$RAW_DATA/CTA_ISLES2024_TUM/cow_seg_labelsTr"  \
    "$DS_JSON" "$PLANS_JSON"  "E1_ISLES2024"

evaluate "$ZEROSHOT_DIR/E1_topology_IXI_MRA"       \
    "$RAW_DATA/MRA_IXI_HH/cow_seg_labelsTr"         \
    "$DS_JSON" "$PLANS_JSON"  "E1_IXI"

evaluate "$ZEROSHOT_DIR/E1_topology_Lausanne_MRA"  \
    "$RAW_DATA/MRA_Lausanne/cow_seg_labelsTr"        \
    "$DS_JSON" "$PLANS_JSON"  "E1_Lausanne"

echo ""
echo "=== ZERO-SHOT EVALUATION COMPLETE ==="
echo "Results in: $ZEROSHOT_DIR"
