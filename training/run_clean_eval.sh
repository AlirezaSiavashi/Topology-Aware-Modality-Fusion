#!/bin/bash
# Clean evaluation on val-only 50 cases for E0, E1, TCMN
# Uses fold 0 val split, evaluates against ground truth labels

BASE=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection
export nnUNet_raw=$BASE/brain_external_dataset/preprocessed/nnUNet_raw
export nnUNet_preprocessed=$BASE/brain_external_dataset/preprocessed/nnUNet_preprocessed
export nnUNet_results=$BASE/brain_external_dataset/results/nnUNet
export CUDA_VISIBLE_DEVICES=MIG-7acbdec9-a39b-5bf1-a5b1-f02688006c7c

NNUNET=$BASE/nnunet_venv/bin/nnUNetv2_predict
EVAL=$BASE/nnunet_venv/bin/nnUNetv2_evaluate_folder
PYTHON=$BASE/nnunet_venv/bin/python3.9

RAW=$BASE/brain_external_dataset/preprocessed/nnUNet_raw/Dataset104_TopCoW_joint
EVAL_DIR=$BASE/brain_external_dataset/results/clean_eval
LOG_DIR=$BASE/brain_external_dataset/logs
mkdir -p "$EVAL_DIR" "$LOG_DIR"

# ── Step 1: Build val-only input folder (symlinks) ───────────────────────────
VAL_INPUT=$EVAL_DIR/val_images
VAL_LABELS=$EVAL_DIR/val_labels
mkdir -p "$VAL_INPUT" "$VAL_LABELS"

$PYTHON - << 'PYEOF'
import json, os, pathlib

base = "/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection"
raw  = f"{base}/brain_external_dataset/preprocessed/nnUNet_raw/Dataset104_TopCoW_joint"
eval_dir = f"{base}/brain_external_dataset/results/clean_eval"

with open(f"{base}/brain_external_dataset/preprocessed/nnUNet_preprocessed/Dataset104_TopCoW_joint/splits_final.json") as f:
    val_cases = json.load(f)[0]['val']

img_dir = pathlib.Path(f"{eval_dir}/val_images")
lbl_dir = pathlib.Path(f"{eval_dir}/val_labels")
img_dir.mkdir(exist_ok=True)
lbl_dir.mkdir(exist_ok=True)

n_img, n_lbl = 0, 0
for case in val_cases:
    src_img = pathlib.Path(raw) / "imagesTr" / f"{case}_0000.nii.gz"
    src_lbl = pathlib.Path(raw) / "labelsTr" / f"{case}.nii.gz"
    dst_img = img_dir / f"{case}_0000.nii.gz"
    dst_lbl = lbl_dir / f"{case}.nii.gz"
    if src_img.exists() and not dst_img.exists():
        os.symlink(src_img, dst_img); n_img += 1
    if src_lbl.exists() and not dst_lbl.exists():
        os.symlink(src_lbl, dst_lbl); n_lbl += 1

print(f"Symlinked {n_img} images, {n_lbl} labels")
PYEOF

echo "Val input: $(ls $VAL_INPUT | wc -l) images"
echo "Val labels: $(ls $VAL_LABELS | wc -l) labels"

# ── Step 2: Predict with each model ──────────────────────────────────────────
run_predict() {
    local TRAINER=$1
    local OUT=$2
    local EXTRA=${3:-""}
    mkdir -p "$OUT"
    echo ""
    echo ">>> Predicting with $TRAINER → $OUT"
    $NNUNET \
        -i "$VAL_INPUT" \
        -o "$OUT" \
        -d 104 -c 3d_fullres -f 0 \
        -tr "$TRAINER" \
        --disable_tta \
        -chk checkpoint_best \
        $EXTRA \
        2>&1
}

# E0
run_predict "nnUNetTrainer" \
    "$EVAL_DIR/pred_E0"

# E1
run_predict "nnUNetTrainerTopologyVessel" \
    "$EVAL_DIR/pred_E1"

# TCMN
run_predict "nnUNetTrainerTCMN" \
    "$EVAL_DIR/pred_TCMN"

# ── Step 3: Evaluate each against ground truth ───────────────────────────────
echo ""
echo ">>> Evaluating predictions..."

for MODEL in E0 E1 TCMN; do
    echo ""
    echo "=== $MODEL ==="
    $EVAL \
        "$EVAL_DIR/pred_${MODEL}" \
        "$VAL_LABELS" \
        -djfile "$RAW/dataset.json" \
        -pfile  "$nnUNet_results/Dataset104_TopCoW_joint/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation/predict_from_raw_data_args.json" \
        2>&1 | tail -5
done

echo ""
echo "Done. Results in $EVAL_DIR/pred_*/summary.json"
