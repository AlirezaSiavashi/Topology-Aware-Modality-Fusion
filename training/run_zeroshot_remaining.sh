#!/bin/bash
# Continue zero-shot inference — only missing predictions
# Runs on GPU 1 MIG partition 1 (free 20GB partition)
# E0 on all 3 datasets: DONE
# E1 on ISLES2024: partially done (resuming), IXI and Lausanne: not done

BASE=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection
export nnUNet_raw=$BASE/brain_external_dataset/preprocessed/nnUNet_raw
export nnUNet_preprocessed=$BASE/brain_external_dataset/preprocessed/nnUNet_preprocessed
export nnUNet_results=$BASE/brain_external_dataset/results/nnUNet
export CUDA_VISIBLE_DEVICES=MIG-50e7ea79-5c28-5c08-8703-7df708730fd8

PREDICT=$BASE/nnunet_venv/bin/nnUNetv2_predict
EVALUATE=$BASE/nnunet_venv/bin/nnUNetv2_evaluate_folder
LOG_DIR=$BASE/brain_external_dataset/logs
ZEROSHOT=$BASE/brain_external_dataset/results/zeroshot
RAW=$BASE/brain_external_dataset/dataset/15692630

DS_JSON=$nnUNet_results/Dataset104_TopCoW_joint/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation/dataset.json
PLANS=$nnUNet_results/Dataset104_TopCoW_joint/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation/plans.json

echo "=== Resuming zero-shot on MIG partition 1 ===" | tee $LOG_DIR/zeroshot_remaining.log

# ── E1 ISLES2024 (resume — 13 done, 13 remaining, nnUNet skips existing) ──
echo "E1 → ISLES2024 CTA (resuming)" | tee -a $LOG_DIR/zeroshot_remaining.log
$PREDICT \
    -i  $ZEROSHOT/input_ISLES2024 \
    -o  $ZEROSHOT/E1_topology_ISLES2024_CTA \
    -d 104 -c 3d_fullres -f 0 \
    -tr nnUNetTrainerTopologyVessel \
    -chk checkpoint_final.pth --disable_tta \
    2>&1 | tee -a $LOG_DIR/zeroshot_remaining.log

# ── E1 IXI MRA ──────────────────────────────────────────────────────────────
echo "E1 → IXI MRA" | tee -a $LOG_DIR/zeroshot_remaining.log
$PREDICT \
    -i  $ZEROSHOT/input_IXI_HH \
    -o  $ZEROSHOT/E1_topology_IXI_MRA \
    -d 104 -c 3d_fullres -f 0 \
    -tr nnUNetTrainerTopologyVessel \
    -chk checkpoint_final.pth --disable_tta \
    2>&1 | tee -a $LOG_DIR/zeroshot_remaining.log

# ── E1 Lausanne MRA ──────────────────────────────────────────────────────────
echo "E1 → Lausanne MRA" | tee -a $LOG_DIR/zeroshot_remaining.log
$PREDICT \
    -i  $ZEROSHOT/input_Lausanne \
    -o  $ZEROSHOT/E1_topology_Lausanne_MRA \
    -d 104 -c 3d_fullres -f 0 \
    -tr nnUNetTrainerTopologyVessel \
    -chk checkpoint_final.pth --disable_tta \
    2>&1 | tee -a $LOG_DIR/zeroshot_remaining.log

echo "=== All inference done. Evaluating... ===" | tee -a $LOG_DIR/zeroshot_remaining.log

# ── Evaluate E0 (all 3 already predicted) ───────────────────────────────────
for tag in ISLES2024_CTA IXI_MRA Lausanne_MRA; do
    case $tag in
        ISLES2024_CTA) GT=$RAW/CTA_ISLES2024_TUM/cow_seg_labelsTr ;;
        IXI_MRA)       GT=$RAW/MRA_IXI_HH/cow_seg_labelsTr ;;
        Lausanne_MRA)  GT=$RAW/MRA_Lausanne/cow_seg_labelsTr ;;
    esac
    echo "Evaluating E0 $tag" | tee -a $LOG_DIR/zeroshot_remaining.log
    $EVALUATE "$GT" "$ZEROSHOT/E0_baseline_${tag}" \
        -djfile "$DS_JSON" -pfile "$PLANS" \
        2>&1 | tee -a $LOG_DIR/zeroshot_remaining.log
done

# ── Evaluate E1 (all 3) ──────────────────────────────────────────────────────
for tag in ISLES2024_CTA IXI_MRA Lausanne_MRA; do
    case $tag in
        ISLES2024_CTA) GT=$RAW/CTA_ISLES2024_TUM/cow_seg_labelsTr ;;
        IXI_MRA)       GT=$RAW/MRA_IXI_HH/cow_seg_labelsTr ;;
        Lausanne_MRA)  GT=$RAW/MRA_Lausanne/cow_seg_labelsTr ;;
    esac
    echo "Evaluating E1 $tag" | tee -a $LOG_DIR/zeroshot_remaining.log
    $EVALUATE "$GT" "$ZEROSHOT/E1_topology_${tag}" \
        -djfile "$DS_JSON" -pfile "$PLANS" \
        2>&1 | tee -a $LOG_DIR/zeroshot_remaining.log
done

echo "=== DONE ===" | tee -a $LOG_DIR/zeroshot_remaining.log
