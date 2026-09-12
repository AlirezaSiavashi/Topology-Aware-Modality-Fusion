#!/bin/bash
cd /scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset
for t in nnUNetTrainerMFSSM nnUNetTrainerMFSSM_SharedAll; do
  D=results/nnUNet/Dataset104_TopCoW_joint/${t}__nnUNetPlans__3d_fullres/fold_0
  until [ "$(ls $D/validation/*.nii.gz 2>/dev/null|wc -l)" -ge 50 ] \
     || ! ps -eo cmd | grep -q "[-]tr $t --npz"; do sleep 300; done
done
echo "=== both MF-SSM runs finished $(date) ==="
for t in nnUNetTrainerMFSSM nnUNetTrainerMFSSM_SharedAll; do
  D=results/nnUNet/Dataset104_TopCoW_joint/${t}__nnUNetPlans__3d_fullres/fold_0
  L=$(ls -t $D/training_log_*.txt 2>/dev/null | head -1)
  echo "$t: epochs=$(grep -c 'Epoch time' "$L" 2>/dev/null) errors=$(grep -cE 'Traceback|RuntimeError' "$L" 2>/dev/null)"
done
