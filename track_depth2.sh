#!/bin/bash
# Report depth ordering alongside depth-2 Dice, to test the prediction that
# radius follows segmentability.
cd /scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset
PY=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/nnunet_venv/bin/python3.9
$PY check_depth_ordering.py
$PY - <<'PYEOF'
import re, glob, sys
sys.path.insert(0, "training")
from cmap import COW_DEPTH
n = {1:'BA',2:'R-PCA',3:'L-PCA',4:'R-ICA',5:'R-MCA',6:'L-ICA',7:'L-MCA',
     8:'R-Pcom',9:'L-Pcom',10:'Acom',11:'R-ACA',12:'L-ACA',13:'3rd-A2'}
L = sorted(glob.glob("results/nnUNet/Dataset104_TopCoW_joint/"
                     "nnUNetTrainerCMAP__nnUNetPlans__3d_fullres/fold_0/"
                     "training_log_*.txt"))[-1]
rows = [l for l in open(L) if "Pseudo dice" in l]
print("\ndepth-2 (communicating artery) Dice over training:")
for i in range(0, len(rows), 50):
    v = [float(x) for x in re.findall(r"np\.float32\(([0-9.]+)\)", rows[i])]
    ks = [k for k in COW_DEPTH if COW_DEPTH[k] == 2]
    print(f"  epoch {i:>4}: " + "  ".join(f"{n[k]}={v[k-1]:.3f}" for k in ks if k-1 < len(v)))
PYEOF
