"""
Convert KiPA22 dataset to nnUNet format as Dataset111_KiPA22.
Labels used: 1=renal_artery (original 3), 2=renal_vein (original 4).
Kidney (1) and tumor (2) labels are discarded.
70 train cases used; open_image has no labels so skipped.
"""

import json
import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

BASE  = Path("/scratch/siyavash/Alireza_thesis/external_dataset/"
             "rsna-intracranial-aneurysm-detection/brain_external_dataset")
SRC   = BASE / "dataset" / "KiPA22" / "train"
DEST  = BASE / "preprocessed" / "nnUNet_raw" / "Dataset111_KiPA22"

IMAGES_DIR = DEST / "imagesTr"
LABELS_DIR = DEST / "labelsTr"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)

src_images = sorted(SRC.glob("image/*.nii.gz"),
                    key=lambda p: int(p.name.replace(".nii.gz", "")))
src_labels = sorted(SRC.glob("label/*.nii.gz"),
                    key=lambda p: int(p.name.replace(".nii.gz", "")))

assert len(src_images) == len(src_labels) == 70, \
    f"Expected 70 pairs, got {len(src_images)} images and {len(src_labels)} labels"

converted = 0
for img_path, lbl_path in zip(src_images, src_labels):
    case_id = f"kipa_{int(img_path.name.replace('.nii.gz','')):03d}"

    # --- image: just copy with _0000 suffix ---
    dst_img = IMAGES_DIR / f"{case_id}_0000.nii.gz"
    shutil.copy2(img_path, dst_img)

    # --- label: remap 3→1 (artery), 4→2 (vein), everything else→0 ---
    lbl_nib = nib.load(lbl_path)
    lbl_data = lbl_nib.get_fdata().astype(np.int32)

    new_lbl = np.zeros_like(lbl_data)
    new_lbl[lbl_data == 3] = 1   # renal artery
    new_lbl[lbl_data == 4] = 2   # renal vein

    new_nib = nib.Nifti1Image(new_lbl.astype(np.uint8),
                               lbl_nib.affine, lbl_nib.header)
    new_nib.set_data_dtype(np.uint8)
    nib.save(new_nib, LABELS_DIR / f"{case_id}.nii.gz")

    converted += 1
    print(f"  {case_id}: artery={int((new_lbl==1).sum())}, "
          f"vein={int((new_lbl==2).sum())} voxels")

# --- dataset.json ---
dataset_json = {
    "channel_names": {"0": "CT"},
    "labels": {
        "background":    0,
        "renal_artery":  1,
        "renal_vein":    2,
    },
    "numTraining": converted,
    "file_ending": ".nii.gz",
    "name": "KiPA22",
    "description": "KiPA22 renal vessel segmentation (artery + vein), 70 CTA cases",
    "reference": "https://zenodo.org/record/6361938",
    "licence": "CC BY 4.0",
    "dataset_id": 111,
    "overwrite_image_reader_writer": "SimpleITKIO",
}

with open(DEST / "dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=2)

print(f"\nDone. {converted} cases written to {DEST}")
print("Next step: nnUNetv2_plan_and_preprocess -d 111 -c 3d_fullres")
