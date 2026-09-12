"""
Run TCMN DS104 validation via nnUNetPredictor (single-process, no multiprocessing).
Bypasses nnUNetv2_train --val which has load_checkpoint strict=True issue.
"""
import sys
import os
import json
import torch
import faulthandler
import shutil
import numpy as np
from pathlib import Path

faulthandler.enable()

BASE = '/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset'
sys.path.insert(0, os.path.join(BASE, 'training'))

os.environ.setdefault('nnUNet_raw', os.path.join(BASE, 'preprocessed/nnUNet_raw'))
os.environ.setdefault('nnUNet_preprocessed', os.path.join(BASE, 'preprocessed/nnUNet_preprocessed'))
os.environ.setdefault('nnUNet_results', os.path.join(BASE, 'results/nnUNet'))

from nnUNetTrainerTCMN import nnUNetTrainerTCMN
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder_simple
import tcmn

ds_name = "Dataset104_TopCoW_joint"
plans_path = os.path.join(os.environ['nnUNet_preprocessed'], ds_name, 'nnUNetPlans.json')
ds_path = os.path.join(os.environ['nnUNet_preprocessed'], ds_name, 'dataset.json')
plans = json.load(open(plans_path))
ds_json = json.load(open(ds_path))

trainer = nnUNetTrainerTCMN(plans, '3d_fullres', 0, ds_json,
                            unpack_dataset=False, device=torch.device('cuda'))

ckpt_path = os.path.join(
    os.environ['nnUNet_results'],
    f'{ds_name}/nnUNetTrainerTCMN__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth'
)
print(f"Loading checkpoint: {ckpt_path}")
trainer.load_checkpoint(ckpt_path)
# Disable deep supervision before inference (critical for correct output)
trainer.set_deep_supervision_enabled(False)
trainer.network.eval()

# Wrap network to return only segmentation logits
class SegOnlyWrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, x):
        out = self.net(x)
        if isinstance(out, dict):
            seg = out['seg']
        else:
            seg = out
        if isinstance(seg, (list, tuple)):
            return seg[0]
        return seg

wrapped_network = SegOnlyWrapper(trainer.network)
wrapped_network.eval()

# Sanity check
with torch.no_grad():
    dummy = torch.randn(1, 1, 64, 64, 64, device='cuda')
    with tcmn.modality_context(torch.tensor([0], device='cuda')):
        out = wrapped_network(dummy)
        print(f"[TCMN] Forward sanity: seg sum={out.sum().item():.1f}, finite={torch.isfinite(out).all()}")

# Set up prediction output
val_dir = os.path.join(
    os.environ['nnUNet_results'],
    f'{ds_name}/nnUNetTrainerTCMN__nnUNetPlans__3d_fullres/fold_0/validation'
)
if os.path.exists(val_dir):
    shutil.rmtree(val_dir)
os.makedirs(val_dir, exist_ok=True)

# Get validation cases from the split
splits_path = os.path.join(os.environ['nnUNet_preprocessed'], ds_name, 'splits_final.json')
splits = json.load(open(splits_path))
val_keys = splits[0]['val']  # fold 0 validation keys
print(f"Validation set: {len(val_keys)} cases")

# Find the validation images in raw data
raw_images = os.path.join(os.environ['nnUNet_raw'], ds_name, 'imagesTr')
raw_labels = os.path.join(os.environ['nnUNet_raw'], ds_name, 'labelsTr')

# Set up predictor
plans_manager = PlansManager(plans)
configuration_manager = plans_manager.get_configuration('3d_fullres')

predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=False,
    device=torch.device('cuda'),
    verbose=True,
    verbose_preprocessing=False,
    allow_tqdm=True,
)

predictor.manual_initialization(
    network=wrapped_network,
    plans_manager=plans_manager,
    configuration_manager=configuration_manager,
    parameters=None,
    dataset_json=ds_json,
    trainer_name='nnUNetTrainerTCMN',
    inference_allowed_mirroring_axes=trainer.inference_allowed_mirroring_axes,
)
predictor.list_of_parameters = [wrapped_network.state_dict()]

# Build input file lists for validation cases
input_files = []
output_files = []
for key in val_keys:
    # Find the image file (channel _0000)
    img_path = os.path.join(raw_images, f"{key}_0000.nii.gz")
    if not os.path.exists(img_path):
        print(f"WARNING: Missing image for {key}, skipping")
        continue
    input_files.append([img_path])
    output_files.append(os.path.join(val_dir, f"{key}.nii.gz"))

print(f"Predicting {len(input_files)} validation cases...")

with tcmn.modality_context(torch.tensor([0], device='cuda')):
    predictor.predict_from_files(
        list_of_lists_or_source_folder=input_files,
        output_folder_or_list_of_truncated_output_files=output_files,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
    )

# Compute metrics
print("\nComputing validation metrics...")
label_files = {key: os.path.join(raw_labels, f"{key}.nii.gz") for key in val_keys}
pred_files = {key: os.path.join(val_dir, f"{key}.nii.gz") for key in val_keys
              if os.path.exists(os.path.join(val_dir, f"{key}.nii.gz"))}

# Use nnU-Net's built-in metric computation
labels = ds_json.get('labels', {})
# Get label IDs (exclude background=0)
label_ids = [int(k) for k in labels.keys() if int(k) > 0]

compute_metrics_on_folder_simple(
    folder_ref=raw_labels,
    folder_pred=val_dir,
    labels=label_ids,
    output_file=os.path.join(val_dir, 'summary.json'),
    num_processes=1,  # single process to avoid fork issues
    chill=True,
)

# Print result
summary_path = os.path.join(val_dir, 'summary.json')
if os.path.exists(summary_path):
    summary = json.load(open(summary_path))
    print(f"\n=== TCMN DS104 validation complete ===")
    print(f"Mean foreground Dice: {summary['foreground_mean']['Dice']:.4f}")
else:
    print("WARNING: summary.json not created")
