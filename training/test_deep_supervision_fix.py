"""Quick test: verify that disabling deep supervision fixes TCMN predictions."""
import sys
import os
import json
import torch
import numpy as np

BASE = '/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset'
sys.path.insert(0, os.path.join(BASE, 'training'))

os.environ.setdefault('nnUNet_raw', os.path.join(BASE, 'preprocessed/nnUNet_raw'))
os.environ.setdefault('nnUNet_preprocessed', os.path.join(BASE, 'preprocessed/nnUNet_preprocessed'))
os.environ.setdefault('nnUNet_results', os.path.join(BASE, 'results/nnUNet'))

from nnUNetTrainerTCMN import nnUNetTrainerTCMN
import tcmn

ds_name = "Dataset104_TopCoW_joint"
plans = json.load(open(os.path.join(os.environ['nnUNet_preprocessed'], ds_name, 'nnUNetPlans.json')))
ds_json = json.load(open(os.path.join(os.environ['nnUNet_preprocessed'], ds_name, 'dataset.json')))

trainer = nnUNetTrainerTCMN(plans, '3d_fullres', 0, ds_json,
                            unpack_dataset=False, device=torch.device('cuda'))

ckpt_path = os.path.join(
    os.environ['nnUNet_results'],
    f'{ds_name}/nnUNetTrainerTCMN__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth'
)
trainer.load_checkpoint(ckpt_path)

# Test WITH deep supervision (old broken behavior)
trainer.set_deep_supervision_enabled(True)
trainer.network.eval()
with torch.no_grad():
    dummy = torch.randn(1, 1, 64, 64, 64, device='cuda')
    with tcmn.modality_context(torch.tensor([0], device='cuda')):
        out_ds_on = trainer.network(dummy)
        print(f"\n=== Deep supervision ON ===")
        print(f"Output type: {type(out_ds_on)}")
        if isinstance(out_ds_on, dict):
            seg = out_ds_on['seg']
            print(f"Dict with keys: {out_ds_on.keys()}")
        else:
            seg = out_ds_on
        if isinstance(seg, (list, tuple)):
            print(f"List of {len(seg)} tensors, first shape: {seg[0].shape}")
            seg0 = seg[0]
        else:
            seg0 = seg
            print(f"Single tensor shape: {seg0.shape}")
        pred_ds_on = seg0.argmax(dim=1)
        nonzero_ds_on = (pred_ds_on > 0).sum().item()
        print(f"Non-zero voxels: {nonzero_ds_on}")

# Test WITHOUT deep supervision (fixed behavior)
trainer.set_deep_supervision_enabled(False)
trainer.network.eval()
with torch.no_grad():
    with tcmn.modality_context(torch.tensor([0], device='cuda')):
        out_ds_off = trainer.network(dummy)
        print(f"\n=== Deep supervision OFF ===")
        print(f"Output type: {type(out_ds_off)}")
        if isinstance(out_ds_off, dict):
            seg = out_ds_off['seg']
            print(f"Dict with keys: {out_ds_off.keys()}")
        else:
            seg = out_ds_off
        if isinstance(seg, (list, tuple)):
            print(f"List of {len(seg)} tensors, first shape: {seg[0].shape}")
            seg0 = seg[0]
        else:
            seg0 = seg
            print(f"Single tensor shape: {seg0.shape}")
        pred_ds_off = seg0.argmax(dim=1)
        nonzero_ds_off = (pred_ds_off > 0).sum().item()
        print(f"Non-zero voxels: {nonzero_ds_off}")

print(f"\n=== Comparison ===")
print(f"DS ON  non-zero: {nonzero_ds_on}")
print(f"DS OFF non-zero: {nonzero_ds_off}")
print(f"Ratio: {nonzero_ds_on / max(nonzero_ds_off, 1):.2f}x")
