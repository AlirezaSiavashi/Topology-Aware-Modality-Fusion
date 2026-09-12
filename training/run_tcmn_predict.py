"""
Custom TCMN prediction script that loads checkpoint with strict=False.
Bypasses nnUNetv2_predict which uses strict=True and fails on TCMN alias keys.

Usage:
    python run_tcmn_predict.py -i /path/to/images -o /path/to/output [-d 104]
"""
import multiprocessing
multiprocessing.set_start_method('fork', force=True)

import sys
import os
import json
import argparse
import torch
import faulthandler
from pathlib import Path

faulthandler.enable()

BASE = '/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset'
sys.path.insert(0, os.path.join(BASE, 'training'))

os.environ.setdefault('nnUNet_raw', os.path.join(BASE, 'preprocessed/nnUNet_raw'))
os.environ.setdefault('nnUNet_preprocessed', os.path.join(BASE, 'preprocessed/nnUNet_preprocessed'))
os.environ.setdefault('nnUNet_results', os.path.join(BASE, 'results/nnUNet'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='Input folder with images')
    parser.add_argument('--output', '-o', required=True, help='Output folder for predictions')
    parser.add_argument('--dataset', '-d', type=int, default=104,
                        help='Training dataset ID (default: 104)')
    parser.add_argument('--checkpoint', '-c', default=None,
                        help='Checkpoint path (default: checkpoint_final.pth)')
    parser.add_argument('--modality', '-m', type=int, default=0,
                        help='Modality context: 0=CTA, 1=MRA (default: 0)')
    args = parser.parse_args()

    from nnUNetTrainerTCMN import nnUNetTrainerTCMN
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    import tcmn

    ds_name = f"Dataset{args.dataset}_TopCoW_joint"
    plans_path = os.path.join(os.environ['nnUNet_preprocessed'], ds_name, 'nnUNetPlans.json')
    ds_path = os.path.join(os.environ['nnUNet_preprocessed'], ds_name, 'dataset.json')
    plans = json.load(open(plans_path))
    ds_json = json.load(open(ds_path))

    # Instantiate trainer to get the network with TCMN
    trainer = nnUNetTrainerTCMN(plans, '3d_fullres', 0, ds_json,
                                unpack_dataset=False, device=torch.device('cuda'))

    # Load checkpoint with strict=False (handles alias keys)
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = os.path.join(
            os.environ['nnUNet_results'],
            f'{ds_name}/nnUNetTrainerTCMN__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth'
        )
    print(f"Loading checkpoint: {ckpt_path}")
    trainer.load_checkpoint(ckpt_path)
    # Disable deep supervision before inference (critical for correct output)
    trainer.set_deep_supervision_enabled(False)
    trainer.network.eval()

    # Create output dir
    os.makedirs(args.output, exist_ok=True)

    # Wrap network to return only segmentation logits (not the full dict
    # with sdf/skel heads). nnUNetPredictor expects a plain tensor output.
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
            # If deep supervision (list), return the full-res output only
            if isinstance(seg, (list, tuple)):
                return seg[0]
            return seg

    wrapped_network = SegOnlyWrapper(trainer.network)
    wrapped_network.eval()

    # Build PlansManager and ConfigurationManager
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration('3d_fullres')

    # Use nnUNet predictor with the already-loaded network
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,  # disable TTA for speed
        device=torch.device('cuda'),
        verbose=True,
        verbose_preprocessing=True,
        allow_tqdm=True,
    )

    # Initialize predictor with wrapped network
    predictor.manual_initialization(
        network=wrapped_network,
        plans_manager=plans_manager,
        configuration_manager=configuration_manager,
        parameters=None,
        dataset_json=ds_json,
        trainer_name='nnUNetTrainerTCMN',
        inference_allowed_mirroring_axes=trainer.inference_allowed_mirroring_axes,
    )

    # The predictor calls network.load_state_dict(params) for each fold.
    # Pass the wrapped network's state_dict so it reloads the same weights.
    predictor.list_of_parameters = [wrapped_network.state_dict()]

    # Set TCMN modality context for inference
    mod_id = torch.tensor([args.modality], device='cuda')
    print(f"Using modality context: {'CTA' if args.modality == 0 else 'MRA'} ({args.modality})")

    with tcmn.modality_context(mod_id):
        predictor.predict_from_files(
            list_of_lists_or_source_folder=args.input,
            output_folder_or_list_of_truncated_output_files=args.output,
            save_probabilities=False,
            overwrite=False,
            num_processes_preprocessing=4,
            num_processes_segmentation_export=4,
        )

    print(f"\n=== Prediction complete → {args.output} ===")


if __name__ == '__main__':
    main()
