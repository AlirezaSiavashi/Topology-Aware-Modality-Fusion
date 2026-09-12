"""
predict_custom.py — Prediction script for custom trainers (TCMN, CMTC, TopologyVessel)
======================================================================================

Bypasses nnUNetv2_predict's strict load_state_dict which fails for trainers
that use module aliasing (e.g. TCMN's all_modules duplication).

Usage:
    python predict_custom.py \
        -i INPUT_FOLDER \
        -o OUTPUT_FOLDER \
        -d DATASET_ID \
        -c CONFIGURATION \
        -tr TRAINER_NAME \
        -f FOLD \
        [--checkpoint CHECKPOINT_NAME] \
        [--disable_tta]
"""

import argparse
import os
import sys

# Make our training/ directory importable
_TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
if _TRAINING_DIR not in sys.path:
    sys.path.insert(0, _TRAINING_DIR)

import torch
import numpy as np
from pathlib import Path

import nnunetv2
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.paths import nnUNet_results
from batchgenerators.utilities.file_and_folder_operations import load_json, join


class CustomPredictor(nnUNetPredictor):
    """
    nnUNetPredictor subclass that uses strict=False for load_state_dict.
    Handles module aliasing from TCMN injection (80 alias keys that appear
    as 'missing' with strict=True but are actually harmless duplicates).
    """

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                              use_folds, checkpoint_name='checkpoint_final.pth'):
        """Override to use strict=False when loading network weights."""
        if use_folds is None:
            use_folds = nnUNetPredictor.auto_detect_available_folds(
                model_training_output_dir, checkpoint_name)

        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != 'all' else f
            checkpoint = torch.load(
                join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                map_location=torch.device('cpu'), weights_only=False)
            if i == 0:
                trainer_name = checkpoint['trainer_name']
                configuration_name = checkpoint['init_args']['configuration']
                inference_allowed_mirroring_axes = checkpoint.get(
                    'inference_allowed_mirroring_axes', None)
            parameters.append(checkpoint['network_weights'])

        configuration_manager = plans_manager.get_configuration(configuration_name)

        from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
        num_input_channels = determine_num_input_channels(
            plans_manager, configuration_manager, dataset_json)

        # Find trainer class — search our training dir first, then nnunetv2
        trainer_class = recursive_find_python_class(
            join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
            trainer_name, 'nnunetv2.training.nnUNetTrainer')
        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name}')

        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        )

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)

        # KEY FIX: use strict=False for the initial load
        missing, unexpected = network.load_state_dict(parameters[0], strict=False)
        if missing:
            print(f"[predict_custom] {len(missing)} missing keys (expected for TCMN alias)")
        if unexpected:
            print(f"[predict_custom] {len(unexpected)} unexpected keys (ignored)")

    def predict_logits_from_preprocessed_data(self, data):
        """Override to use strict=False when switching between folds."""
        from torch._dynamo import OptimizedModule
        prediction = None
        for params in self.list_of_parameters:
            # strict=False for TCMN alias compatibility
            if not isinstance(self.network, OptimizedModule):
                self.network.load_state_dict(params, strict=False)
            else:
                self.network._orig_mod.load_state_dict(params, strict=False)

            if prediction is None:
                prediction = self.predict_sliding_window_return_logits(data).to('cpu')
            else:
                prediction += self.predict_sliding_window_return_logits(data).to('cpu')

        if len(self.list_of_parameters) > 1:
            prediction /= len(self.list_of_parameters)

        return prediction


def main():
    parser = argparse.ArgumentParser(description='Custom nnU-Net prediction with strict=False loading')
    parser.add_argument('-i', '--input_folder', required=True)
    parser.add_argument('-o', '--output_folder', required=True)
    parser.add_argument('-d', '--dataset_name_or_id', required=True)
    parser.add_argument('-c', '--configuration', default='3d_fullres')
    parser.add_argument('-tr', '--trainer_class_name', default='nnUNetTrainer')
    parser.add_argument('-p', '--plans_identifier', default='nnUNetPlans')
    parser.add_argument('-f', '--folds', nargs='+', default=[0])
    parser.add_argument('--checkpoint', default='checkpoint_final.pth')
    parser.add_argument('--disable_tta', action='store_true')
    parser.add_argument('--continue_prediction', action='store_true')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    # Resolve model folder
    dataset_name = maybe_convert_to_dataset_name(args.dataset_name_or_id)
    model_folder = join(
        nnUNet_results, dataset_name,
        f'{args.trainer_class_name}__{args.plans_identifier}__{args.configuration}')

    print(f"Model folder: {model_folder}")
    print(f"Input:  {args.input_folder}")
    print(f"Output: {args.output_folder}")
    print(f"Folds:  {args.folds}")
    print(f"Checkpoint: {args.checkpoint}")

    os.makedirs(args.output_folder, exist_ok=True)

    predictor = CustomPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=not args.disable_tta,
        perform_everything_on_device=True,
        device=torch.device(args.device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=args.folds,
        checkpoint_name=args.checkpoint,
    )

    predictor.predict_from_files(
        list_of_lists_or_source_folder=args.input_folder,
        output_folder_or_list_of_truncated_output_files=args.output_folder,
        save_probabilities=False,
        overwrite=not args.continue_prediction,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )

    print("Prediction complete!")


if __name__ == '__main__':
    main()
