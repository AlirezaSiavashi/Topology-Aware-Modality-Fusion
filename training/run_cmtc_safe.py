"""
Safe CMTC training wrapper that avoids the segfault on SLURM restart.

The segfault occurs because:
1. checkpoint_latest.pth is loaded with CUDA tensors
2. batchgeneratorsv2 DA workers fork after CUDA context is initialized
3. fft_conv_pytorch in the forked workers triggers a segfault

Fix: set CUDA_LAUNCH_BLOCKING=1 and use torch multiprocessing start method 'spawn'
before any CUDA operations.
"""
import os
import sys
import torch.multiprocessing as mp

# Force spawn for all multiprocessing to avoid fork-after-CUDA segfault
mp.set_start_method('spawn', force=True)

# Now run nnU-Net training normally
from nnunetv2.run.run_training import run_training

if __name__ == '__main__':
    run_training(
        dataset_name_or_id='104',
        configuration='3d_fullres',
        fold=0,
        trainer_class_name='nnUNetTrainerCMTC',
        continue_training=True,
    )
