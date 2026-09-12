"""
Debug CMTC training — catch segfault with faulthandler to get traceback.
"""
import os
import sys
import faulthandler

# Enable faulthandler to print traceback on SIGSEGV
faulthandler.enable(all_threads=True)

# Also dump traceback on SIGUSR1 for manual debugging
import signal
faulthandler.register(signal.SIGUSR1, all_threads=True)

from nnunetv2.run.run_training import run_training

if __name__ == '__main__':
    run_training(
        dataset_name_or_id='104',
        configuration='3d_fullres',
        fold=0,
        trainer_class_name='nnUNetTrainerCMTC',
        continue_training=True,
    )
