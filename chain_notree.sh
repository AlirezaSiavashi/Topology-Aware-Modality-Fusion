#!/bin/bash
cd /scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset
# wait for the euclid zero-shot inference to finish, then evaluate it and
# start the _NoTree ablation on the freed GPU
until ! pgrep -f "nnUNetv2_predict" >/dev/null; do sleep 30; done
TAG=euclid ./eval_zeroshot_chain.sh > logs/zeroshot_euclid_eval.log 2>&1
TRAINER=nnUNetTrainerCMAP_NoTree GPU=0 FOLD=0 ./run_cmap.sh
