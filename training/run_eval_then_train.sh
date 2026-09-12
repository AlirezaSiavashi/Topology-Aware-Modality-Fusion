#!/bin/bash
# Wait for DS104 baseline eval to finish, then start TCMN_NoTopo training on GPU 1

EVAL_PID=$1
BASE=/scratch/siyavash/Alireza_thesis/external_dataset/rsna-intracranial-aneurysm-detection/brain_external_dataset

echo "Waiting for eval process PID $EVAL_PID to finish..."
while kill -0 $EVAL_PID 2>/dev/null; do
    sleep 60
done
echo "Eval process finished at $(date)"

echo "Starting TCMN_NoTopo training on GPU 1..."
exec ${BASE}/training/train_tcmn_notopo_104.sh
