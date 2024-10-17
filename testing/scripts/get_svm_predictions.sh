#!/bin/bash

cd ../src

DEBUG=0

declare -a DATASET_LIST=("DDPM" "NDDPM_SHUFFLE" "NDDPM_NORMAL" "NDDPM_UNIFORM" "CDDPM" "DFDC" "HKBU_MARs_V2" "KITTI" "PURE" "ARPM" "UBFC_rPPG")
 
for K in {1..5}; do
    for TESTING_DATASET in ${DATASET_LIST[@]}; do
        python get_svm_predictions.py --K $K \
            --testing_dataset $TESTING_DATASET \
            --debug $DEBUG \
            --task_id 0 \
            --task_count 1
    done
done
