#!/bin/bash

cd ../src

MODEL_TYPE="rpnet"

MODEL_DATASET="negall"
#MODEL_DATASET="negnormal"
#MODEL_DATASET="neguniform"
#MODEL_DATASET="negshuffle"

LOSS="specentropy"
#LOSS="specflatness"
#LOSS="deviation"

NEGATIVE_PROB=0.5

## For vanilla training
#MODEL_TYPE="rpnet"
#LOSS="np"
#MODEL_DATASET="ddpm"
#NEGATIVE_PROB=0.0

DEBUG=0
NOISE_WIDTH=3.0

echo $MODEL_DATASET
echo $MODEL_TYPE
echo $LOSS
echo $DEBUG

declare -a DATASET_LIST=("DDPM" "NDDPM_SHUFFLE" "NDDPM_NORMAL" "NDDPM_UNIFORM" "CDDPM" "DFDC" "HKBU_MARs_V2" "KITTI" "PURE" "ARPM" "UBFC_rPPG")
 
# Iterate the string array using for loop
for TESTING_DATASET in ${DATASET_LIST[@]}; do
    python predict.py \
        --K $SGE_TASK_ID \
        --testing_dataset $TESTING_DATASET \
        --dataset $MODEL_DATASET \
        --model_type $MODEL_TYPE \
        --loss_type $LOSS \
        --negative_prob $NEGATIVE_PROB \
        --noise_width $NOISE_WIDTH \
        --debug $DEBUG
done
