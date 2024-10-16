#!/bin/bash

MODEL="rpnet"

LOSS="specentropy"
#LOSS="specflatness"
#LOSS="deviation"

DATASET="neguniform"
#DATASET="negnormal"
#DATASET="negshuffle"
#DATASET="negall"

NFFT=5400
NEG_PROB=0.5
NOISE_WIDTH=3

BS=8
FPC=270
STEP=135
SKIP=1
CHANNELS="rgb"
AUGMENTATION="fig"
DTYPE="f"

DEBUG=0
CONTINUE_TRAINING=0
NUM_WORKERS=4
NUM_EPOCHS=40

cd ../src
python train.py --debug $DEBUG \
                --nfft $NFFT \
                --negative_prob $NEG_PROB \
                --noise_width $NOISE_WIDTH \
                --dataset $DATASET \
                --model_type $MODEL \
                --augmentation $AUGMENTATION \
                --channels $CHANNELS \
                --loss $LOSS \
                --batch_size $BS \
                --dtype $DTYPE \
                --fpc $FPC \
                --step $STEP \
                --skip $SKIP \
                --num_workers $NUM_WORKERS \
                --end_epoch $NUM_EPOCHS \
                --K $SGE_TASK_ID
