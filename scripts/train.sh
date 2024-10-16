#!/bin/bash

DATASET="ddpm"
#DATASET="neguniform"
#DATASET="negnormal"
#DATASET="negshuffle"
#DATASET="negall"

LOSS="np"
#LOSS="specentropy"
#LOSS="specflatness"
#LOSS="deviation"
#LOSS="deviationmargin"

MODEL="rpnet"
AUGMENTATION="fig" # flipping, illumination, gaussian
CHANNELS="rgb"      # desired channel order for input video (expects bgr input)
FPC=270             # frames-per-clip
STEP=135            # step between clips fed during training
SKIP=1              # frame rate from skipping (e.g. skip=2 would give (90/2)=45 fps)
NUM_WORKERS=4
NUM_EPOCHS=40

cd ../src
python train.py --dataset $DATASET \
                --model_type $MODEL \
                --augmentation $AUGMENTATION \
                --channels $CHANNELS \
                --loss $LOSS \
                --fpc $FPC \
                --step $STEP \
                --skip $SKIP \
                --num_workers $NUM_WORKERS \
                --end_epoch $NUM_EPOCHS
