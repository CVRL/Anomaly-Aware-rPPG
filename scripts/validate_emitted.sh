#!/bin/bash

MODEL_PATH=${1}
MODEL=${2}
SPLIT=${3}
VAL_DB=${4}
LOSS=${5}
NEG_PROB=${7}
NOISE_WIDTH=${8}
FPC=${9}
STEP=${10}
SKIP=${11}

python validate.py \
    --fpc $FPC \
    --step $STEP \
    --skip $SKIP \
    --load_path $MODEL_PATH \
    --model_type $MODEL \
    --split $SPLIT \
    --val_db $VAL_DB \
    --loss $LOSS \
    --negative_prob $NEG_PROB \
    --noise_width $NOISE_WIDTH
