#!/bin/bash

cd ../src

DEBUG=0

for K in {1..5}; do
    python fit_svms.py --K $K \
    --debug $DEBUG \
    --task_id 0 \
    --task_count 1
done
