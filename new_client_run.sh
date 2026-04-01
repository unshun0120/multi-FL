#!/bin/bash

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
CKPT="logs/2026-02-23_12-37-52/Ours/checkpoint.pth"
DEVICE="cuda:1"
EPOCHS=1

echo "=========================================="
echo "New Client Experiment Timestamp: $TIMESTAMP"
echo "Target Checkpoint: $CKPT"
echo "=========================================="

COMMON_ARGS="--model_path=$CKPT --device=$DEVICE --new_client_epochs=$EPOCHS --exp_timestamp=$TIMESTAMP"

DATA_MODES=("single" "super")

EXPERIMENTS=(
    # "new_client_helpTrain.py" 
    # "new_client_helpTest.py" 
    "new_client_our_global_model.py"
    "new_client_train_scratch.py"
)

for MODE in "${DATA_MODES[@]}"; do
    echo ""
    echo "=========================================="
    echo ">>> Running [ $MODE ] Dataset Experiments"
    echo "=========================================="
    
    for EXP in "${EXPERIMENTS[@]}"; do
        echo "\n---> Running: $EXP (mode: $MODE) ..."
        python "$EXP" $COMMON_ARGS --dataset_mode="$MODE"
    done
done

echo ""
echo "=========================================="
echo "All new client logs saved in: $(dirname $CKPT)/new_client_$TIMESTAMP/"
echo "=========================================="