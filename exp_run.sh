#!/bin/bash

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Experiment Timestamp: $TIMESTAMP"

ALGORITHMS=("Ours" "Local" "FedTED")

for ALGO in "${ALGORITHMS[@]}"
do
    echo ""
    echo "=========================================="
    echo ">>> Running $ALGO ..."
    echo "=========================================="

    python main.py \
        --seed 15698 \
        --algorithm $ALGO \
        --exp_timestamp $TIMESTAMP \
        --exp_conf ./configs/het-exp.yaml \
        --num_train_cifar100 0 \
        --num_train_fashionmnist 0 \
        --num_train_usps 0

    echo ">>> $ALGO finished!"
done

echo ""
echo "=========================================="
echo "All experiments saved in: logs/$TIMESTAMP/"
echo "=========================================="
