#!/bin/bash

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Experiment Timestamp: $TIMESTAMP"

ALGORITHMS=("Ours" "Local" "FedTED" "PACFL" "GeFL" "UDON-hom" "UDON-het" "GLFC")

for ALG in "${ALGORITHMS[@]}"
do
    echo ""
    echo "=========================================="
    echo ">>> Running $ALGO ..."
    echo "=========================================="

    python main.py \
        --algorithm $ALG \
        --exp_timestamp $TIMESTAMP \
        --exp_conf ./configs/het-exp.yaml \
        --device cuda:0 \
        --seed 15968

    echo ">>> $ALGO finished!"
done

echo ""
echo "=========================================="
echo "All experiments saved in: logs/$TIMESTAMP/"
echo "=========================================="