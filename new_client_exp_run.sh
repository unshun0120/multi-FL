#!/bin/bash

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Experiment Timestamp: $TIMESTAMP"

TOTAL_START=$SECONDS
CUDA="cuda:1"

# python new_client_our_global_model.py --model_path=logs/2026-05-11_18-56-10/Ours_GeFL/server_checkpoints.pth --device=$CUDA --dataset_mode=single
# python new_client_our_global_model.py --model_path=logs/2026-05-12_02-19-31/FedTED/config_checkpoints.pth --device=$CUDA --dataset_mode=single
# python new_client_our_global_model.py --model_path=logs/2026-05-12_12-45-34/FedTED_dir/config_checkpoints.pth --device=$CUDA --dataset_mode=single
# python new_client_our_global_model.py --model_path=logs/2026-05-12_00-51-46/UDON/config_checkpoints.pth --device=$CUDA --dataset_mode=single

# python new_client_our_global_model.py --model_path=logs/2026-05-11_18-56-10/Ours_GeFL/server_checkpoints.pth --device=$CUDA --dataset_mode=super
# python new_client_our_global_model.py --model_path=logs/2026-05-12_02-19-31/FedTED/config_checkpoints.pth --device=$CUDA --dataset_mode=super
# python new_client_our_global_model.py --model_path=logs/2026-05-12_12-45-34/FedTED_dir/config_checkpoints.pth --device=$CUDA --dataset_mode=super
# python new_client_our_global_model.py --model_path=logs/2026-05-12_00-51-46/UDON/config_checkpoints.pth --device=$CUDA --dataset_mode=super

python new_client_train_scratch.py --model_path=logs/2026-05-11_18-56-10/Ours_GeFL/server_checkpoints.pth --device=$CUDA --dataset_mode=single
# python new_client_train_scratch.py --model_path=logs/2026-05-11_18-56-10/Ours_GeFL/server_checkpoints.pth --device=$CUDA --dataset_mode=super


python new_client_our_global_model.py --model_path=logs/2026-05-12_10-39-36/Ours_GeFL/server_checkpoints.pth --device=$CUDA --dataset_mode=single
python new_client_our_global_model.py --model_path=logs/2026-05-12_07-04-29/FedTED/config_checkpoints.pth --device=$CUDA --dataset_mode=single
python new_client_our_global_model.py --model_path=logs/2026-05-12_17-34-01/FedTED_dir/config_checkpoints.pth --device=$CUDA --dataset_mode=single
python new_client_our_global_model.py --model_path=logs/2026-05-12_05-37-28/UDON/config_checkpoints.pth --device=$CUDA --dataset_mode=single

# python new_client_our_global_model.py --model_path=logs/2026-05-12_10-39-36/Ours_GeFL/server_checkpoints.pth --device=$CUDA --dataset_mode=super
# python new_client_our_global_model.py --model_path=logs/2026-05-12_07-04-29/FedTED/config_checkpoints.pth --device=$CUDA --dataset_mode=super
# python new_client_our_global_model.py --model_path=logs/2026-05-12_17-34-01/FedTED_dir/config_checkpoints.pth --device=$CUDA --dataset_mode=super
# python new_client_our_global_model.py --model_path=logs/2026-05-12_05-37-28/UDON/config_checkpoints.pth --device=$CUDA --dataset_mode=super

python new_client_train_scratch.py --model_path=logs/2026-05-12_10-39-36/Ours_GeFL/server_checkpoints.pth --device=$CUDA --dataset_mode=single
# python new_client_train_scratch.py --model_path=logs/2026-05-12_10-39-36/Ours_GeFL/server_checkpoints.pth --device=$CUDA --dataset_mode=super


# python new_client_our_global_model.py --model_path=logs/2026-05-01_16-40-45/Ours_GeFL_classname/server_checkpoints.pth --device=$CUDA --dataset_mode=super
# python new_client_our_global_model.py --model_path=logs/2026-05-01_18-03-18/Ours_GeFL_realimg/server_checkpoints.pth --device=$CUDA --dataset_mode=super
# python new_client_our_global_model.py --model_path=logs/2026-05-02_15-52-18/Ours_GeFL_gan/server_checkpoints.pth --device=$CUDA --dataset_mode=super

TOTAL_TIME=$((SECONDS - TOTAL_START))
echo "Total execution time: $((TOTAL_TIME / 3600)) hours, $((TOTAL_TIME / 60)) minutes and $((TOTAL_TIME % 60)) seconds."