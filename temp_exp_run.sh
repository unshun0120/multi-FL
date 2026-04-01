#!/bin/bash

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Experiment Timestamp: $TIMESTAMP"

TOTAL_START=$SECONDS

METHOD="syn"
CUDA="cuda:1"

# python main.py --seed=15698 --algorithm=Local --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=cuda:1

#python label_mapping/label_mapping.py --model_path=logs/Local_iid_het_gb40/Local/ --device=$CUDA --method=real
python label_mapping/label_mapping.py --model_path=logs/Local_iid_het_gb40/Local/ --device=$CUDA --method=syn --gen_mode=old
python label_mapping/label_mapping.py --model_path=logs/Local_iid_het_gb40/Local/ --device=$CUDA --method=syn --gen_mode=new

python label_mapping/label_mapping.py --model_path=logs/Local_iid_hom_gb40/Local/ --device=$CUDA --method=real
python label_mapping/label_mapping.py --model_path=logs/Local_iid_hom_gb40/Local/ --device=$CUDA --method=syn --gen_mode=old
python label_mapping/label_mapping.py --model_path=logs/Local_iid_hom_gb40/Local/ --device=$CUDA --method=syn --gen_mode=new

python label_mapping/label_mapping.py --model_path=logs/Local_noniid_het_gb40/Local/ --device=$CUDA --method=real
python label_mapping/label_mapping.py --model_path=logs/Local_noniid_het_gb40/Local/ --device=$CUDA --method=syn --gen_mode=old
python label_mapping/label_mapping.py --model_path=logs/Local_noniid_het_gb40/Local/ --device=$CUDA --method=syn --gen_mode=new

python label_mapping/label_mapping.py --model_path=logs/Local_noniid_hom_gb40/Local/ --device=$CUDA --method=real
python label_mapping/label_mapping.py --model_path=logs/Local_noniid_hom_gb40/Local/ --device=$CUDA --method=syn --gen_mode=old
python label_mapping/label_mapping.py --model_path=logs/Local_noniid_hom_gb40/Local/ --device=$CUDA --method=syn --gen_mode=new

TOTAL_TIME=$((SECONDS - TOTAL_START))
echo "Total execution time: $((TOTAL_TIME / 60)) minutes and $((TOTAL_TIME % 60)) seconds."