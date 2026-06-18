#!/bin/bash

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Experiment Timestamp: $TIMESTAMP"

TOTAL_START=$SECONDS

METHOD="syn"
CUDA="cuda:1"

# python main.py --seed=15698 --algorithm=Local --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=cuda:1 --exp_conf=./configs/het-iid-exp.yaml
# python main.py --seed=15698 --algorithm=Local --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=cuda:1 --exp_conf=./configs/het-noniid-exp.yaml
# python main.py --seed=15698 --algorithm=Local --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=cuda:1 --exp_conf=./configs/homo-iid-exp.yaml
# python main.py --seed=15698 --algorithm=Local --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=cuda:1 --exp_conf=./configs/homo-noniid-exp.yaml

# python label_mapping/label_mapping.py --model_path=logs/Local_iid_het_gb40/Local/ --device=$CUDA --method=real
# python label_mapping/label_mapping.py --model_path=logs/Local_iid_het_gb40/Local/ --device=$CUDA --method=DI --gen_mode=old
# python label_mapping/label_mapping.py --model_path=logs/Local_iid_het_gb40/Local/ --device=$CUDA --method=DI --gen_mode=new
# python label_mapping/label_mapping.py --model_path=logs/Local_iid_het_gb40/Local/ --device=$CUDA --method=FAST --gen_mode=old
# python label_mapping/label_mapping.py --model_path=logs/Local_iid_het_gb40/Local/ --device=$CUDA --method=FAST --gen_mode=new
# python label_mapping/label_mapping.py --model_path=logs/Local_iid_het_gb40/Local/ --device=$CUDA --method=NAYER --gen_mode=old
# python label_mapping/label_mapping.py --model_path=logs/Local_iid_het_gb40/Local/ --device=$CUDA --method=NAYER --gen_mode=new
# python label_mapping/label_mapping.py --model_path=logs/Local_iid_het_gb40/Local/ --device=$CUDA --method=Fed --gen_mode=old
# python label_mapping/label_mapping.py --model_path=logs/Local_iid_het_gb40/Local/ --device=$CUDA --method=Fed --gen_mode=new

# python label_mapping/label_mapping.py --model_path=logs/Local_noniid_het_gb40/Local/ --device=$CUDA --method=real
# python label_mapping/label_mapping.py --model_path=logs/Local_noniid_het_gb40/Local/ --device=$CUDA --method=DI --gen_mode=old
# python label_mapping/label_mapping.py --model_path=logs/Local_noniid_het_gb40/Local/ --device=$CUDA --method=FAST --gen_mode=old
# python label_mapping/label_mapping.py --model_path=logs/Local_noniid_het_gb40/Local/ --device=$CUDA --method=NAYER --gen_mode=old

# python label_mapping/label_mapping.py --model_path=logs/Local_noniid_het_gb40/Local/ --device=$CUDA --method=DI --gen_mode=new
# python label_mapping/label_mapping.py --model_path=logs/Local_noniid_het_gb40/Local/ --device=$CUDA --method=FAST --gen_mode=new
# python label_mapping/label_mapping.py --model_path=logs/Local_noniid_het_gb40/Local/ --device=$CUDA --method=NAYER --gen_mode=new

# python label_mapping/label_mapping.py --model_path=logs/Local_iid_hom_gb40/Local/ --device=$CUDA --method=real
# python label_mapping/label_mapping.py --model_path=logs/Local_iid_hom_gb40/Local/ --device=$CUDA --method=DI --gen_mode=old
# python label_mapping/label_mapping.py --model_path=logs/Local_iid_hom_gb40/Local/ --device=$CUDA --method=DI --gen_mode=new
# python label_mapping/label_mapping.py --model_path=logs/Local_iid_hom_gb40/Local/ --device=$CUDA --method=FAST --gen_mode=old
# python label_mapping/label_mapping.py --model_path=logs/Local_iid_hom_gb40/Local/ --device=$CUDA --method=FAST --gen_mode=new
# python label_mapping/label_mapping.py --model_path=logs/Local_iid_hom_gb40/Local/ --device=$CUDA --method=NAYER --gen_mode=old
# python label_mapping/label_mapping.py --model_path=logs/Local_iid_hom_gb40/Local/ --device=$CUDA --method=NAYER --gen_mode=new

# python label_mapping/label_mapping.py --model_path=logs/Local_noniid_hom_gb40/Local/ --device=$CUDA --method=real
# python label_mapping/label_mapping.py --model_path=logs/Local_noniid_hom_gb40/Local/ --device=$CUDA --method=DI --gen_mode=old
# python label_mapping/label_mapping.py --model_path=logs/Local_noniid_hom_gb40/Local/ --device=$CUDA --method=FAST --gen_mode=old
# python label_mapping/label_mapping.py --model_path=logs/Local_noniid_hom_gb40/Local/ --device=$CUDA --method=NAYER --gen_mode=old

# python label_mapping/label_mapping.py --model_path=logs/Local_noniid_hom_gb40/Local/ --device=$CUDA --method=DI --gen_mode=new
# python label_mapping/label_mapping.py --model_path=logs/Local_noniid_hom_gb40/Local/ --device=$CUDA --method=FAST --gen_mode=new
# python label_mapping/label_mapping.py --model_path=logs/Local_noniid_hom_gb40/Local/ --device=$CUDA --method=NAYER --gen_mode=new

# python label_mapping/label_mapping.py --model_path=logs/2026-05-11_16-26-00/GeFL_local/ --method=real_seperate
# python label_mapping/label_mapping.py --model_path=logs/2026-05-11_16-26-00/GeFL_local/ --method=GeFL_local --gan_path=logs/2026-05-11_16-26-00/GeFL_local/

# python label_mapping/label_mapping.py --model_path=logs/2026-04-16_13-34-59/GeFL/ --method=GeFL --gan_path=logs/2026-04-16_13-34-59/GeFL/

# python label_mapping/label_mapping.py --model_path=logs/2026-05-01_00-34-57/GeFL_DDPM/ --method=GeFL_DDPM --gan_path=logs/2026-05-01_00-34-57/GeFL_DDPM/
# python label_mapping/label_mapping.py --model_path=logs/2026-05-01_00-34-57/GeFL_DDPM/ --method=real

# python label_mapping/label_mapping.py --model_path=logs/2026-05-12_17-40-17/Local/ --device=$CUDA --method=DI --gen_mode=old
# python label_mapping/label_mapping.py --model_path=logs/2026-05-12_17-40-17/Local/ --device=$CUDA --method=FAST --gen_mode=old
# python label_mapping/label_mapping.py --model_path=logs/2026-05-12_17-40-17/Local/ --device=$CUDA --method=NAYER --gen_mode=old
# python label_mapping/label_mapping.py --model_path=logs/2026-05-12_17-40-17/Local/ --device=$CUDA --method=Fed --gen_mode=old

# python main.py --seed=15698 --algorithm=Ours --num_train_cifar100=0 --num_train_cifar10=0 --num_train_usps=0 --device=cuda:1 --exp_conf=./configs/het-iid-exp.yaml
# python main.py --seed=15698 --algorithm=Ours --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=cuda:1 --exp_conf=./configs/het-iid-exp.yaml

python main.py --seed=15698 --algorithm=GeFL_DDPM_baseline_total --label_mapping=image-bi --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=cuda:0 --exp_conf=./configs/het-iid-exp.yaml
python main.py --seed=15698 --algorithm=GeFL_DDPM_baseline_total --label_mapping=feature-bi --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=cuda:0 --exp_conf=./configs/het-iid-exp.yaml

# python main.py --seed=15698 --algorithm=GeFL_DDPM_baseline_total_public --label_mapping=image-bi --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=cuda:0 --exp_conf=./configs/het-iid-exp.yaml
# python main.py --seed=15698 --algorithm=GeFL_DDPM_baseline_total_public --label_mapping=image-single --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=cuda:0 --exp_conf=./configs/het-iid-exp.yaml
# python main.py --seed=15698 --algorithm=GeFL_DDPM_baseline_total_public --label_mapping=feature-bi --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=cuda:0 --exp_conf=./configs/het-iid-exp.yaml
# python main.py --seed=15698 --algorithm=GeFL_DDPM_baseline_total_public --label_mapping=image-cs --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=cuda:0 --exp_conf=./configs/het-iid-exp.yaml


TOTAL_TIME=$((SECONDS - TOTAL_START))
echo "Total execution time: $((TOTAL_TIME / 3600)) hours, $((TOTAL_TIME / 60)) minutes and $((TOTAL_TIME % 60)) seconds."