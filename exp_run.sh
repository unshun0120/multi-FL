#!/bin/bash

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Experiment Timestamp: $TIMESTAMP"

TOTAL_START=$SECONDS
CUDA="cuda:1"

# python main.py --seed=15698 --algorithm=GeFL --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=$CUDA --exp_conf=./configs/het-iid-exp.yaml
# python main.py --seed=15698 --algorithm=GeFL --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=$CUDA --exp_conf=./configs/het-noniid-exp.yaml
# python main.py --seed=15698 --algorithm=GeFL --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=$CUDA --exp_conf=./configs/homo-iid-exp.yaml
# python main.py --seed=15698 --algorithm=GeFL --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=$CUDA --exp_conf=./configs/homo-noniid-exp.yaml

# python main.py --seed=15698 --algorithm=Ours --num_train_cifar100=0 --num_train_cifar10=0 --num_train_usps=0 --device=$CUDA --exp_conf=./configs/het-iid-exp.yaml
# python main.py --seed=15698 --algorithm=FedTED --num_train_cifar100=0 --num_train_cifar10=0 --num_train_usps=0 --device=$CUDA --exp_conf=./configs/het-iid-exp.yaml
# python main.py --seed=15698 --algorithm=UDON --num_train_cifar100=0 --num_train_cifar10=0 --num_train_usps=0 --device=$CUDA --exp_conf=./configs/het-iid-exp.yaml

# python main.py --seed=15698 --algorithm=Ours_GeFL --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=$CUDA --exp_conf=./configs/het-iid-exp.yaml

# python main.py --seed=15698 --algorithm=Ours --num_train_cifar100=0 --num_train_cifar10=0 --num_train_usps=0 --device=$CUDA --exp_conf=./configs/het-noniid-exp.yaml
# python main.py --seed=15698 --algorithm=FedTED --num_train_cifar100=0 --num_train_cifar10=0 --num_train_usps=0 --device=$CUDA --exp_conf=./configs/het-noniid-exp.yaml
# python main.py --seed=15698 --algorithm=UDON --num_train_cifar100=0 --num_train_cifar10=0 --num_train_usps=0 --device=$CUDA --exp_conf=./configs/het-noniid-exp.yaml

# python main.py --seed=15698 --algorithm=Ours --num_train_cifar100=0 --num_train_cifar10=0 --num_train_usps=0 --device=$CUDA --exp_conf=./configs/homo-iid-exp.yaml
# python main.py --seed=15698 --algorithm=FedTED --num_train_cifar100=0 --num_train_cifar10=0 --num_train_usps=0 --device=$CUDA --exp_conf=./configs/homo-iid-exp.yaml
# python main.py --seed=15698 --algorithm=UDON --num_train_cifar100=0 --num_train_cifar10=0 --num_train_usps=0 --device=$CUDA --exp_conf=./configs/homo-iid-exp.yaml

# python main.py --seed=15698 --algorithm=Ours --num_train_cifar100=0 --num_train_cifar10=0 --num_train_usps=0 --device=$CUDA --exp_conf=./configs/homo-noniid-exp.yaml
# python main.py --seed=15698 --algorithm=FedTED --num_train_cifar100=0 --num_train_cifar10=0 --num_train_usps=0 --device=$CUDA --exp_conf=./configs/homo-noniid-exp.yaml
# python main.py --seed=15698 --algorithm=UDON --num_train_cifar100=0 --num_train_cifar10=0 --num_train_usps=0 --device=$CUDA --exp_conf=./configs/homo-noniid-exp.yaml

# python main.py --seed=15698 --algorithm=GeFL_local --num_train_mnist=5 --num_train_emnist=5 --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_cifar10=0 --num_train_usps=0 --device=cuda:1 --exp_conf=./configs/het-iid-exp.yaml
# python main.py --seed=15698 --algorithm=GeFL_local --num_train_mnist=5 --num_train_emnist=5 --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_cifar10=0 --num_train_usps=0 --device=cuda:1 --exp_conf=./configs/het-noniid-exp.yaml

# python main.py --seed=15698 --algorithm=Ours_GeFL --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=cuda:1 --exp_conf=./configs/het-iid-exp.yaml

# python main.py --seed=15698 --algorithm=GeFL_DDPM --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=cuda:1 --exp_conf=./configs/het-iid-exp.yaml

# mapping : real_img, class_name, gan
# python main.py --seed=15698 --algorithm=UDON --num_train_cifar100=0 --num_train_cifar10=0 --num_train_usps=0 --device=cuda:1 --exp_conf=./configs/het-iid-exp.yaml
# python main.py --seed=15698 --algorithm=FedTED --num_train_cifar100=0 --num_train_cifar10=0 --num_train_usps=0 --device=cuda:1 --exp_conf=./configs/het-iid-exp.yaml
# python main.py --seed=15698 --algorithm=BaseFL_public --num_train_cifar100=0 --num_train_cifar10=0 --num_train_usps=0 --device=cuda:1 --exp_conf=./configs/het-iid-exp.yaml
# python main.py --seed=15698 --algorithm=Ours_GeFL --num_train_cifar100=0 --num_train_cifar10=0 --num_train_usps=0 --device=cuda:1 --exp_conf=./configs/het-iid-exp.yaml
# python main.py --seed=15698 --algorithm=FedTED_dir --num_train_cifar100=0 --num_train_cifar10=0 --num_train_usps=0 --device=cuda:1 --exp_conf=./configs/het-iid-exp.yaml

# python main.py --seed=15698 --algorithm=UDON --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=cuda:1 --exp_conf=./configs/het-iid-exp.yaml
# python main.py --seed=15698 --algorithm=FedTED --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=cuda:1 --exp_conf=./configs/het-iid-exp.yaml
# python main.py --seed=15698 --algorithm=Ours_GeFL --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=cuda:1 --exp_conf=./configs/het-iid-exp.yaml
# python main.py --seed=15698 --algorithm=GeFL_DDPM --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=cuda:1 --exp_conf=./configs/het-iid-exp.yaml
# python main.py --seed=15698 --algorithm=Ours --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=cuda:1 --exp_conf=./configs/het-iid-exp.yaml

# python main.py --seed=15698 --algorithm=GeFL_DDPM_baseline_total --label_mapping=image-bi --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=cuda:1 --exp_conf=./configs/het-iid-exp.yaml
# python main.py --seed=15698 --algorithm=GeFL_DDPM_baseline_total --label_mapping=image-single --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=cuda:1 --exp_conf=./configs/het-iid-exp.yaml
# python main.py --seed=15698 --algorithm=GeFL_DDPM_baseline_total --label_mapping=feature-bi --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=cuda:1 --exp_conf=./configs/het-iid-exp.yaml
# python main.py --seed=15698 --algorithm=GeFL_DDPM_baseline_total --label_mapping=image-cs --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=cuda:1 --exp_conf=./configs/het-iid-exp.yaml

# python main.py --seed=15698 --algorithm=GeFL_DDPM_baseline_total --label_mapping=independent --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=cuda:0 --exp_conf=./configs/het-iid-exp.yaml
# python main.py --seed=15698 --algorithm=GeFL_DDPM_baseline_total --label_mapping=identical --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=cuda:0 --exp_conf=./configs/het-iid-exp.yaml
# python main.py --seed=15698 --algorithm=GeFL_DDPM_baseline_total --label_mapping=class_name --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=cuda:0 --exp_conf=./configs/het-iid-exp.yaml

# python main.py --seed=15698 --algorithm=GeFL_DDPM_baseline_total_public --label_mapping=image-bi --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=cuda:1 --exp_conf=./configs/het-iid-exp.yaml
# python main.py --seed=15698 --algorithm=GeFL_DDPM_baseline_total_public --label_mapping=image-single --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=cuda:1 --exp_conf=./configs/het-iid-exp.yaml
# python main.py --seed=15698 --algorithm=GeFL_DDPM_baseline_total_public --label_mapping=feature-bi --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=cuda:1 --exp_conf=./configs/het-iid-exp.yaml
# python main.py --seed=15698 --algorithm=GeFL_DDPM_baseline_total_public --label_mapping=image-cs --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=cuda:1 --exp_conf=./configs/het-iid-exp.yaml


python main.py --seed=15698 --algorithm=GeFL_DDIM_baseline_total --label_mapping=image-bi --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=cuda:1 --exp_conf=./configs/het-iid-exp.yaml
# python main.py --seed=15698 --algorithm=GeFL_DDPM_baseline_total_gan --label_mapping=image-single --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=cuda:1 --exp_conf=./configs/het-iid-exp.yaml
python main.py --seed=15698 --algorithm=GeFL_DDIM_baseline_total --label_mapping=feature-bi --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=cuda:1 --exp_conf=./configs/het-iid-exp.yaml
# python main.py --seed=15698 --algorithm=GeFL_DDPM_baseline_total_gan --label_mapping=image-cs --num_train_cifar100=0 --num_train_fashionmnist=0 --num_train_usps=0 --device=cuda:1 --exp_conf=./configs/het-iid-exp.yaml


TOTAL_TIME=$((SECONDS - TOTAL_START))
echo "Total execution time: $((TOTAL_TIME / 3600)) hours, $((TOTAL_TIME / 60)) minutes and $((TOTAL_TIME % 60)) seconds."