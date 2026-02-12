import argparse
import os
import shutil
from importlib import import_module
import sys
import torch
import numpy as np
import random
from omegaconf import OmegaConf
import copy
import time 
from datetime import timedelta

from data.datasets import load_partitioned_datasets
from utils.logger import Logger
from utils.nets import get_heterogeneous_model, TwinBranchNets
from utils.train_utils import (
    initialize_training_clients
)

  
DATA_ROOT = './data/raw'

IMPLEMENTED_ALGORITHMS = ["Local", # FL with local only
                          "FedTED", # FL with data free knowledge distillation 
                          "PACFL", # FL with mix-dataset
                          "GeFL", # FL with generative model 
                          "UDON-hom", "UDON-het", # Only KD not FL
                          "GLFC", # FL with incremental learning
                          "", # FL with foudation model
                          "Ours" # Ours
                          ]

MODEL_HET_ALGORITHMS = ["Local", "FedTED", "GeFL", "UDON-het", "Ours"]
NEED_PUBLIC_DATASET_ALGORITHMS = ["FedTED", "UDON-hom", "UDON-het"]

DATASET_META = { 
    'MNIST':        {'in_ch': 3, 'classes': 10,  'size': 32},  
    'FashionMNIST': {'in_ch': 3, 'classes': 10,  'size': 32},
    'EMNIST':       {'in_ch': 3, 'classes': 47,  'size': 32},
    'CIFAR10':      {'in_ch': 3, 'classes': 10,  'size': 32},
    'CIFAR100':     {'in_ch': 3, 'classes': 100, 'size': 32}
}

MODEL_LIST = {
    0: 'MLP', 1: 'CNN', 2: 'ResNet8', 3: 'ResNet18',
    4: 'MobileNetV2', 5: 'MobileNetV3', 6: 'LeNet',
    7: 'AlexNet', 8: 'ShuffleNet', 9: 'SqueezeNet'
}


def parser_args():
    parser = argparse.ArgumentParser()

    # --- System setup ---
    parser.add_argument('--exp_timestamp', type=str, default=None, 
                        help='Shared timestamp for batch experiments (from .sh script)')
    parser.add_argument('--no_write_log', action='store_true', 
                        help='logging and plotting to files')
    parser.add_argument("--exp_conf", type=str, default="./configs/het-exp.yaml",
                        help="experiment config yaml files")
    parser.add_argument("--device", type=str, default="cuda:0", 
                        help="run device (cpu | cuda:x, x:int > 0)")
    parser.add_argument("--seed", type=int, default=None, 
                        help="random seed")
    parser.add_argument('--no_save_model', action='store_true', 
                        help='Whether to save the trained model checkpoints')
    parser.add_argument("--algorithm", type=str, default="Ours", choices=IMPLEMENTED_ALGORITHMS,
                        help=f"the implemented algorithms, choices include: {IMPLEMENTED_ALGORITHMS}")
    
    # --- Dataset ---
    parser.add_argument('--num_new_clients', type=int, default=1, 
                        help='Number of IID New clients for generalization test (for all dataset)')
    parser.add_argument("--num_train_mnist", type=int, default=10, 
                        help="number of training clients")
    parser.add_argument("--num_train_emnist", type=int, default=10)
    parser.add_argument("--num_train_fashionmnist", type=int, default=10)
    parser.add_argument("--num_train_cifar10", type=int, default=10)
    parser.add_argument("--num_train_cifar100", type=int, default=10)

    return parser.parse_args()


def exp_run(args, logger, **exp_conf):
    """ Run experiments with args and conf.yaml
    :return: log_dir
    """
    # 1. set random seed
    set_seed(args.seed)

    # 2. prepare dataset and public dataset
    all_client_data_loaders, global_id_map, global_registry, super_train_ds, super_test_ds = load_partitioned_datasets(args, DATA_ROOT, **exp_conf)

    logger.log(global_id_map)
    logger.log('='*30)
    logger.log(global_registry)

    # 5. create clients
    Client = getattr(import_module("trainer.%s.client" % args.algorithm), 'Client')
    train_clients = initialize_training_clients(
        Client, all_client_data_loaders, DATASET_META, MODEL_LIST, args, DATA_ROOT, global_registry, logger, **exp_conf
    )

    # 6. create server
    Server = getattr(import_module("trainer.%s.server" % args.algorithm), 'Server')

    model_arch_id = next(k for k, v in MODEL_LIST.items() if v == exp_conf.get('global_model_arch', 'ResNet8'))
    global_in_channels = 3 
    global_img_size = 32
    global_num_classes = len(global_registry)

    global_full_model = get_heterogeneous_model(
        node_id=model_arch_id,
        in_channels=global_in_channels,
        num_classes=global_img_size,
        img_size=global_num_classes,
        global_dim=exp_conf.get('global_feature_dim', 128)
    )

    server = Server(
        node_id=0,
        args=args, 
        clients=train_clients,
        dataset_name = None, 
        train_loader=super_train_ds, 
        test_loader=super_test_ds,   
        model=global_full_model,
        class_name_set=None,
        model_name=MODEL_LIST[model_arch_id],
        global_registry=global_registry, 
        device=args.device,
        logger=logger,
        **exp_conf,
    )

    # 7. start training
    server.run()


def set_seed(seed):
    if seed is not None:
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)                
        torch.backends.cudnn.deterministic = True        
        torch.backends.cudnn.benchmark = False           


def main():
    total_start_time = time.time()

    # parser args, read config yaml file
    args = parser_args()
    exp_conf = OmegaConf.load(args.exp_conf)
    logger = Logger(args, mode=args.algorithm)

    logger.log('\n')
    logger.log("=" * 30 + "{:^20}".format("Args") + "=" * 30)
    for arg, value in vars(args).items():
        logger.log(f"  {arg}: {value}\n")
    logger.log("=" * 80)

    logger.log("=" * 30 + "{:^20}".format("Exp Configs") + "=" * 30)
    logger.log(OmegaConf.to_yaml(exp_conf))
    logger.log("=" * 80)

    # run experiment with given args & conf
    exp_run(args, logger, **exp_conf)

    # End
    total_duration = time.time() - total_start_time
    logger.log("=" * 30 + f"    Finished. Total Time: {str(timedelta(seconds=int(total_duration)))}    " + "=" * 30)


if __name__ == "__main__":
    main()






