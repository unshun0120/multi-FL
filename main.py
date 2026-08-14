import argparse
import time 
import os
import random
import numpy as np
import torch
from omegaconf import OmegaConf
from datetime import timedelta
from importlib import import_module

from utils.logger import Logger
from data.datasets import load_partitioned_datasets
from utils.train_utils import initialize_training_clients
from utils.nets import ResNet, BasicBlock 


DATA_ROOT = './data/raw'

IMPLEMENTED_ALGORITHMS = ["Local", # FL with local only
                          "FedTED", # FL with data free knowledge distillation 
                          "FedTED_DDPM",
                          "FedTED_DDPM_2",
                          "FedTED_dir", # FL with data free knowledge distillation (dirichlet soft labels)
                          "GeFL", # FL with generative model 
                          "GeFL_local", # no aggregate generative model
                          "UDON", # Only KD not FL
                          "Ours", # Ours
                          "Ours_GeFL", # Ours with generative model
                          "GeFL_DeepInversion",
                          "GeFL_DeepInversion_gen",
                          "GeFL_DDPM",
                          "GeFL_slamdunk",  
                          "GeFL_GAN_DDPM",
                          'GeFL_DDPM_baseline', 
                          'GeFL_GAN_baseline', 
                          'GeFL_DDPM_baseline_total',
                          'GeFL_DDPM_baseline_total_DI',
                          'GeFL_DDPM_baseline_total_gan',
                          'GeFL_DDPM_baseline_total_gan_sep',
                          'GeFL_DDPM_baseline_total_gan_noniid',
                          'GeFL_DDPM_baseline_total_public',
                          'GeFL_DDIM_baseline_total',
                          'GeFL_DDPM_baseline_total_sep',
                          'GeFL_gan_missing_link',
                          "GeFL_gan_pacfl_iid",
                          "GeFL_gan_pacfl_iid_2",
                          "GeFL_gan_pacfl_noniid",
                          "BaseFL_public",
                          "BaseFL_DDPM",
                          "BaseFL_public_kd",
                          ]

IMPLEMENTED_LM = ["image-bi", "image-single", "feature-bi", "image-cs", "slam_dunk", "missing_link", 
                  "class_name", "independent", "identical", 
                  "cs_mapping", "feature_mapping", "single_mapping", "slam_dunk_mapping",
                  "ours_5", "ours_10", "ours_15", "ours_20", "ours_25",
                  "ours_5_noEntropy", 
                  "slam_dunk_mapping_5", "slam_dunk_mapping_10", "slam_dunk_mapping_15", "slam_dunk_mapping_20", "slam_dunk_mapping_25",
                  "improve_single_noniid",
                  ]   

DATASET_META = { 
    'MNIST':        {'in_ch': 3, 'classes': 10,  'size': 32},  
    'FashionMNIST': {'in_ch': 3, 'classes': 10,  'size': 32},
    #'EMNIST':       {'in_ch': 3, 'classes': 47,  'size': 32},
    'EMNIST':       {'in_ch': 3, 'classes': 62,  'size': 32},
    'CIFAR10':      {'in_ch': 3, 'classes': 10,  'size': 32},
    'CIFAR100':     {'in_ch': 3, 'classes': 100, 'size': 32},
    'USPS':        {'in_ch': 3, 'classes': 10,  'size': 32}, 
    # 'MNIST':        {'in_ch': 1, 'classes': 10,  'size': 28}, 
    # 'USPS':        {'in_ch': 1, 'classes': 10,  'size': 28}
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
    parser.add_argument("--exp_conf", type=str, default="./configs/het-exp.yaml",
                        help="experiment config yaml files")
    parser.add_argument("--device", type=str, default="cuda:1", 
                        help="run device (cpu | cuda:x, x:int > 0)")
    parser.add_argument("--seed", type=int, default=None, 
                        help="random seed")
    parser.add_argument("--algorithm", type=str, default="Ours", choices=IMPLEMENTED_ALGORITHMS,
                        help=f"the implemented algorithms, choices include: {IMPLEMENTED_ALGORITHMS}")
    parser.add_argument("--noniid_partition", type=str, default="dirichlet", choices=["dirichlet", "noniid_label", "quantity_skew", "quantity_skew_equalSize"])
    parser.add_argument("--aggregate_method", type=str, default="pacfl", choices=["dataset_name", "pacfl"])
    
    # --- Dataset ---
    parser.add_argument('--num_new_clients', type=int, default=0, 
                        help='Number of IID New clients for generalization test (for all dataset)')
    parser.add_argument("--num_train_mnist", type=int, default=10, 
                        help="number of training clients for mnist")
    parser.add_argument("--num_train_emnist", type=int, default=10, 
                        help="number of training clients for emnist")
    parser.add_argument("--num_train_fashionmnist", type=int, default=10, 
                        help="number of training clients for fashion-mnist")
    parser.add_argument("--num_train_cifar10", type=int, default=10, 
                        help="number of training clients for cifar-10")
    parser.add_argument("--num_train_cifar100", type=int, default=10, 
                        help="number of training clients for cifar100")
    parser.add_argument("--num_train_usps", type=int, default=10, 
                        help="number of training clients for USPS")
    
    # --- Config ---
    parser.add_argument("--label_mapping", type=str, default="image-bi", choices=IMPLEMENTED_LM,
                        help="label mapping method")
    parser.add_argument("--start_mapping_epoch", type=int, default=25,
                        help="epoch to start label mapping")
    parser.add_argument("--pacfl_basis_budget", type=int, default=20, 
                                help="cluster alpha for PACFL")
    parser.add_argument("--pacfl_cluster_alpha", type=int, default=20, 
                            help="cluster alpha for PACFL")

    return parser.parse_args()


def exp_run(args, logger, **exp_conf):
    # 1. set random seed
    set_seed(args.seed)

    # 2. prepare dataset and public dataset
    all_client_data_loaders, public_train_loaders, public_test_loaders = load_partitioned_datasets(args, DATA_ROOT, **exp_conf)

    # 3. create FL training clients
    Client = getattr(import_module("trainer.%s.client" % args.algorithm), 'Client')
    train_clients = initialize_training_clients(
        Client, all_client_data_loaders, DATASET_META, MODEL_LIST, args, DATA_ROOT, logger, **exp_conf
    )

    # 4. create server
    Server = getattr(import_module("trainer.%s.server" % args.algorithm), 'Server')
    # server_model = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=3, num_classes=100, global_dim=256)
    server = Server(
        node_id=-1,
        args=args, 
        clients=train_clients,
        dataset_name = None, 
        train_loader=public_train_loaders, 
        test_loader=public_test_loaders,
        model=None,
        class_name_set=None,
        model_name="ServerResNet18",
        device=args.device,
        logger=logger,
        exp_conf=dict(exp_conf),
        **exp_conf,
    )

    # 5. start training
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
        logger.log(f"  {arg}: {value}")
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
