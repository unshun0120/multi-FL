import os
import time 
from tqdm import tqdm
from collections import defaultdict
from datetime import timedelta

from config import get_config
from utils.seed import set_seed
from utils.logger import Logger
from data.datasets import load_partitioned_datasets
from utils.train_utils import (
    initialize_training_clients, 
    final_round_acc, 
    record_round_acc,
    initial_evaluation, 
    client_local_training
)
from utils.plotting import plot_accuracy_curves
  
DATA_ROOT = './data/raw'

# data input parameter of model & model architecture list 
DATASET_META = { 
    'MNIST':        {'in_ch': 1, 'classes': 10,  'size': 28},  
    'FashionMNIST': {'in_ch': 1, 'classes': 10,  'size': 28},
    'EMNIST':       {'in_ch': 1, 'classes': 47,  'size': 28},
    'CIFAR10':      {'in_ch': 3, 'classes': 10,  'size': 32},
    'CIFAR100':     {'in_ch': 3, 'classes': 100, 'size': 32},
    'CIFAR100_SUPER':{'in_ch': 3, 'classes': 20,  'size': 32},
    'TinyImageNet': {'in_ch': 3, 'classes': 200, 'size': 64} 
}

MODEL_LIST = {
    0: 'MLP', 1: 'CNN', 2: 'ResNet8', 3: 'ResNet18',
    4: 'MobileNetV2', 5: 'MobileNetV3', 6: 'LeNet',
    7: 'AlexNet', 8: 'ShuffleNet', 9: 'SqueezeNet'
}
 
def main():
    total_start_time = time.time()
    args = get_config()
    set_seed(args.seed)
    logger = Logger(args, mode="local_only")

    # Start Messages
    logger.log("=== Local-Only Baseline Started ===") 
    logger.log(f"{'='*100}")

    # Preparing & Loading dataset
    # 'MNIST': [ {'train': DataLoader, 'test': DataLoader}, {'train': DataLoader, 'test': DataLoader}, ... ],
    # 'CIFAR10': [ ... ], ...
    all_client_data_loaders = load_partitioned_datasets(args, DATA_ROOT)

    # Initializing Client
    print("Initializing Clients...")
    train_clients, id_to_dataset = initialize_training_clients(
        all_client_data_loaders, DATASET_META, MODEL_LIST, args, DATA_ROOT
    )

    history = defaultdict(list)      # {dataset_name: [avg_acc_round0, avg_acc_round1, ...]}
    round_acc = defaultdict(list)    # 存當前round的準確率

    # =========================================
    # Round 0: Testing of Random initialized model
    # =========================================
    logger.log("--- Round 0: Testing of Random initialized model ---")
    round_acc = initial_evaluation(train_clients, id_to_dataset, logger)
    history = final_round_acc(0, round_acc, history, logger, args, "local_only")

    # =========================================
    # Local-Only Loop
    # =========================================
    logger.log("\n" + "="*100)
    logger.log("--- Start Local Only Loop ---")
    logger.log("="*100)

    for rnd in range(args.global_rounds):
        round_start_time = time.time()
        logger.log(f"--- Round {rnd} ---")

        _, round_acc = client_local_training(
            train_clients, id_to_dataset, round_acc, rnd, logger
        )
        history = final_round_acc(rnd, round_acc, history, logger, args, mode="local_only")

        plot_accuracy_curves(history, save_dir=logger.get_log_dir(), args=args, mode="local_only")
        
        round_duration = time.time() - round_start_time
        print(f"> Round Time: {str(timedelta(seconds=int(round_duration)))}")

    # End Messages
    total_duration = time.time() - total_start_time
    logger.log(f"=== Baseline Finished. Total Time: {str(timedelta(seconds=int(total_duration)))} ===")

if __name__ == "__main__": 
    main()


