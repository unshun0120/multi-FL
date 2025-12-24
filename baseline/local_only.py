import os
import time 
from tqdm import tqdm
from collections import defaultdict
from datetime import timedelta

from config import get_config
from utils.seed import set_seed
from utils.logger import Logger
from data.datasets import load_partitioned_datasets
from utils.utils import initialize_training_clients, record_plot_total_acc, record_round_acc
from server import Server
from utils.plotting import plot_accuracy_curves
  
DATA_ROOT = './data/raw'
 
def main():
    # Preparing & Loading dataset
    # 'MNIST': [ {'train': DataLoader, 'test': DataLoader}, {'train': DataLoader, 'test': DataLoader}, ... ],
    # 'CIFAR10': [ ... ], ...
    all_client_data_loaders = load_partitioned_datasets(args, DATA_ROOT)

    # data input parameter of model & model architecture list 
    dataset_meta = { 
        'MNIST':        {'in_ch': 1, 'classes': 10,  'size': 28}, 
        'FashionMNIST': {'in_ch': 1, 'classes': 10,  'size': 28},
        'EMNIST':       {'in_ch': 1, 'classes': 47,  'size': 28},
        'CIFAR10':      {'in_ch': 3, 'classes': 10,  'size': 32},
        'CIFAR100':     {'in_ch': 3, 'classes': 100, 'size': 32},
        'CIFAR100_SUPER':{'in_ch': 3, 'classes': 20,  'size': 32},
        'TinyImageNet': {'in_ch': 3, 'classes': 200, 'size': 64} 
    }

    model_list = {
        0: 'MLP', 
        1: 'CNN', 
        2: 'ResNet8', 
        3: 'ResNet18',
        4: 'MobileNetV2', 
        5: 'MobileNetV3', 
        6: 'LeNet',
        7: 'AlexNet', 
        8: 'ShuffleNet', 
        9: 'SqueezeNet'
    }

    # Initializing Server
    print("Initializing Server...")
    server = Server(args)

    # Initializing Client
    train_clients, id_to_dataset = initialize_training_clients(
        all_client_data_loaders, 
        dataset_meta, 
        model_list, 
        args, 
        DATA_ROOT
    )

    # 記錄數據用來畫準確率圖
    history = {
        'train_detail': defaultdict(list) # 每個global round後的準確率
    }
    round_acc_dataset = defaultdict(list)

    # =========================================
    # Round 0: Testing of Random initialized model
    # =========================================
    logger.log("--- Round 0: Testing of Random initialized model ---")

    for client in tqdm(train_clients, desc="Initial Testing", ncols=100):
        acc = client.test()
        round_acc_dataset = record_round_acc(0, client, 0, acc, id_to_dataset, round_acc_dataset, logger)

    history = record_plot_total_acc(0, round_acc_dataset, history, logger, args)

    # =========================================
    # Local-Only Loop
    # =========================================
    logger.log("\n" + "="*100)
    logger.log("--- Start Local-Only Loop ---")
    logger.log("="*100)

    for rnd in range(args.global_rounds):
        round_start_time = time.time()
        logger.log(f"--- Round {rnd + 1} ---")

        round_acc_by_dataset = defaultdict(list)

        for client in tqdm(train_clients, desc="Local Training"):
            loss = client.local_train()
            acc = client.test()

            round_acc_dataset = record_round_acc(rnd+1, client, loss, acc, id_to_dataset, round_acc_dataset, logger)

        history = record_plot_total_acc(rnd+1, round_acc_dataset, history, logger, args, mode="local_only")

        plot_accuracy_curves(history, save_dir=logger.get_log_dir(), args=args, mode="local_only")
        round_end_time = time.time()
        round_duration = round_end_time - round_start_time
        print(f"> Round Time: {str(timedelta(seconds=int(round_duration)))}")

if __name__ == "__main__": 
    total_start_time = time.time()
    args = get_config()
    set_seed(args.seed)
    logger = Logger(args, mode="local_only")

    # Start Messages
    logger.log("=== Local-Only Baseline Started ===") 
    logger.log(f"{'='*100}")

    main()

    # End Messages
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    formatted_total_time = str(timedelta(seconds=int(total_duration)))
    logger.log(f"=== Baseline Finished. Total Time: {formatted_total_time} ===")


