import os
import time 
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
from datetime import timedelta

from config import get_config
from utils.seed import set_seed
from utils.logger import Logger
from data.datasets import load_partitioned_datasets, get_readable_class_names
from utils.test_utils import (
    initialize_new_clients, 
    load_checkpoint, 
    load_generator, 
    build_label_mapping, 
    train_with_generator
)
from server import Server
from utils.plotting import plot_new_client_accuracy
from models.hetero_model import get_heterogeneous_model, ConditionalGenerator, Classifier
  
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

    logger = Logger(args, mode="new_client_single_dataset")
    args = get_config()
    set_seed(args.seed)

    # =================================
    # Load generator checkpoint
    # =================================
    checkpoint = load_checkpoint(args.model_path, args.device, logger)

    # Create folder to save training result
    parent_dir = os.path.dirname(args.model_path) or '.'
    sub_dir_name = f"new_client_{total_start_time}"
    save_dir = os.path.join(parent_dir, sub_dir_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    logger.log("\n" + "="*100, print_to_console=False)
    logger.log(f"   New Clients Training from with Generator Started")
    logger.log(f"   Target Model: {args.model_path}")
    logger.log(f"   Output Dir:   {save_dir}")
    logger.log(f"{'='*100}\n")

    # Load datasets
    all_client_data_loaders = load_partitioned_datasets(args, DATA_ROOT)
    generator = load_generator(checkpoint, args)
    
    # Initialize Server
    server = Server(args)

    # Initializing Client
    new_clients, id_to_dataset = initialize_new_clients(
        all_client_data_loaders, DATASET_META, MODEL_LIST, args, DATA_ROOT
    )

    # =========================================
    # New client Training Loop
    # =========================================
    logger.log("\n" + "="*100)
    logger.log("--- Start New Client Training with Generator ---")
    logger.log("="*100)
    for client in new_clients:
        d_name = id_to_dataset[client.client_id]
        d_meta = DATASET_META[d_name]

        full_class_names = get_readable_class_names(d_name, DATA_ROOT)
        ls_id = server._get_label_space_id(full_class_names)
        
        if ls_id not in checkpoint['dataset_classifiers']:
            logger.log(f"[Skip] No classifier found for {d_name}")
            continue

        label_to_global_id = build_label_mapping(full_class_names, checkpoint)

        logger.log(f"\n{'='*50}")
        logger.log(f"Dataset: {d_name} | Client ID: {client.client_id}")
        logger.log(f"{'='*50}")

        client_curves = {}
        for arch_id, arch_name in MODEL_LIST.items():
            logger.log(f"\n  [{arch_name}] Training...")

            client.model = get_heterogeneous_model(
                client_id=arch_id, 
                in_channels=d_meta['in_ch'],
                num_classes=d_meta['classes'],
                img_size=d_meta['size'],
                global_dim=args.global_feature_dim
            )

            # Train with generator assistance
            acc_history = train_with_generator(
                client, generator, label_to_global_id, args, logger
            )

            client_curves[arch_id] = [round(float(a), 2) for a in acc_history]
            logger.log(f"      [{arch_name}] History: {client_curves[arch_id]}")

        plot_new_client_accuracy(client_curves, args, d_name, MODEL_LIST, save_dir=logger.get_log_dir())

    # End Messages
    total_duration = time.time() - total_start_time
    logger.log(f"=== Finished. Total Time: {str(timedelta(seconds=int(total_duration)))} ===")

if __name__ == "__main__": 
    main()

    


