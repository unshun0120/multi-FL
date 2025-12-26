import os
import time 
import torch
from tqdm import tqdm
from collections import defaultdict
from datetime import timedelta

from config import get_config
from utils.seed import set_seed
from utils.logger import Logger
from data.datasets import load_partitioned_datasets
from utils.utils import initialize_training_clients, record_plot_total_acc, record_round_acc, save_gen_model
from server import Server
from utils.plotting import plot_new_client_accuracy
from models.hetero_model import get_heterogeneous_model
  
DATA_ROOT = './data/raw'
 
def load_checkpoint(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model checkpoint not found at {path}")
    print(f"Loading checkpoint from {path}...")
    return torch.load(path, map_location=device, weights_only=False)

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
    new_clients, id_to_dataset = initialize_training_clients(
        all_client_data_loaders, 
        dataset_meta, 
        model_list, 
        args, 
        DATA_ROOT
    )

    history = {
        'train_detail': defaultdict(list) # 每個global round後的準確率
    }
    round_acc_dataset = defaultdict(list)

    # =========================================
    # Round 0: Testing of Random initialized model
    # =========================================
    logger.log("--- Round 0: Testing of Random initialized model ---")

    for client in tqdm(new_clients, desc="Initial Testing", ncols=100):
        acc = client.test()
        round_acc_dataset = record_round_acc(0, client, 0, acc, id_to_dataset, round_acc_dataset, logger)

    history = record_plot_total_acc(0, round_acc_dataset, history, logger, args, mode="new_client_single_dataset")

    # =========================================
    # New client Training Loop
    # =========================================
    logger.log("\n" + "="*100)
    logger.log("--- Start New client Training Loop ---")
    logger.log("="*100)

    for client in new_clients:
        d_name = id_to_dataset[client.client_id]
        d_meta = dataset_meta[d_name]

        ls_id = server._get_label_space_id(client.class_names)

        if ls_id in checkpoint['classifiers'] and ls_id in checkpoint['generators']:
            global_clf_weight = checkpoint['classifiers'][ls_id]
            gen_weight = checkpoint['generators']

        client_curves = {}

        for arch_id in range(10):
            arch_name = model_list[arch_id]
            model = get_heterogeneous_model(
                client_id=arch_id, 
                in_channels=d_meta['in_ch'],
                num_classes=d_meta['classes'],
                img_size=d_meta['size'],
                global_dim=args.global_feature_dim
            )
            client.model = model

            acc_history = []
            acc_history.append(client.test())
            for epoch in tqdm(range(args.new_client_epochs)):
                client.model.train()
                for imgs, labels in client.train_loader:
                    imgs, labels = imgs.to(args.device), labels.to(args.device)
                    client.optimizer.zero_grad()
                    _, logits = client.model(imgs)
                    loss = client.criterion(logits, labels)
                    loss.backward()
                    client.optimizer.step()
                acc_history.append(client.test())

            msg = f"   [{arch_name}] Final {acc_history:.2f}%"
            logger.log(msg)

            client_curves[arch_id] = [round(float(a), 2) for a in acc_history]
            logger.log(f"      [{arch_name}] History: {client_curves[arch_id]}", print_to_console=False)

        plot_new_client_accuracy(client_curves, args, d_name, model_list, save_dir=logger.get_log_dir())

    

if __name__ == "__main__": 
    total_start_time = time.time()
    args = get_config()
    set_seed(args.seed)

    # =================================
    # Load generator checkpoint
    # =================================
    checkpoint = load_checkpoint(args.model_path, args.device)

    # Create folder to save training result
    if args.target_dir:
        parent_dir = args.target_dir
    else:
        parent_dir = os.path.dirname(args.model_path) or '.'
    sub_dir_name = f"new_client_{total_start_time}"
    save_dir = os.path.join(parent_dir, sub_dir_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger = Logger(args, mode="new_client_single_dataset")
    logger.log("\n" + "="*100, print_to_console=False)
    logger.log(f"   New Clients Training from with Generator Started")
    logger.log(f"   Target Model: {args.model_path}")
    logger.log(f"   Output Dir:   {save_dir}")
    logger.log(f"{'='*100}\n")

    main()

    # End Messages
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    formatted_total_time = str(timedelta(seconds=int(total_duration)))
    logger.log(f"=== Finished. Total Time: {formatted_total_time} ===")


