import os
import time
import random
import torch
import numpy as np
import argparse
from datetime import timedelta
from importlib import import_module
from torch.utils.data import ConcatDataset, DataLoader

from data.datasets import Global_Dataset, load_partitioned_datasets, get_readable_class_names
from utils.logger import Logger
from utils.csv_logger import init_csv, append_csv
from utils.plotting import plot_new_client_accuracy
from utils.nets import get_heterogeneous_model
from utils.test_utils import (
    initialize_new_clients, load_checkpoint, load_generator, load_shared_classifier, evaluate_client
)

DATA_ROOT = './data/raw'
MODEL_LIST = {
    0: 'MLP', 1: 'CNN', 2: 'ResNet8', 3: 'ResNet18',
    4: 'MobileNetV2', 5: 'MobileNetV3', 6: 'LeNet',
    7: 'AlexNet', 8: 'ShuffleNet', 9: 'SqueezeNet'
}
DATASET_META = { 
    'MNIST':        {'in_ch': 3, 'classes': 10,  'size': 32},  
    'FashionMNIST': {'in_ch': 3, 'classes': 10,  'size': 32},
    'EMNIST':       {'in_ch': 3, 'classes': 47,  'size': 32},
    'CIFAR10':      {'in_ch': 3, 'classes': 10,  'size': 32},
    'CIFAR100':     {'in_ch': 3, 'classes': 100, 'size': 32}
}

class BaseNewClientTrainer:
    def __init__(self, mode_name, args):
        self.mode_name = mode_name
        self.args = args
        self.total_start_time = time.time()
        
        self.checkpoint = load_checkpoint(self.args.model_path, self.args.device)
        self.exp_conf = self.checkpoint['exp_conf']
        self.saved_args = self.checkpoint['args']

        self.args.algorithm = self.saved_args.get('algorithm', 'Ours')
        self.args.seed = self.saved_args.get('seed', None)
        self.args.num_new_clients = self.saved_args.get('num_new_clients', 0)
        self.args.num_train_mnist = self.saved_args.get('num_train_mnist', 0)
        self.args.num_train_emnist = self.saved_args.get('num_train_emnist', 0)
        self.args.num_train_fashionmnist = self.saved_args.get('num_train_fashionmnist', 0)
        self.args.num_train_cifar10 = self.saved_args.get('num_train_cifar10', 0)
        self.args.num_train_cifar100 = self.saved_args.get('num_train_cifar100', 0)
        
        self.set_seed(self.args.seed)

        self.global_feature_dim = self.checkpoint['global_feature_dim']
        self.feat_gen_noise_dim = self.checkpoint.get('feat_gen_noise_dim', 128)

        self.global_registry = self.checkpoint.get('global_registry', {})
        if 'num_global_classes' in self.checkpoint:
             self.num_global_classes = self.checkpoint['num_global_classes']
        else:
             # 如果 checkpoint 沒存，就自己算
             all_gids = set()
             for d_map in self.global_registry.values():
                 all_gids.update(d_map.values())
             self.num_global_classes = max(all_gids) + 1 if all_gids else 0

        logger_mode = f"{self.mode_name}_{self.args.dataset_mode}_dataset"
        self.logger = Logger(args, mode=logger_mode)
        self.save_dir = self.logger.log_dir
        self.csv_path = init_csv(self.save_dir)

        self.logger.log(f"\n{'='*30} New Client: {self.mode_name} {'='*30}")
        self.logger.log(f"  Target Model: {self.args.model_path}")
        self.logger.log(f"  Dataset Mode: {self.args.dataset_mode}")
        self.logger.log(f"  Output Dir:   {self.save_dir}")
        self.logger.log(f"{'='*80}")

        # 3. 載入資料與共用模型
        (self.all_client_data_loaders, self.global_id_map, self.global_registry, 
         self.public_train_loaders, self.public_test_loaders) = load_partitioned_datasets(
            self.args, DATA_ROOT, **self.exp_conf
        )
        # self.generator = load_generator(self.checkpoint, self.args.device)
        # self.shared_classifier = load_shared_classifier(self.checkpoint, self.args.device)

    def set_seed(self, seed):
        if seed is not None:
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)                
            torch.backends.cudnn.deterministic = True        
            torch.backends.cudnn.benchmark = False 

    def run(self):
        if self.args.dataset_mode == "single":
            self._run_single_dataset()
        elif self.args.dataset_mode == "super":
            self._run_super_dataset()
        
        total_duration = time.time() - self.total_start_time
        self.logger.log(f"\n{'='*30} Finished. Total Time: {str(timedelta(seconds=int(total_duration)))} {'='*30}")

    def _run_single_dataset(self):
        ClientClass = getattr(import_module(f"trainer.{self.args.algorithm}.client"), 'Client')
        new_clients = initialize_new_clients(
            ClientClass, self.all_client_data_loaders, DATASET_META, MODEL_LIST, DATA_ROOT, self.global_registry, 
            self.args, self.logger, **self.exp_conf
        )

        for client in new_clients:
            d_name = client.dataset_name
            d_meta = DATASET_META[d_name]
            ls_id = str(sorted(get_readable_class_names(d_name, DATA_ROOT)))

            if d_name not in self.global_registry:
                continue

            self.logger.log(f"\n{'='*50}\nDataset: {d_name} | Client ID: {client.id}\n{'='*50}")
            label_to_global_id = self.global_id_map[d_name]
            client_curves = {}

            for arch_id, arch_name in MODEL_LIST.items():
                self.logger.log(f"\n  [{arch_name}] Training...")
                client.model = get_heterogeneous_model(
                    node_id=arch_id, in_channels=d_meta['in_ch'], num_classes=d_meta['classes'],
                    img_size=d_meta['size'], global_dim=self.global_feature_dim
                )

                acc_history = self.train_client(
                    client=client, d_name=d_name, arch_name=arch_name, 
                    num_classes=d_meta['classes'], label_to_global_id=label_to_global_id
                )
                
                client_curves[arch_id] = [round(float(a), 2) for a in acc_history]

            plot_new_client_accuracy(client_curves, self.args, d_name, MODEL_LIST, save_dir=self.save_dir)

    def _run_super_dataset(self):
        self.logger.log(f"\n{'='*100}\n--- [Super Dataset] Mode ---\n{'='*100}")
        
        train_datasets, test_datasets = [], []
        for d_name in self.public_train_loaders.keys():
            if d_name not in self.global_id_map: 
                continue

            label_to_global_id = self.global_id_map[d_name]
            train_datasets.append(Global_Dataset(self.public_train_loaders[d_name].dataset, label_to_global_id))
            test_datasets.append(Global_Dataset(self.public_test_loaders[d_name].dataset, label_to_global_id))

        batch_size = self.exp_conf.get('batch_size', 64)
        super_train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=batch_size, shuffle=True)
        super_test_loader = DataLoader(ConcatDataset(test_datasets), batch_size=batch_size, shuffle=False)

        class SuperClient:
            def __init__(self, model, train_loader, test_loader):
                self.model, self.train_loader, self.test_loader = model, train_loader, test_loader
                self.id, self.dataset_name = "super", "SuperDataset"

        identity_mapping = {i: i for i in range(self.num_global_classes)}
        client_curves = {}

        for arch_id, arch_name in MODEL_LIST.items():
            self.logger.log(f"\n{'=' * 50}\n[{arch_name}] Training on Super Dataset...\n{'=' * 50}")
            model = get_heterogeneous_model(
                node_id=arch_id, in_channels=3, num_classes=self.num_global_classes,
                img_size=32, global_dim=self.global_feature_dim
            )
            client = SuperClient(model, super_train_loader, super_test_loader)

            acc_history = self.train_client(
                client=client, d_name="SuperDataset", arch_name=arch_name, 
                num_classes=self.num_global_classes, label_to_global_id=identity_mapping
            )
            client_curves[arch_id] = [round(float(a), 2) for a in acc_history]

        plot_new_client_accuracy(client_curves, self.args, "SuperDataset", MODEL_LIST, save_dir=self.save_dir)

    def train_client(self, client, d_name, arch_name, num_classes, label_to_global_id):
        pass


def get_base_parser():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--exp_timestamp", type=str, default=None)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset_mode", type=str, default="single", choices=["single", "super"])
    parser.add_argument("--new_client_epochs", type=int, default=30)
    return parser