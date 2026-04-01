import os
import json
import torch
import numpy as np
from collections import Counter
from torchvision import datasets, transforms
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Subset
from data.dirichlet_noniid import partition_data

class Global_Dataset(Dataset):
    def __init__(self, dataset, mapping):
        self.dataset = dataset
        self.mapping = mapping

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, local_label = self.dataset[idx]
        global_label = self.mapping[int(local_label)]
        return img, global_label

def get_split_cache_path(DATA_ROOT, dataset_name, alpha, total_clients, num_new_clients, seed):
    cache_dir = os.path.join(DATA_ROOT, "splits")
    os.makedirs(cache_dir, exist_ok=True)

    # 把alpha的.去掉改成p (e.g. 0.1 -> 0p1)
    alpha_str = f"{alpha:.1f}".replace(".", "p")

    filename = f"{dataset_name}_C{total_clients}_New{num_new_clients}_alpha{alpha_str}_seed{seed}.json"
    return os.path.join(cache_dir, filename)

# ==========================================
# Get Label Counts
# ==========================================
def get_label_counts(dataset, indices):
    """
    label count of subset
    """
    if hasattr(dataset, 'targets'):
        all_labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        all_labels = np.array(dataset.labels)
    else:
        all_labels = np.array([y for x, y in dataset.samples])
    
    subset_labels = all_labels[indices]

    counter = Counter(subset_labels)
    
    return ", ".join([f"{k}:{v}" for k, v in sorted(counter.items())])

# ==========================================
# Dataset Transform
# ==========================================
def get_transforms(name):
    if name in ['MNIST', 'FashionMNIST', 'USPS']:
        return transforms.Compose([
            transforms.Resize((32, 32)),                 
            transforms.Grayscale(num_output_channels=3), 
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))

            # transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            # transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,))
        ])

    elif name == 'EMNIST':
        return transforms.Compose([
            transforms.Resize((32, 32)),                 
            transforms.Grayscale(num_output_channels=3),
            # transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.transpose(1, 2)),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    elif name in ['CIFAR10', 'CIFAR100']:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    return None

# ==========================================
# Reading Raw Datasets
# ==========================================
def get_raw_dataset_transform(name, root, train=True):
    transform = get_transforms(name)
    
    if name == 'MNIST':
        return datasets.MNIST(root, train=train, download=False, transform=transform)
    
    elif name == 'FashionMNIST':
        return datasets.FashionMNIST(root, train=train, download=False, transform=transform)
    
    elif name == 'EMNIST':
        # return datasets.EMNIST(root, split='balanced', train=train, download=False, transform=transform)
        # return datasets.EMNIST(root, split='bymerge', train=train, download=False, transform=transform)
        return datasets.EMNIST(root, split='byclass', train=train, download=False, transform=transform)
    
    elif name == 'CIFAR10':
        return datasets.CIFAR10(root, train=train, download=False, transform=transform)
    
    elif name == 'CIFAR100':
        return datasets.CIFAR100(root, train=train, download=False, transform=transform)

    elif name == 'USPS':
        return datasets.USPS(root, train=train, download=True, transform=transform)
    
# ==========================================
# Get Readable Class Names
# ==========================================
def get_readable_class_names(name, root='./data/raw'):
    if name in ['MNIST', 'USPS']:
        return [str(i) for i in range(10)]

    elif name == 'FashionMNIST':
        return ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
    elif name == 'EMNIST':
        # d = datasets.EMNIST(root, split='balanced', train=True, download=False)
        # d = datasets.EMNIST(root, split='bymerge', train=True, download=True)
        d = datasets.EMNIST(root, split='byclass', train=True, download=True)
        return d.classes
        
    elif name == 'CIFAR10':
        # CIFAR10 in-build classes: ['airplane', 'automobile', 'bird', ...]
        d = datasets.CIFAR10(root, train=True, download=False)
        return d.classes
    
    elif name == 'CIFAR100':
        # CIFAR100 in-build classes: ['apple', 'aquarium_fish', ...]
        d = datasets.CIFAR100(root, train=True, download=False)
        return d.classes        

# ==========================================
# Loading Datasets
# ==========================================
def load_partitioned_datasets(args, DATA_ROOT, **exp_conf):
    # configs
    dataset_configs = {
        'MNIST': (args.num_train_mnist + args.num_new_clients) if args.num_train_mnist > 0 else 0,
        'FashionMNIST': (args.num_train_fashionmnist + args.num_new_clients) if args.num_train_fashionmnist > 0 else 0,
        'EMNIST': (args.num_train_emnist + args.num_new_clients) if args.num_train_emnist > 0 else 0,
        'CIFAR10': (args.num_train_cifar10 + args.num_new_clients) if args.num_train_cifar10 > 0 else 0,
        'CIFAR100': (args.num_train_cifar100 + args.num_new_clients) if args.num_train_cifar100 > 0 else 0,
        'USPS': (args.num_train_usps + args.num_new_clients) if args.num_train_usps > 0 else 0
    }
    batch_size = exp_conf.get('batch_size', 64)
    dirichlet_alpha = exp_conf.get('dirichlet_alpha', 0.1)
    public_ratio = exp_conf.get('public_data_ratio', 1.0)

    # Partition Datasets
    print(f"{'='*100}")
    print(f"Loading Datasets with Non-IID Split (Dirichlet distribution, Alpha={dirichlet_alpha})")
    print(f"{'='*100}")

    all_client_data_loaders = {}
    server_train_loaders = {}
    server_test_loaders = {}

    for d_name, n_clients in dataset_configs.items():
        if n_clients == 0:
            continue
        
        # --- Readable Class Names (e.g. dog, cat ...) ---
        class_names = get_readable_class_names(d_name, root=DATA_ROOT)
        print(f"[-] Readable Class Names ({len(class_names)} classes): {class_names}\n")

        print(f"\n>>> Processing {d_name} ({n_clients} Clients)")   

        # Getting Datasets
        train_dataset = get_raw_dataset_transform(d_name, DATA_ROOT, train=True)
        test_dataset = get_raw_dataset_transform(d_name, DATA_ROOT, train=False)

        # Getting Labels
        # 拿這個 dataset 底下每一筆資料的 label
        if hasattr(train_dataset, 'targets'): 
            train_labels = train_dataset.targets
        elif hasattr(train_dataset, 'labels'): 
            train_labels = train_dataset.labels
        else: 
            train_labels = []
            for path, label in train_dataset.samples:
                train_labels.append(label)

        if hasattr(test_dataset, 'targets'): 
            test_labels = test_dataset.targets
        elif hasattr(test_dataset, 'labels'): 
            test_labels = test_dataset.labels
        else: 
            test_labels = []
            for path, label in test_dataset.samples:
                test_labels.append(label)
                
        train_labels_for_split = train_labels
        test_labels_for_split  = test_labels

        # Dirichlet Non-IID Partition
        cache_path = get_split_cache_path(
            DATA_ROOT,
            d_name,
            dirichlet_alpha,
            n_clients,
            args.num_new_clients,
            args.seed 
        )

        if os.path.exists(cache_path):
            print(f"Found existing split for {d_name}, loading from {cache_path}")
            with open(cache_path, "r") as f:
                cached = json.load(f)
            # json 會把 key 變成字串所以要轉回 int
            train_idcs = {int(k): v for k, v in cached["train"].items()}
            test_idcs  = {int(k): v for k, v in cached["test"].items()}
        else:
            print(f"No existing split for {d_name}, generating new partition...")
            train_idcs, test_idcs = partition_data(
                train_labels_for_split, 
                test_labels_for_split, 
                alpha=dirichlet_alpha, 
                total_clients=n_clients, 
                num_new_clients=args.num_new_clients
            )
            # 把用這次參數切的資料集存在 json
            to_save = {
                "train": train_idcs,
                "test": test_idcs,
                "meta": {
                    "dataset": d_name,
                    "alpha": dirichlet_alpha,
                    "total_clients": n_clients,
                    "num_new_clients": args.num_new_clients,
                    "seed": args.seed,
                }
            }
            with open(cache_path, "w") as f:
                json.dump(to_save, f, indent=2)
            print(f"Saved split for {d_name} to {cache_path}")

        # Dataloader
        client_loaders = []
        print(f"{'Client':<6} | {'Train':<6} | {'Test':<6} | {'Sample'}")
        print("-" * 120)

        for i in range(n_clients):
            train_subset = Subset(train_dataset, train_idcs[i])
            test_subset = Subset(test_dataset, test_idcs[i])

            train_cnt = len(train_subset)
            test_cnt = len(test_subset)
            train_info_str = get_label_counts(train_dataset, train_idcs[i])
            test_info_str  = get_label_counts(test_dataset,  test_idcs[i])

            print(f"{i:<6} | {train_cnt:<6} | {test_cnt:<6} | Train: [{train_info_str}]")
            print(f"{'':<6} | {'':<6} | {'':<6} | Test : [{test_info_str}]")
            print("-" * 60) 
            
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0)
            
            client_loaders.append({
                'train': train_loader,
                'test': test_loader
            })

        all_client_data_loaders[d_name] = client_loaders

        if public_ratio < 1.0:
            total_len = len(train_dataset)
            subset_len = int(total_len * public_ratio)
            indices = torch.randperm(total_len)[:subset_len]
            public_train_dataset = Subset(train_dataset, indices)
            
            print(f"   [Data Subsampling] Using {subset_len}/{total_len} samples ({public_ratio*100}%) for {d_name}")
        else:
            public_train_dataset = train_dataset

        server_train_loaders[d_name] = DataLoader(
            public_train_dataset, 
            batch_size=batch_size, 
            shuffle=True,  
            num_workers=0
        )

        server_test_loaders[d_name] = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0
        )

    print(f"{'='*100}\n")

    return all_client_data_loaders, server_train_loaders, server_test_loaders

