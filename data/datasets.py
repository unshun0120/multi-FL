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
# 統計每個類別出現次數
# ==========================================
def get_label_counts(dataset, indices, mapping=None):
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
    if name in ['MNIST', 'EMNIST', 'FashionMNIST']:
        return transforms.Compose([
            transforms.Resize((32, 32)),                 
            transforms.Grayscale(num_output_channels=3), 
            transforms.ToTensor(),
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
        return datasets.EMNIST(root, split='balanced', train=train, download=False, transform=transform)
    
    elif name == 'CIFAR10':
        return datasets.CIFAR10(root, train=train, download=False, transform=transform)
    
    elif name == 'CIFAR100':
        return datasets.CIFAR100(root, train=train, download=False, transform=transform)

    
# ==========================================
# Get Readable Class Names
# ==========================================
def get_readable_class_names(name, root='./data/raw'):
    if name == 'MNIST':
        return [str(i) for i in range(10)]

    elif name == 'FashionMNIST':
        return ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
    elif name == 'EMNIST':
        d = datasets.EMNIST(root, split='balanced', train=True, download=False)
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
def load_partitioned_datasets(args, DATA_ROOT, **kwargs):
    dataset_configs = {
        'MNIST': args.num_train_mnist + args.num_new_clients,
        'FashionMNIST': args.num_train_fashionmnist + args.num_new_clients,
        'EMNIST': args.num_train_emnist + args.num_new_clients,
        'CIFAR10': args.num_train_cifar10 + args.num_new_clients,
        'CIFAR100': args.num_train_cifar100 + args.num_new_clients
    }
    batch_size = kwargs.get('batch_size', 64)
    dirichlet_alpha = kwargs.get('dirichlet_alpha', 0.1)

    all_client_data_loaders = {}
    
    global_id_map = {}
    global_registry = {}  
    next_global_id = 0    
    super_dataset_train_list = [] 
    super_dataset_test_list = []

    print(f"{'='*100}")
    print(f"Loading Datasets with Non-IID Split (Dirichlet distribution, Alpha={dirichlet_alpha})")
    print(f"{'='*100}")

    for d_name, n_clients in dataset_configs.items():
        if n_clients == 0:
            continue
        
        # --- Readable Class Names (e.g. dog, cat ...) ---
        class_names = get_readable_class_names(d_name, root=DATA_ROOT)
        print(f"[-] Readable Class Names ({len(class_names)} classes): {class_names}\n")

        current_dataset_map = {}

        for local_id, class_name in enumerate(class_names):
            if class_name in global_registry:
                gid = global_registry[class_name]
            else:
                gid = next_global_id
                global_registry[class_name] = gid
                next_global_id += 1
            
            current_dataset_map[local_id] = gid

        global_id_map[d_name] = current_dataset_map

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

        global_train_ds = Global_Dataset(train_dataset, mapping=current_dataset_map)
        global_test_ds = Global_Dataset(test_dataset, mapping=current_dataset_map)
        
        super_dataset_train_list.append(global_train_ds)
        super_dataset_test_list.append(global_test_ds)

    print(f"{'='*100}\n")

    super_train_dataset = ConcatDataset(super_dataset_train_list)
    super_test_dataset = ConcatDataset(super_dataset_test_list)

    server_train_loader = DataLoader(
        super_train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0
    )

    server_test_loader = DataLoader(
        super_test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )

    return all_client_data_loaders, global_id_map, global_registry, server_train_loader, server_test_loader

