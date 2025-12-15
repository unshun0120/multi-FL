import os
import json
import numpy as np
from collections import Counter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from data.dirichlet_noniid import partition_data

CIFAR100_SUPER_NAMES = [
    "aquatic_mammals",
    "fish",
    "flowers",
    "food_containers",
    "fruit_and_vegetables",
    "household_electrical_devices",
    "household_furniture",
    "insects",
    "large_carnivores",
    "large_man_made_outdoor_things",
    "large_natural_outdoor_scenes",
    "large_omnivores_and_herbivores",
    "medium_mammals",
    "non_insect_invertebrates",
    "people",
    "reptiles",
    "small_mammals",
    "trees",
    "vehicles_1",
    "vehicles_2"
]

FINE_TO_COARSE = [
    4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
    3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
    6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
    0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
    5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
    16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
    10, 3,  2, 12, 12, 16, 12,  1,  9, 19, 
    2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
    16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
    18, 1,  2, 15,  6,  0, 17,  8, 14, 13
]

def get_split_cache_path(DATA_ROOT, dataset_name, alpha, total_clients, num_new_clients, seed):
    """
    回傳某個(dataset, alpha, client 數, new_client 數, seed)對應的split檔案路徑
    只要參數一樣就會用同一個non-iid切的檔案
    """
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

    # cifar100 從 sub-class 轉成 super-class
    if mapping is not None:
        subset_labels = np.array([mapping[y] for y in subset_labels])

    counter = Counter(subset_labels)
    
    return ", ".join([f"{k}:{v}" for k, v in sorted(counter.items())])

# ==========================================
# Dataset Transform
# ==========================================
def get_transforms(name):
    if name in ['MNIST', 'EMNIST', 'FashionMNIST']:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    elif name in ['CIFAR10', 'CIFAR100', 'CIFAR100_SUPER']:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    return None

# ==========================================
# Reading Raw Datasets
# ==========================================
def get_raw_dataset(name, root, train=True):
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
    
    elif name == 'CIFAR100_SUPER':
        return datasets.CIFAR100(root, train=train, download=False, transform=transform, target_transform=lambda y: FINE_TO_COARSE[y]
)
    
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
    
    elif name == 'CIFAR100_SUPER':
        return CIFAR100_SUPER_NAMES

# ==========================================
# Loading Datasets
# ==========================================
def load_partitioned_datasets(args, DATA_ROOT):
    dataset_configs = {
        'MNIST': args.num_mnist,
        'FashionMNIST': args.num_fashionmnist,
        'EMNIST': args.num_emnist,
        'CIFAR10': args.num_cifar10,
        'CIFAR100': args.num_cifar100,
        'CIFAR100_SUPER': args.num_cifar100_super
    }

    all_client_data_loaders = {}

    print(f"{'='*100}")
    print(f"Loading Datasets with Non-IID Split (Dirichlet distribution, Alpha={args.dirichlet_alpha})")
    print(f"{'='*100}")

    # 把資料集分給 client
    for d_name, n_clients in dataset_configs.items():
        if n_clients == 0:
            continue

        print(f"\n>>> Processing {d_name} ({n_clients} Clients)...")

        # --- Readable Class Names (e.g. dog, cat ...)---
        class_names = get_readable_class_names(d_name, root=DATA_ROOT)
        print(f"[-] Readable Class Names ({len(class_names)} classes): {class_names}\n")

        # Getting Datasets
        train_dataset = get_raw_dataset(d_name, DATA_ROOT, train=True)
        test_dataset = get_raw_dataset(d_name, DATA_ROOT, train=False)

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

        if d_name == 'CIFAR100_SUPER':
            train_labels_for_split = [FINE_TO_COARSE[y] for y in train_labels]
            test_labels_for_split  = [FINE_TO_COARSE[y] for y in test_labels]
        else:
            train_labels_for_split = train_labels
            test_labels_for_split  = test_labels

        # Dirichlet Non-IID Partition
        cache_path = get_split_cache_path(
            DATA_ROOT,
            d_name,
            args.dirichlet_alpha,
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
                alpha=args.dirichlet_alpha, 
                total_clients=n_clients, 
                num_new_clients=args.num_new_clients
            )
            # 把用這次參數切的資料集存在 json
            to_save = {
                "train": train_idcs,
                "test": test_idcs,
                "meta": {
                    "dataset": d_name,
                    "alpha": args.dirichlet_alpha,
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

            if d_name == 'CIFAR100_SUPER':
                # super-class 0~19
                train_info_str = get_label_counts(train_dataset, train_idcs[i], mapping=FINE_TO_COARSE)
                test_info_str  = get_label_counts(test_dataset,  test_idcs[i], mapping=FINE_TO_COARSE)
            else:
                train_info_str = get_label_counts(train_dataset, train_idcs[i])
                test_info_str  = get_label_counts(test_dataset,  test_idcs[i])


            print(f"{i:<6} | {train_cnt:<6} | {test_cnt:<6} | Train: [{train_info_str}]")
            print(f"{'':<6} | {'':<6} | {'':<6} | Test : [{test_info_str}]")
            print("-" * 60) 
            
            train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            
            client_loaders.append({
                'train': train_loader,
                'test': test_loader
            })

        all_client_data_loaders[d_name] = client_loaders

    print(f"{'='*100}\n")

    return all_client_data_loaders

