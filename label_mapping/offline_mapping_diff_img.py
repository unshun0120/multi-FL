import os
import sys
import re
import csv
import glob
import argparse
import torch
import time
from datetime import datetime
from contextlib import redirect_stdout
import random
import numpy as np
from torchvision import datasets, transforms
import json
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.nets import (
    ConditionalImageGenerator,
    DCGANGenerator,
    MLP,
    CNN,
    ResNet,
    BasicBlock,
    MobileNetV2,
    MobileNetV3,
    LeNet,
    AlexNet,
    ShuffleNetV2,
    SqueezeNet
)

from utils.DFKD_utils import get_image_prior_losses, DeepInversionHook

from label_mapping_utils import (
    get_gen_images,
    clear_image_caches,
    label_mapping,
    single_direction_label_mapping,
    image_cosine_similarity_mapping,
    evaluate_mapping_results,
    global_to_local_mapping,
    missing_link_label_mapping,
    improve_label_mapping,
    improve_label_mapping_noniid,
    improve_label_mapping_noniid_temp,
    improve_label_mapping_noniid_temp_2,
    feature_bi_direction_label_mapping,
)
from slam_dunk import slam_dunk_label_mapping
from utils.loss import Gen_DiversityLoss
import label_mapping_utils as lm_utils

di_weight = {
    'epochs': 1,
    'g_steps': 15000,
    'lr': 1e-3,
    'ce': 1.0,
    'bn': 0.05,
    'tv': 0.005,
    'l2': 0.0,
    'adv': 0.0,
}

class SimpleLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def log(self, msg):
        print(msg)


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip() != ""]


def parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip() != ""]


def load_server_checkpoint(log_dir, global_round, device):
    ckpt_path = os.path.join(log_dir, f"server_checkpoints_{global_round}.pth")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Server checkpoint not found: {ckpt_path}")

    checkpoint = safe_torch_load(ckpt_path, device)
    print(f"[Loaded Server] {ckpt_path}")
    return checkpoint


def get_client_id_from_path(path):
    match = re.search(r"_c(\d+)", os.path.basename(path))

    if match is None:
        raise ValueError(f"Cannot parse client id from filename: {path}")

    return int(match.group(1))


def get_arch_name_from_path(model_path):
    filename = os.path.basename(model_path).replace(".pth", "")
    return filename.split("_")[-1]


def get_test_transform(dataset_name):
    if dataset_name in ["MNIST"]:
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    elif dataset_name == "EMNIST":
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.transpose(1, 2)),
            transforms.Normalize((0.5,), (0.5,))
        ])

    elif dataset_name in ["CIFAR10"]:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


def load_train_datasets(args):
    train_dataset_dict = {
        "MNIST": datasets.MNIST(
            root=args.data_dir,
            train=True,
            download=True,
            transform=get_test_transform("MNIST")
        ),

        "EMNIST": datasets.EMNIST(
            root=args.data_dir,
            split="byclass",
            train=True,
            download=True,
            transform=get_test_transform("EMNIST")
        ),

        "CIFAR10": datasets.CIFAR10(
            root=args.data_dir,
            train=True,
            download=True,
            transform=get_test_transform("CIFAR10")
        )
    }

    return train_dataset_dict


def load_test_datasets(args):
    test_dataset_dict = {
        "MNIST": datasets.MNIST(
            root=args.data_dir,
            train=False,
            download=True,
            transform=get_test_transform("MNIST")
        ),

        "EMNIST": datasets.EMNIST(
            root=args.data_dir,
            split="byclass",
            train=False,
            download=True,
            transform=get_test_transform("EMNIST")
        ),

        "CIFAR10": datasets.CIFAR10(
            root=args.data_dir,
            train=False,
            download=True,
            transform=get_test_transform("CIFAR10")
        )
    }

    return test_dataset_dict


def load_split_files(args):
    split_paths = {
        "MNIST": args.mnist_split,
        "EMNIST": args.emnist_split,
        "CIFAR10": args.cifar10_split
    }

    split_dict = {}

    for dataset_name, path in split_paths.items():
        if path == "":
            continue

        with open(path, "r", encoding="utf-8") as f:
            split_dict[dataset_name] = json.load(f)

        print(f"[Loaded Split] {dataset_name}: {path}")

    return split_dict

def build_dataset_local_client_id_map(client_metadata):
    dataset_client_ids = {}

    for client_id, client_info in client_metadata.items():
        client_id = int(client_id)
        dataset_name = client_info["dataset_name"]

        if dataset_name not in dataset_client_ids:
            dataset_client_ids[dataset_name] = []

        dataset_client_ids[dataset_name].append(client_id)

    global_to_local = {}

    for dataset_name, client_ids in dataset_client_ids.items():
        client_ids = sorted(client_ids)

        for local_id, global_id in enumerate(client_ids):
            global_to_local[global_id] = local_id

    return global_to_local


def build_test_label_indices(test_dataset_dict):
    test_label_indices = {}

    for dataset_name, dataset in test_dataset_dict.items():
        targets = dataset.targets

        if not torch.is_tensor(targets):
            targets = torch.tensor(targets)

        test_label_indices[dataset_name] = {}

        for label_idx in torch.unique(targets).tolist():
            indices = torch.where(targets == label_idx)[0].tolist()
            test_label_indices[dataset_name][int(label_idx)] = indices

    return test_label_indices


def get_test_images(dataset_id, label_idx, args=None, test_dataset_dict=None, test_label_indices=None, client_dataset_names=None):
    dataset_name = client_dataset_names[dataset_id]
    dataset = test_dataset_dict[dataset_name]

    indices = test_label_indices[dataset_name].get(
        int(label_idx),
        []
    )

    if len(indices) == 0:
        raise ValueError(
            f"No test images found: "
            f"{dataset_id}, {dataset_name}, label={label_idx}"
        )

    sample_num = min(args.img_num_samples, len(indices))
    selected_indices = random.sample(indices, sample_num)

    images = []

    for index in selected_indices:
        image, _ = dataset[index]
        images.append(image)

    return torch.stack(images).to(args.device)


def build_client_model(model_path, d_name, label_space_meta, global_dim, args, device):
    arch_name = get_arch_name_from_path(model_path)
    num_classes = len(label_space_meta[d_name])

    if arch_name == "MLP":
        model = MLP(args.channels, num_classes, args.img_size, global_dim)
    elif arch_name == "CNN":
        model = CNN(args.channels, num_classes, global_dim)
    elif arch_name == "ResNet8":
        model = ResNet(BasicBlock, [1, 1, 1, 0], args.channels, num_classes, global_dim)
    elif arch_name == "ResNet18":
        model = ResNet(BasicBlock, [2, 2, 2, 2], args.channels, num_classes, global_dim)
    elif arch_name == "MobileNetV2":
        model = MobileNetV2(args.channels, num_classes, global_dim)
    elif arch_name == "MobileNetV3":
        model = MobileNetV3(args.channels, num_classes, global_dim)
    elif arch_name == "LeNet":
        model = LeNet(args.channels, num_classes, global_dim)
    elif arch_name == "AlexNet":
        model = AlexNet(args.channels, num_classes, global_dim)
    elif arch_name == "ShuffleNet":
        model = ShuffleNetV2(args.channels, num_classes, global_dim)
    elif arch_name == "SqueezeNet":
        model = SqueezeNet(args.channels, num_classes, global_dim)
    else:
        raise ValueError(f"Unknown arch_name: {arch_name} from {model_path}")

    return model.to(device)


def get_label_space(dataset_name, num_classes):
    dataset_name_lower = dataset_name.lower()

    if dataset_name_lower == "mnist":
        return [str(i) for i in range(num_classes)]

    elif dataset_name_lower == "emnist":
        return list(
            "0123456789"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
        )

    elif dataset_name_lower == "cifar10":
        return [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

    elif dataset_name_lower == "fashionmnist":
        return [
            "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]

    return [f"{dataset_name}_{i}" for i in range(num_classes)]


def load_global_gans(log_dir, global_round, group_label_space_meta, args, device):
    generators = {}
    group_dataset_names = {}

    gan_dir = os.path.join(log_dir, f"global_gans_round_{global_round}")

    if not os.path.exists(gan_dir):
        raise FileNotFoundError(f"Global GAN directory not found: {gan_dir}")

    gan_paths = sorted(
        glob.glob(os.path.join(gan_dir, "global_GAN_*.pth"))
    )

    for gan_path in gan_paths:
        checkpoint = safe_torch_load(gan_path, device)

        group_name = checkpoint["group_name"]
        datasets = checkpoint.get("datasets", [])

        if group_name not in group_label_space_meta:
            print(f"[Skip GAN] {group_name} not found in group_label_space_meta")
            continue

        if len(datasets) != 1:
            raise ValueError(
                f"{group_name} contains multiple datasets: {datasets}"
            )

        labels = group_label_space_meta[group_name]

        gen = DCGANGenerator(
            num_classes=len(labels),
            noise_dim=args.gen_noise_dim,
            img_size=args.img_size,
            channels=args.channels
        ).to(device)

        gen.load_state_dict(checkpoint["generator"])
        gen.eval()

        generators[group_name] = gen
        group_dataset_names[group_name] = datasets[0]

        print(
            f"[Loaded Global GAN] {group_name} | "
            f"dataset={datasets[0]} | path={gan_path}"
        )

    return generators, group_dataset_names


def load_client_models(log_dir, global_round, group_label_space_meta, global_dim, args, device):
    group_clients_dict = {}
    client_valid_labels_dict = {}
    clients_dir = os.path.join(log_dir, f"clients_round_{global_round}_checkpoints")

    if not os.path.exists(clients_dir):
        raise FileNotFoundError(f"Client checkpoint directory not found: {clients_dir}")

    model_paths = sorted(glob.glob(os.path.join(clients_dir, "client_model_*_c*_*.pth")))

    for model_path in model_paths:
        checkpoint = safe_torch_load(model_path, device)

        group_name = checkpoint["group_name"]
        client_id = checkpoint["client_id"]

        if group_name  not in group_label_space_meta:
            print(f"[Skip Model] {group_name } not found in label_space_meta")
            continue

        model = build_client_model(
            model_path=model_path,
            d_name=group_name,
            label_space_meta=group_label_space_meta,
            global_dim=global_dim,
            args=args,
            device=device
        )

        model.load_state_dict(checkpoint["model"])
        model.eval()
        valid_labels = checkpoint.get("valid_labels", list(range(len(group_label_space_meta[group_name]))))
        model.client_id = client_id
        model.valid_labels = valid_labels

        if group_name not in group_clients_dict:
            group_clients_dict[group_name] = []
            client_valid_labels_dict[group_name] = []

        group_clients_dict[group_name].append(model)
        client_valid_labels_dict[group_name].append(checkpoint.get("valid_labels", list(range(len(group_label_space_meta[group_name])))))

        print(f"[Loaded Client] client_{client_id} -> " f"{group_name}: {os.path.basename(model_path)}")

    return group_clients_dict, client_valid_labels_dict


def save_mapping_summary(path, mapping, label_space_meta, threshold_name, threshold):
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n" + "#" * 50 + "\n")
        f.write(f"{threshold_name} = {threshold}\n")
        f.write("#" * 50 + "\n")

        with redirect_stdout(f):
            global_to_local_mapping(
                mapping,
                logger=None,
                label_space_meta=label_space_meta
            )

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


def get_target_dataset(dataset):
    dataset_map = {"all": None, "mnist": "MNIST", "emnist": "EMNIST", "cifar10": "CIFAR10"}
    return dataset_map[dataset]


def get_output_folder_name(args):
    mapping_name_map = {"image-bi": "image-bi", "image-single": "single-direct", "missing_link": "missing_link", "improve_single": "improve_single", "image-cs": "image-cs"}
    mapping_name = mapping_name_map.get(args.label_mapping, args.label_mapping)
    dataset_name = args.dataset.replace("-", "")
    dataset_suffix = "" if dataset_name == "all" else f"_{dataset_name}"
    source_suffix = "" if args.image_source == "gan" else f"({args.image_source})"
    return f"{mapping_name}{dataset_suffix}_seed{args.seed}{source_suffix}"


def get_unique_output_dir(output_dir, folder_name):
    output_path = os.path.join(output_dir, folder_name)

    if not os.path.exists(output_path):
        return output_path

    index = 1
    while os.path.exists(os.path.join(output_dir, f"{folder_name}({index})")):
        index += 1

    return os.path.join(output_dir, f"{folder_name}({index})")


def train_dfkd_fl_generators(clients_dict, label_space_meta, args, device, logger, global_round):
    generators = {}
    save_dir = os.path.join(args.output_dir, f"dfkd_FL_generators_round_{global_round}")
    os.makedirs(save_dir, exist_ok=True)

    for d_name, client_models in clients_dict.items():
        num_classes = len(label_space_meta[d_name])
        save_path = os.path.join(save_dir, f"DFKD_FL_{d_name}.pth")

        logger.log(f"[DFKD_FL] Training {d_name} | num_classes={num_classes} | clients={len(client_models)}")

        generator = ConditionalImageGenerator(num_classes=num_classes, noise_dim=args.gen_noise_dim, img_channels=args.channels, img_size=args.img_size).to(device)
        generator_optimizer = torch.optim.Adam(generator.parameters(), lr=di_weight['lr'], betas=[0.5, 0.99])

        model_training_modes = {}
        for model in client_models:
            model_training_modes[id(model)] = model.training
            model.eval()
            for p in model.parameters():
                p.requires_grad = False

        batch_size = 64
        diversity_loss_fn = Gen_DiversityLoss(metric='l1').to(args.device)

        for epoch in tqdm(range(di_weight['epochs']), colour='blue', ncols=100, desc=f"DFKD_FL-{d_name}"):
            generator.train()
            epoch_loss = 0.0

            for it in tqdm(range(di_weight['g_steps']), leave=False):
                batch_labels = torch.randint(0, num_classes, (batch_size,), dtype=torch.long, device=device)
                z = torch.randn(batch_size, args.gen_noise_dim, device=device)
                gen_imgs = generator(z, batch_labels)

                generator_optimizer.zero_grad()

                loss_ce = 0.0
                for client_model in client_models:
                    preds = client_model(gen_imgs)
                    logits = preds[1] if isinstance(preds, tuple) else preds
                    loss_ce += F.cross_entropy(logits, batch_labels)

                loss_ce = loss_ce / len(client_models)

                flat_gen_imgs = gen_imgs.view(gen_imgs.size(0), -1)
                div_loss = diversity_loss_fn(flat_gen_imgs, z)
                loss = div_loss + loss_ce

                loss.backward()
                generator_optimizer.step()

                epoch_loss += loss.item()

            logger.log(f"[DFKD_FL] {d_name} | Epoch {epoch + 1}/{di_weight['epochs']} | Loss={epoch_loss / di_weight['g_steps']:.4f}")

        for model in client_models:
            for p in model.parameters():
                p.requires_grad = True
            model.train(model_training_modes[id(model)])

        generator.eval()
        torch.save({"generator": generator.state_dict(), "group_name": d_name, "num_classes": num_classes}, save_path)
        logger.log(f"Save {d_name} | {save_path}")

        generators[d_name] = generator

    return generators


def train_deepinversion_generators(clients_dict, label_space_meta, args, device, logger, global_round):
    generators = {}
    save_dir = os.path.join(args.log_dir, f"dfkd_generators_round_{global_round}")
    os.makedirs(save_dir, exist_ok=True)

    for d_name, client_models in clients_dict.items():
        num_classes = len(label_space_meta[d_name])
        save_path = os.path.join(save_dir, f"DFKD_{d_name}.pth")

        logger.log(f"[DFKD] Training {d_name} | num_classes={num_classes} | clients={len(client_models)}")

        generator = ConditionalImageGenerator(num_classes=num_classes, noise_dim=args.gen_noise_dim, img_channels=args.channels, img_size=args.img_size).to(device)
        generator_optimizer = torch.optim.Adam(generator.parameters(), lr=di_weight['lr'], betas=[0.5, 0.99])

        model_training_modes = {}
        for model in client_models:
            model_training_modes[id(model)] = model.training
            model.eval()
            for p in model.parameters():
                p.requires_grad = False

        bn_hooks_per_model = {}
        for model in client_models:
            hooks = []
            for module in model.modules():
                if hasattr(module, 'inplace'):
                    module.inplace = False
                if isinstance(module, nn.BatchNorm2d):
                    hooks.append(DeepInversionHook(module))
            bn_hooks_per_model[id(model)] = hooks

        epoch_loss_tracker = []
        batch_size = 64

        for epoch in tqdm(range(di_weight['epochs']), colour='blue', ncols=100, desc=f"DFKD-{d_name}"):
            generator.train()
            epoch_loss = 0.0
            valid_steps = 0

            for it in tqdm(range(di_weight['g_steps']), leave=False):
                batch_labels = torch.randint(0, num_classes, (batch_size,), dtype=torch.long, device=device)
                noise = torch.randn(batch_size, args.gen_noise_dim, device=device)
                inputs = generator(noise, batch_labels)

                generator_optimizer.zero_grad()
                batch_loss = 0.0
                clients_contributed = 0

                for client_model in client_models:
                    preds = client_model(inputs)
                    logits_t = preds[1] if isinstance(preds, tuple) else preds

                    cls_loss = F.cross_entropy(logits_t, batch_labels)

                    bn_loss = 0.0
                    hooks = bn_hooks_per_model[id(client_model)]
                    if hooks and di_weight['bn'] != 0:
                        bn_loss = sum(h.r_feature for h in hooks) / len(hooks)

                    tv_loss = get_image_prior_losses(inputs)
                    l2_loss = torch.norm(inputs, 2)

                    client_loss = di_weight['ce'] * cls_loss + di_weight['bn'] * bn_loss + di_weight['tv'] * tv_loss + di_weight['l2'] * l2_loss
                    batch_loss += client_loss
                    clients_contributed += 1

                if clients_contributed > 0:
                    avg_loss = batch_loss / clients_contributed
                    avg_loss.backward()
                    generator_optimizer.step()
                    epoch_loss += avg_loss.item()
                    valid_steps += 1

            avg_epoch_loss = epoch_loss / max(1, valid_steps)
            epoch_loss_tracker.append(avg_epoch_loss)
            logger.log(f"[DFKD] {d_name} | Epoch {epoch + 1}/{di_weight['epochs']} | Loss={avg_epoch_loss:.4f}")

        for hooks in bn_hooks_per_model.values():
            for h in hooks:
                if hasattr(h, 'close'):
                    h.close()
                elif hasattr(h, 'remove'):
                    h.remove()

        for model in client_models:
            for p in model.parameters():
                p.requires_grad = True
            model.train(model_training_modes[id(model)])

        generator.eval()
        torch.save({"generator": generator.state_dict(), "group_name": d_name, "num_classes": num_classes}, save_path)
        logger.log(f"Save {d_name} | {save_path}")

        generators[d_name] = generator

    return generators


def get_dfkd_images(dataset_id, label_idx, args=None, gen_dict=None, image_cache=None):
    key = (dataset_id, int(label_idx))

    if image_cache is not None and key in image_cache:
        return image_cache[key]

    generator = gen_dict[dataset_id]
    labels = torch.full((args.img_num_samples,), int(label_idx), dtype=torch.long, device=args.device)
    noise = torch.randn(args.img_num_samples, args.gen_noise_dim, device=args.device)

    generator.eval()
    with torch.no_grad():
        imgs = generator(noise, labels)

    if image_cache is not None:
        image_cache[key] = imgs

    return imgs


def run_offline(args):
    set_seed(args.seed)

    device = torch.device(args.device)
    rounds = parse_int_list(args.rounds)
    entropy_ratios = parse_float_list(args.entropy_ratios)
    target_dataset = get_target_dataset(args.dataset)

    os.makedirs(args.output_dir, exist_ok=True)

    run_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # run_output_dir = os.path.join(args.output_dir, run_time)
    # os.makedirs(run_output_dir, exist_ok=False)

    folder_name = get_output_folder_name(args)
    run_output_dir = get_unique_output_dir(args.output_dir, folder_name)
    os.makedirs(run_output_dir, exist_ok=False)

    logger = SimpleLogger(run_output_dir)

    csv_dir = os.path.join(run_output_dir, "label_mapping")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"offline_{args.label_mapping}_noniid_mapping_acc.csv")
    file_exists = os.path.isfile(csv_path)

    mapping_log_dir = os.path.join(run_output_dir, "mapping_summary")
    os.makedirs(mapping_log_dir, exist_ok=True)

    lm_utils.img_num_samples = args.img_num_samples
    lm_utils.feat_gen_noise_dim = args.gen_noise_dim

    test_dataset_dict = None
    test_label_indices = None

    train_dataset_dict = None
    split_dict = None

    if args.image_source == "raw":
        test_dataset_dict = load_test_datasets(args)
        test_label_indices = build_test_label_indices(test_dataset_dict)

        print("[Loaded Test Sets] MNIST, EMNIST, CIFAR10")

    for rnd in rounds:
        print("\n" + "=" * 60)
        print(f"Offline Non-IID Mapping | Global Round {rnd}")
        print("=" * 60)

        checkpoint = load_server_checkpoint(args.log_dir, rnd, device)
        label_space_meta = checkpoint["group_label_space_meta"]
        group_metadata = checkpoint["group_metadata"]
        client_metadata = checkpoint["client_metadata"]
        global_dim = checkpoint.get("global_feature_dim", 256)

        valid_labels_dict = {}
        for group_name, group_info in group_metadata.items():
            group_valid_labels = set()

            for client_id in group_info["client_ids"]:
                client_info = client_metadata.get(client_id)

                if client_info is None:
                    client_info = client_metadata.get(str(client_id))

                if client_info is None:
                    raise KeyError(
                        f"Client metadata not found: client_id={client_id}"
                    )

                group_valid_labels.update(
                    client_info["valid_labels"]
                )

            valid_labels_dict[group_name] = sorted(
                group_valid_labels
            )

        generators, group_dataset_names = load_global_gans(
            log_dir=args.log_dir,
            global_round=rnd,
            group_label_space_meta=label_space_meta,
            args=args,
            device=device
        )

        dataset_clients_dict, client_valid_labels_dict = load_client_models(
            log_dir=args.log_dir,
            global_round=rnd,
            group_label_space_meta=label_space_meta,
            global_dim=global_dim,
            args=args,
            device=device
        )

        active_datasets = []
        dataset_label_space_meta = {}

        for group_name in label_space_meta.keys():
            if target_dataset is not None and group_dataset_names.get(group_name) != target_dataset:
                continue

            if group_name in generators and group_name in dataset_clients_dict:
                active_datasets.append(group_name)
                dataset_label_space_meta[group_name] = label_space_meta[group_name]

        if len(active_datasets) == 0:
            print(f"[Skip Round {rnd}] No active clients.")
            continue

        print(f"Run Dataset: {args.dataset}")
        print(f"Active clusters: {active_datasets}")

        clear_image_caches()

        dfkd_image_cache = {}

        if args.image_source == "deepinversion":
            generators = train_deepinversion_generators(
                clients_dict=dataset_clients_dict,
                label_space_meta=dataset_label_space_meta,
                args=args,
                device=device,
                logger=logger,
                global_round=rnd
            )

        elif args.image_source == "dfkd_FL":
            generators = train_dfkd_fl_generators(
                clients_dict=dataset_clients_dict,
                label_space_meta=dataset_label_space_meta,
                args=args,
                device=device,
                logger=logger,
                global_round=rnd
            )

        if args.image_source == "gan":
            get_images_func = get_gen_images

            image_source_kwargs = {
                "args": args,
                "gen_dict": generators
            }

            mapping_get_images_func = get_images_func
            mapping_image_source_kwargs = image_source_kwargs

        elif args.image_source == "raw":
            get_images_func = get_test_images

            image_source_kwargs = {
                "args": args,
                "test_dataset_dict": test_dataset_dict,
                "test_label_indices": test_label_indices,
                "client_dataset_names": group_dataset_names
            }

        elif args.image_source in ["deepinversion", "dfkd_FL"]:
            get_images_func = get_dfkd_images
            image_source_kwargs = {"args": args, "gen_dict": generators, "image_cache": dfkd_image_cache}

        mapping_log_path = os.path.join(mapping_log_dir, f"global_round_{rnd}.log")

        if args.log_model_outputs and args.image_source == "gan":
            model_output_dir = os.path.join(run_output_dir, "local_model_outputs")
            os.makedirs(model_output_dir, exist_ok=True)

            model_output_path = os.path.join(
                model_output_dir,
                f"global_round_{rnd}.log"
            )


        mapping_log_path = os.path.join(
            mapping_log_dir,
            f"global_round_{rnd}.log"
        )

        if args.label_mapping in ["image-bi", "image-single", "improve_single", "improve_single_noniid", "temp", "temp2", "feature"]:
            thresholds = entropy_ratios
            threshold_name = "Entropy Ratio"
            threshold_col = "entropy_ratio"
        elif args.label_mapping in ["missing_link"]:
            thresholds = parse_float_list(args.missing_thresholds)
            threshold_name = "Missing Link Threshold"
            threshold_col = "missing_threshold"
        elif args.label_mapping == "image-cs":
            thresholds = parse_float_list(args.cs_thresholds)
            threshold_name = "Cosine Similarity Threshold"
            threshold_col = "cs_threshold"
        else:
            thresholds = [None]
            threshold_name = "Threshold"
            threshold_col = "threshold"

        total_time = 0.0

        for threshold in thresholds:
            print(f"\n[Run] Round {rnd} | {threshold_name} = {threshold}")

            single_start_time = time.time()

            mapping = improve_label_mapping_noniid(
                get_images_func=get_images_func,
                dataset_ids=active_datasets,
                clients_dict=dataset_clients_dict,
                label_space_meta=dataset_label_space_meta,
                entropy_ratio=threshold,
                logger=logger,
                **image_source_kwargs
            )

            global_to_local_mapping(
                mapping,
                logger=logger,
                label_space_meta=dataset_label_space_meta
            )

            single_time = time.time() - single_start_time
            total_time += single_time

            save_mapping_summary(
                path=mapping_log_path,
                mapping=mapping,
                label_space_meta=dataset_label_space_meta,
                threshold_name=threshold_name,
                threshold=threshold
            )

            metrics = evaluate_mapping_results(
                dataset_ids=active_datasets,
                label_space_meta=dataset_label_space_meta,
                local_id_to_global_id=mapping,
            )

            with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                if not file_exists:
                    writer.writerow(["global_round", threshold_col,
                        "recall", "specificity", "precision", "average_accuracy", "f1_score", "mcc",
                        "TP", "FP", "TN", "FN"])
                    file_exists = True

                writer.writerow([rnd, threshold,
                    metrics["Recall"], metrics["Specificity"], metrics["Precision"], metrics["AvgAccuracy"], metrics["F1-Score"], metrics["MCC"], 
                    metrics["TP"], metrics["FP"], metrics["TN"], metrics["FN"]])

        logger.log(f"\nAll Threshold Time: {args.label_mapping} | Round {rnd} | {total_time:.6f} seconds")
        logger.log(f"\nAverage Time: {args.label_mapping} | Round {rnd} | {total_time / len(entropy_ratios):.6f} seconds")

    print("\n" + "=" * 60)
    print(f"Saved results: {run_output_dir}")
    print(f"Saved CSV: {csv_path}")
    print("=" * 60)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, default="./logs/seed15698_iid_gan_weight/GeFL_gan_pacfl_iid")
    parser.add_argument("--data_dir", type=str, default="./data/raw")
    parser.add_argument("--output_dir", type=str, default="./label_mapping/pacfl_3cluster/diff_img")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--image_source", type=str, default="gan", choices=["gan", "raw", "deepinversion", "dfkd_FL"])
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "mnist", "emnist", "cifar10"])

    parser.add_argument("--mnist_split", type=str, default="")
    parser.add_argument("--emnist_split", type=str, default="")
    parser.add_argument("--cifar10_split", type=str, default="")

    # parser.add_argument("--rounds", type=str, default="5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100")
    parser.add_argument("--rounds", type=str, default="5,10,15,20,25")
    parser.add_argument("--label_mapping", type=str, default="improve_single", 
                        choices=["image-bi", "image-single", "missing_link", "improve_single", "image-cs", "improve_single_noniid", "temp", "temp2", "feature"])

    parser.add_argument("--entropy_ratios", type=str, default="0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0")
    parser.add_argument("--missing_thresholds", type=str, default="0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0")
    parser.add_argument("--fixed_entropy_ratio", type=float, default=0.1)
    parser.add_argument("--cs_thresholds", type=str, default="-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--use_new_entropy_method", action="store_true")
    parser.add_argument("--log_model_outputs", action="store_true")

    parser.add_argument("--gan_quality_threshold", type=float, default=0.5)

    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--img_num_samples", type=int, default=8)
    parser.add_argument("--gen_noise_dim", type=int, default=128)
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--channels", type=int, default=3)

    run_offline(parser.parse_args())