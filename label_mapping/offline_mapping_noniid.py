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
import math

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.nets import (
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

import label_mapping_utils as lm_utils


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
    source_suffix = "(raw)" if args.image_source == "testset" else ""
    return f"{mapping_name}{dataset_suffix}_seed{args.seed}{source_suffix}"


def get_unique_output_dir(output_dir, folder_name):
    output_path = os.path.join(output_dir, folder_name)

    if not os.path.exists(output_path):
        return output_path

    index = 1
    while os.path.exists(os.path.join(output_dir, f"{folder_name}({index})")):
        index += 1

    return os.path.join(output_dir, f"{folder_name}({index})")


def log_local_model_outputs(get_images_func, dataset_ids, clients_dict, label_space_meta, group_dataset_names, valid_labels_dict, save_path, **get_images_kwargs):
    dataset_pairs = []

    for i in range(len(dataset_ids)):
        for j in range(i + 1, len(dataset_ids)):
            d1 = dataset_ids[i]
            d2 = dataset_ids[j]

            if len(label_space_meta[d1]) <= len(label_space_meta[d2]):
                dataset_pairs.append((d1, d2))
            else:
                dataset_pairs.append((d2, d1))

    images_cache = {}

    with open(save_path, "w", encoding="utf-8") as f:
        for src_id, tgt_id in dataset_pairs:
            src_name = group_dataset_names.get(src_id, src_id)
            tgt_name = group_dataset_names.get(tgt_id, tgt_id)

            src_names = label_space_meta[src_id]
            tgt_names = label_space_meta[tgt_id]
            tgt_clients = clients_dict.get(tgt_id, [])

            for l_idx, l_name in enumerate(src_names):
                if valid_labels_dict is not None and src_id in valid_labels_dict:
                    if l_idx not in valid_labels_dict[src_id]:
                        continue

                key = (src_id, l_idx)

                if key not in images_cache:
                    imgs_src = get_images_func(src_id, l_idx, **get_images_kwargs) if get_images_kwargs else get_images_func(src_id, l_idx)

                    if imgs_src is None:
                        continue

                    images_cache[key] = imgs_src

                imgs_src = images_cache[key]

                for model_idx, model in enumerate(tgt_clients):
                    device = next(model.parameters()).device

                    with torch.no_grad():
                        output = model(imgs_src.to(device))
                        logits = output[-1]
                        probs = torch.softmax(logits, dim=1)
                        mean_probs = probs.mean(dim=0)

                    entropy = -(mean_probs * torch.log(mean_probs + 1e-12)).sum().item()
                    pred_idx = torch.argmax(mean_probs).item()

                    client_id = getattr(model, "client_id", model_idx)
                    train_labels = getattr(model, "valid_labels", [])
                    pred_name = tgt_names[pred_idx]

                    train_label_text = ", ".join(
                        f"{label_idx}({tgt_names[label_idx]})"
                        for label_idx in train_labels
                    )

                    distribution = ", ".join(
                        f"{tgt_names[i]}:{mean_probs[i].item():.10f}"
                        for i in range(len(tgt_names))
                    )

                    # soft_label = ", ".join(
                    #     f"{p.item():.4f}" for p in mean_probs
                    # )

                    f.write(
                        f"{src_name} label {l_idx} ({l_name}) -> "
                        f"{tgt_name} client_{client_id}\n"
                    )
                    f.write(f"training labels = [{train_label_text}]\n")
                    f.write(f"distribution = {{{distribution}}}\n")
                    # f.write(f"soft_label = [{soft_label}]\n")
                    f.write(f"result = {pred_idx} ({pred_name})\n")
                    f.write(f"entropy = {entropy:.6f}\n\n")


def log_emnist_model_outputs(get_images_func, dataset_ids, clients_dict, label_space_meta, group_dataset_names, client_metadata, entropy_ratios, save_path, **get_images_kwargs):
    emnist_id = None
    for d_id in dataset_ids:
        if group_dataset_names.get(d_id) == "EMNIST":
            emnist_id = d_id
            break

    if emnist_id is None:
        return

    emnist_names = label_space_meta[emnist_id]
    emnist_clients = clients_dict.get(emnist_id, [])
    images_cache = {}
    client_ground_truth = {}
    client_detected_labels = {}

    with open(save_path, "w", encoding="utf-8") as f:
        for model_idx, model in enumerate(emnist_clients):
            client_id = model.client_id
            client_info = client_metadata.get(client_id)

            if client_info is None:
                client_info = client_metadata.get(str(client_id), {})

            ground_truth_labels = sorted(client_info.get("valid_labels", []))
            client_ground_truth[client_id] = ground_truth_labels
            detected_labels = []
            emnist_results = []

            for l_idx, l_name in enumerate(emnist_names):
                key = (emnist_id, l_idx)

                if key not in images_cache:
                    imgs_src = get_images_func(emnist_id, l_idx, **get_images_kwargs) if get_images_kwargs else get_images_func(emnist_id, l_idx)
                    if imgs_src is None:
                        continue
                    images_cache[key] = imgs_src

                imgs_src = images_cache[key]
                device = next(model.parameters()).device

                with torch.no_grad():
                    output = model(imgs_src.to(device))
                    logits = output[-1]
                    probs = torch.softmax(logits, dim=1)
                    mean_probs = probs.mean(dim=0)

                entropy = -(mean_probs * torch.log(mean_probs + 1e-12)).sum().item()
                pred_idx = torch.argmax(mean_probs).item()
                required_ratio = entropy / math.log(len(emnist_names))
                first_pass_threshold = next((threshold for threshold in entropy_ratios if entropy <= math.log(len(emnist_names)) * threshold), None)

                if pred_idx == l_idx:
                    detected_labels.append(l_idx)

                emnist_results.append((l_idx, l_name, pred_idx, entropy, required_ratio, first_pass_threshold, mean_probs))

            client_detected_labels[client_id] = detected_labels

            detected_text = ", ".join(f"{idx}({emnist_names[idx]})" for idx in detected_labels)
            ground_truth_text = ", ".join(f"{idx}({emnist_names[idx]})" for idx in ground_truth_labels)
            correct_labels = sorted(set(detected_labels) & set(ground_truth_labels))
            missed_labels = sorted(set(ground_truth_labels) - set(detected_labels))
            extra_labels = sorted(set(detected_labels) - set(ground_truth_labels))

            correct_text = ", ".join(f"{idx}({emnist_names[idx]})" for idx in correct_labels)
            missed_text = ", ".join(f"{idx}({emnist_names[idx]})" for idx in missed_labels)
            extra_text = ", ".join(f"{idx}({emnist_names[idx]})" for idx in extra_labels)

            f.write("=" * 100 + "\n")
            f.write(f"EMNIST client_{client_id}\n")
            f.write("=" * 100 + "\n")
            f.write(f"GAN detected training labels = [{detected_text}]\n")
            f.write(f"Ground truth training labels = [{ground_truth_text}]\n")
            f.write(f"Correct detected labels = [{correct_text}]\n")
            f.write(f"Missed labels = [{missed_text}]\n")
            f.write(f"Extra detected labels = [{extra_text}]\n\n")

            # f.write("[EMNIST Global GAN -> EMNIST Local Model]\n\n")

            # for l_idx, l_name, pred_idx, entropy, required_ratio, first_pass_threshold, mean_probs in emnist_results:
            #     pred_name = emnist_names[pred_idx]
            #     distribution = ", ".join(f"{emnist_names[i]}:{mean_probs[i].item():.10f}" for i in range(len(emnist_names)))
            #     pass_text = f"{first_pass_threshold:.2f}" if first_pass_threshold is not None else "> 1.00"

            #     f.write(f"EMNIST label {l_idx} ({l_name}) -> EMNIST client_{client_id}\n")
            #     f.write(f"distribution = {{{distribution}}}\n")
            #     f.write(f"result = {pred_idx} ({pred_name})\n")
            #     f.write(f"same label = {pred_idx == l_idx}\n")
            #     f.write(f"entropy = {entropy:.6f}\n")
            #     f.write(f"required entropy ratio = {required_ratio:.6f}\n")
            #     f.write(f"first pass threshold = {pass_text}\n\n")

        f.write("[MNIST / CIFAR10 Global GAN -> EMNIST Local Model]\n\n")

        for src_id in dataset_ids:
            src_name = group_dataset_names.get(src_id, src_id)

            if src_name not in ["MNIST", "CIFAR10"]:
                continue

            src_names = label_space_meta[src_id]

            for l_idx, l_name in enumerate(src_names):
                key = (src_id, l_idx)

                if key not in images_cache:
                    imgs_src = get_images_func(src_id, l_idx, **get_images_kwargs) if get_images_kwargs else get_images_func(src_id, l_idx)
                    if imgs_src is None:
                        continue
                    images_cache[key] = imgs_src

                imgs_src = images_cache[key]

                for model_idx, model in enumerate(emnist_clients):
                    client_id = model.client_id
                    device = next(model.parameters()).device
                    detected_labels = client_detected_labels.get(client_id, [])
                    detected_text = ", ".join(f"{idx}({emnist_names[idx]})" for idx in detected_labels)
                    ground_truth_labels = client_ground_truth.get(client_id, [])
                    ground_truth_text = ", ".join(f"{idx}({emnist_names[idx]})" for idx in ground_truth_labels)

                    with torch.no_grad():
                        output = model(imgs_src.to(device))
                        logits = output[-1]
                        probs = torch.softmax(logits, dim=1)
                        mean_probs = probs.mean(dim=0)

                    entropy = -(mean_probs * torch.log(mean_probs + 1e-12)).sum().item()
                    pred_idx = torch.argmax(mean_probs).item()
                    pred_name = emnist_names[pred_idx]

                    required_ratio = entropy / math.log(len(emnist_names))
                    first_pass_threshold = next((threshold for threshold in entropy_ratios if entropy <= math.log(len(emnist_names)) * threshold), None)
                    pass_text = f"{first_pass_threshold:.2f}" if first_pass_threshold is not None else "> 1.00"

                    distribution = ", ".join(f"{emnist_names[i]}:{mean_probs[i].item():.10f}" for i in range(len(emnist_names)))

                    f.write(f"{src_name} label {l_idx} ({l_name}) -> EMNIST client_{client_id}\n")
                    f.write(f"GAN detected labels = [{detected_text}]\n")
                    f.write(f"Ground truth labels = [{ground_truth_text}]\n")
                    f.write(f"distribution = {{{distribution}}}\n")
                    f.write(f"result = {pred_idx} ({pred_name})\n")
                    f.write(f"entropy = {entropy:.6f}\n")
                    f.write(f"required entropy ratio = {required_ratio:.6f}\n")
                    f.write(f"first pass threshold = {pass_text}\n\n")

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

    if args.image_source == "testset":
        test_dataset_dict = load_test_datasets(args)
        test_label_indices = build_test_label_indices(test_dataset_dict)

        print("[Loaded Test Sets] MNIST, EMNIST, CIFAR10")

    if args.label_mapping == "feature":
        train_dataset_dict = load_train_datasets(args)
        split_dict = load_split_files(args)

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

        get_images_func = get_gen_images

        image_source_kwargs = {
            "args": args,
            "gen_dict": generators
        }

        mapping_get_images_func = get_images_func
        mapping_image_source_kwargs = image_source_kwargs

        mapping_log_path = os.path.join(mapping_log_dir, f"global_round_{rnd}.log")

        if args.log_model_outputs and args.image_source == "gan":
            model_output_dir = os.path.join(run_output_dir, "local_model_outputs")
            os.makedirs(model_output_dir, exist_ok=True)

            model_output_path = os.path.join(model_output_dir, f"global_round_{rnd}.log")

            # log_local_model_outputs(
            #     get_images_func=get_images_func,
            #     dataset_ids=active_datasets,
            #     clients_dict=dataset_clients_dict,
            #     label_space_meta=dataset_label_space_meta,
            #     group_dataset_names=group_dataset_names,
            #     valid_labels_dict=valid_labels_dict,
            #     save_path=model_output_path,
            #     **image_source_kwargs
            # )

            log_emnist_model_outputs(
                get_images_func=get_images_func,
                dataset_ids=active_datasets,
                clients_dict=dataset_clients_dict,
                label_space_meta=dataset_label_space_meta,
                group_dataset_names=group_dataset_names,
                client_metadata=client_metadata,
                entropy_ratios=entropy_ratios,
                save_path=model_output_path,
                **image_source_kwargs
            )

        mapping_log_path = os.path.join(
            mapping_log_dir,
            f"global_round_{rnd}.log"
        )

        thresholds = entropy_ratios
        threshold_name = "Entropy Ratio"
        threshold_col = "entropy_ratio"

        total_time = 0.0

        for threshold in thresholds:
            print(f"\n[Run] Round {rnd} | {threshold_name} = {threshold}")

            single_start_time = time.time()

            if args.label_mapping == "temp":
                mapping = improve_label_mapping_noniid_temp(
                    get_images_func=get_images_func,
                    dataset_ids=active_datasets,
                    clients_dict=dataset_clients_dict,
                    label_space_meta=dataset_label_space_meta,
                    entropy_ratio=threshold,
                    logger=logger,
                    valid_labels_dict=valid_labels_dict,
                    **image_source_kwargs
                )

            elif args.label_mapping == "temp2":
                mapping = improve_label_mapping_noniid_temp_2(
                    get_images_func=get_images_func,
                    dataset_ids=active_datasets,
                    clients_dict=dataset_clients_dict,
                    label_space_meta=dataset_label_space_meta,
                    entropy_ratio=threshold,
                    logger=logger,
                    valid_labels_dict=valid_labels_dict,
                    client_valid_labels_dict=client_valid_labels_dict,
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
                valid_labels_dict=valid_labels_dict
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

    #parser.add_argument("--log_dir", type=str, default="./logs/2026-08-10_10-24-36/GeFL_gan_pacfl_iid")
    parser.add_argument("--log_dir", type=str, default="./logs/seed15698_noniid_gan_weight/GeFL_gan_pacfl_iid")
    parser.add_argument("--data_dir", type=str, default="./data/raw")
    parser.add_argument("--output_dir", type=str, default="./label_mapping/pacfl_3cluster_test/iid")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--image_source", type=str, default="gan", choices=["gan", "testset"])
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "mnist", "emnist", "cifar10"])

    parser.add_argument("--mnist_split", type=str, default="")
    parser.add_argument("--emnist_split", type=str, default="")
    parser.add_argument("--cifar10_split", type=str, default="")

    # parser.add_argument("--rounds", type=str, default="5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100")
    parser.add_argument("--rounds", type=str, default="5,10,15,20,25")
    parser.add_argument("--label_mapping", type=str, default="temp", 
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