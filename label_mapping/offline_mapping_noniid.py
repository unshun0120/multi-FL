import os
import sys
import re
import csv
import glob
import argparse
import torch
from datetime import datetime
from contextlib import redirect_stdout
import random
import numpy as np
from torchvision import datasets, transforms

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
    improve_label_mapping_label_noniid,
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


def load_local_gans(log_dir, global_round, label_space_meta, args, device):
    generators = {}
    client_label_space_meta = {}
    client_dataset_names = {}
    gan_dir = os.path.join(log_dir, f"local_gans_round_{global_round}")

    if not os.path.exists(gan_dir):
        raise FileNotFoundError(f"Local GAN directory not found: {gan_dir}")

    gan_paths = sorted(glob.glob(os.path.join(gan_dir, "client_GAN_*_c*.pth")))

    for gan_path in gan_paths:
        checkpoint = safe_torch_load(gan_path, device)
        client_id = checkpoint.get("client_id", get_client_id_from_path(gan_path))
        dataset_name = checkpoint["dataset_name"]
        d_name = f"client_{client_id}"
        client_dataset_names[d_name] = dataset_name

        generator_state_dict = checkpoint["generator"]

        dataset_num_classes = {
            "MNIST": 10,
            "EMNIST": 62,
            "CIFAR10": 10,
        }

        if dataset_name not in dataset_num_classes:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        num_classes = dataset_num_classes[dataset_name]

        client_label_space_meta[d_name] = get_label_space(dataset_name, num_classes)

        gen = DCGANGenerator(
            num_classes=len(client_label_space_meta[d_name]),
            noise_dim=args.gen_noise_dim,
            img_size=args.img_size,
            channels=args.channels
        ).to(device)

        gen.load_state_dict(checkpoint["generator"])
        gen.eval()
        generators[d_name] = gen
        print(f"[Loaded GAN] {d_name}: {gan_path}")

    return generators, client_label_space_meta, client_dataset_names


def load_client_models(log_dir, global_round, label_space_meta, global_dim, args, device):
    dataset_clients_dict = {}
    clients_dir = os.path.join(log_dir, f"clients_round_{global_round}_checkpoints")

    if not os.path.exists(clients_dir):
        raise FileNotFoundError(f"Client checkpoint directory not found: {clients_dir}")

    model_paths = sorted(glob.glob(os.path.join(clients_dir, "client_model_*_c*_*.pth")))

    for model_path in model_paths:
        client_id = get_client_id_from_path(model_path)
        d_name = f"client_{client_id}"

        if d_name not in label_space_meta:
            print(f"[Skip Model] {d_name} not found in label_space_meta")
            continue

        model = build_client_model(
            model_path=model_path,
            d_name=d_name,
            label_space_meta=label_space_meta,
            global_dim=global_dim,
            args=args,
            device=device
        )

        state_dict = safe_torch_load(model_path, device)
        model.load_state_dict(state_dict)
        model.eval()

        dataset_clients_dict[d_name] = [model]
        print(f"[Loaded Client] {d_name}: {os.path.basename(model_path)}")

    return dataset_clients_dict


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
    return f"{mapping_name}{dataset_suffix}{source_suffix}"


def get_unique_output_dir(output_dir, folder_name):
    output_path = os.path.join(output_dir, folder_name)

    if not os.path.exists(output_path):
        return output_path

    index = 1
    while os.path.exists(os.path.join(output_dir, f"{folder_name}({index})")):
        index += 1

    return os.path.join(output_dir, f"{folder_name}({index})")


def run_offline(args):
    set_seed(args.seed)

    device = torch.device(args.device)
    rounds = parse_int_list(args.rounds)
    entropy_ratios = parse_float_list(args.entropy_ratios)
    target_dataset = get_target_dataset(args.dataset)

    os.makedirs(args.output_dir, exist_ok=True)

    # run_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
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

    if args.image_source == "testset":
        test_dataset_dict = load_test_datasets(args)
        test_label_indices = build_test_label_indices(
            test_dataset_dict
        )

        print("[Loaded Test Sets] MNIST, EMNIST, CIFAR10")

    for rnd in rounds:
        print("\n" + "=" * 60)
        print(f"Offline Non-IID Mapping | Global Round {rnd}")
        print("=" * 60)

        checkpoint = load_server_checkpoint(args.log_dir, rnd, device)
        label_space_meta = checkpoint["label_space_meta"]
        global_dim = checkpoint.get("global_feature_dim", 256)
        client_label_distributions = checkpoint["client_label_distributions"]

        valid_labels_dict = {}
        for client_id, labels in client_label_distributions.items():
            valid_labels_dict[f"client_{int(client_id)}"] = sorted(labels)

        generators, client_label_space_meta, client_dataset_names = load_local_gans(
            log_dir=args.log_dir,
            global_round=rnd,
            label_space_meta=label_space_meta,
            args=args,
            device=device
        )

        dataset_clients_dict = load_client_models(
            log_dir=args.log_dir,
            global_round=rnd,
            label_space_meta=client_label_space_meta,
            global_dim=global_dim,
            args=args,
            device=device
        )

        active_datasets = []
        dataset_label_space_meta = {}

        for d_name in client_label_space_meta.keys():
            if target_dataset is not None and client_dataset_names.get(d_name) != target_dataset:
                continue

            if d_name in generators and d_name in dataset_clients_dict:
                active_datasets.append(d_name)
                dataset_label_space_meta[d_name] = client_label_space_meta[d_name]

        if len(active_datasets) == 0:
            print(f"[Skip Round {rnd}] No active clients.")
            continue

        print(f"Run Dataset: {args.dataset}")
        print(f"Active clients: {active_datasets}")

        clear_image_caches()

        if args.image_source == "local_gan":
            get_images_func = get_gen_images

            image_source_kwargs = {
                "args": args,
                "gen_dict": generators
            }

        elif args.image_source == "testset":
            get_images_func = get_test_images

            image_source_kwargs = {
                "args": args,
                "test_dataset_dict": test_dataset_dict,
                "test_label_indices": test_label_indices,
                "client_dataset_names": client_dataset_names
            }

        else:
            raise ValueError(
                f"Unknown image source: {args.image_source}"
            )

        mapping_log_path = os.path.join(mapping_log_dir, f"global_round_{rnd}.log")

        if args.label_mapping in ["image-bi", "image-single", "improve_single"]:
            thresholds = entropy_ratios
            threshold_name = "Entropy Ratio"
            threshold_col = "entropy_ratio"
        elif args.label_mapping in ["missing_link", "improve_single_label_noniid"]:
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

        for threshold in thresholds:
            print(f"\n[Run] Round {rnd} | {threshold_name} = {threshold}")

            if args.label_mapping == "image-bi":
                mapping = label_mapping(
                    get_images_func=get_images_func,
                    dataset_ids=active_datasets,
                    clients_dict=dataset_clients_dict,
                    label_space_meta=dataset_label_space_meta,
                    entropy_ratio=threshold,
                    use_new_entropy_method=args.use_new_entropy_method,
                    logger=logger,
                    **image_source_kwargs,
                    valid_labels_dict=valid_labels_dict
                )

            elif args.label_mapping == "image-single":
                mapping = single_direction_label_mapping(
                    get_images_func=get_images_func,
                    dataset_ids=active_datasets,
                    clients_dict=dataset_clients_dict,
                    label_space_meta=dataset_label_space_meta,
                    entropy_ratio=threshold,
                    use_new_entropy_method=args.use_new_entropy_method,
                    logger=logger,
                    **image_source_kwargs,
                    valid_labels_dict=valid_labels_dict
                )

            elif args.label_mapping == "missing_link":
                mapping = missing_link_label_mapping(
                    get_images_func=get_images_func,
                    dataset_ids=active_datasets,
                    clients_dict=dataset_clients_dict,
                    label_space_meta=dataset_label_space_meta,
                    missing_threshold=threshold,
                    logger=logger,
                    valid_labels_dict=valid_labels_dict,
                    **image_source_kwargs
                )

            elif args.label_mapping == "improve_single":
                mapping = improve_label_mapping(
                    get_images_func=get_images_func,
                    dataset_ids=active_datasets,
                    clients_dict=dataset_clients_dict,
                    label_space_meta=dataset_label_space_meta,
                    entropy_ratio=threshold,
                    use_new_entropy_method=True,
                    logger=logger,
                    valid_labels_dict=valid_labels_dict,
                    **image_source_kwargs
                )

            elif args.label_mapping == "improve_single_label_noniid":
                mapping = improve_label_mapping_label_noniid(
                    get_images_func=get_images_func,
                    dataset_ids=active_datasets,
                    clients_dict=dataset_clients_dict,
                    label_space_meta=dataset_label_space_meta,
                    entropy_ratio=args.fixed_entropy_ratio,
                    missing_threshold=threshold,
                    logger=logger,
                    valid_labels_dict=valid_labels_dict,
                    **image_source_kwargs
                )

            global_to_local_mapping(
                mapping,
                logger=logger,
                label_space_meta=dataset_label_space_meta
            )

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
                    writer.writerow([
                        "global_round",
                        threshold_col,
                        "recall",
                        "specificity",
                        "precision",
                        "average_accuracy",
                        "f1_score",
                        "mcc",
                        "TP",
                        "FP",
                        "TN",
                        "FN"
                    ])
                    file_exists = True

                writer.writerow([
                    rnd,
                    threshold,
                    metrics["Recall"],
                    metrics["Specificity"],
                    metrics["Precision"],
                    metrics["AvgAccuracy"],
                    metrics["F1-Score"],
                    metrics["MCC"],
                    metrics["TP"],
                    metrics["FP"],
                    metrics["TN"],
                    metrics["FN"]
                ])

    print("\n" + "=" * 60)
    print(f"Saved results: {run_output_dir}")
    print(f"Saved CSV: {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, default="./logs/round_5-45_label_noniid_gan_weight/GeFL_DDPM_baseline_total_gan_noniid")
    parser.add_argument("--data_dir", type=str, default="./data/raw")
    parser.add_argument("--output_dir", type=str, default="./label_mapping/offline_noniid_results(noniid_label)")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--image_source", type=str, default="local_gan", choices=["local_gan", "testset"])
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "mnist", "emnist", "cifar10"])

    parser.add_argument("--rounds", type=str, default="5,10,15,20,25")
    parser.add_argument("--label_mapping", type=str, default="improve_single", choices=["image-bi", "image-single", "missing_link", "improve_single", "image-cs", "improve_single_label_noniid"])

    parser.add_argument("--entropy_ratios", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--missing_thresholds", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--fixed_entropy_ratio", type=float, default=0.1)
    parser.add_argument("--cs_thresholds", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--use_new_entropy_method", action="store_true")

    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--img_num_samples", type=int, default=8)
    parser.add_argument("--gen_noise_dim", type=int, default=128)
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--channels", type=int, default=3)

    run_offline(parser.parse_args())