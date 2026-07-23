import os
import re
import math
import glob
import argparse
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
import json
import textwrap

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.nets import DCGANGenerator

def load_train_class_counts(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip() != ""]


def get_client_id_from_path(path):
    match = re.search(r"_c(\d+)", os.path.basename(path))
    if match is None:
        raise ValueError(f"Cannot parse client id from filename: {path}")
    return int(match.group(1))


def get_num_classes_from_generator_state(state_dict, noise_dim):
    first_weight = state_dict["net.0.weight"]
    in_dim = first_weight.shape[0]
    num_classes = in_dim - noise_dim
    return num_classes


def default_label_space(dataset_name, num_classes):
    dataset_name_lower = dataset_name.lower()

    if dataset_name_lower == "mnist":
        return [str(i) for i in range(num_classes)]

    if dataset_name_lower == "emnist":
        all_labels = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        return all_labels[:num_classes]

    if dataset_name_lower == "cifar10":
        labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        return labels[:num_classes]

    if dataset_name_lower == "fashionmnist":
        labels = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        return labels[:num_classes]

    return [f"{dataset_name}_{i}" for i in range(num_classes)]


def get_label_space(label_space_meta, dataset_name, client_name, num_classes):
    if isinstance(label_space_meta, dict):
        if client_name in label_space_meta and len(label_space_meta[client_name]) == num_classes:
            return label_space_meta[client_name]
        if dataset_name in label_space_meta and len(label_space_meta[dataset_name]) == num_classes:
            return label_space_meta[dataset_name]

    return default_label_space(dataset_name, num_classes)


def tensor_to_display_img(img_tensor):
    img = img_tensor.detach().cpu()
    img = (img + 1.0) / 2.0
    img = img.clamp(0, 1)

    if img.dim() == 3:
        img = img.permute(1, 2, 0).numpy()

    if img.shape[-1] == 1:
        img = img[:, :, 0]

    return img


def generate_one_image(generator, label_idx, noise_dim, device):
    generator.eval()
    z = torch.randn(1, noise_dim, device=device)
    y = torch.tensor([label_idx], dtype=torch.long, device=device)

    with torch.no_grad():
        x_gen = generator(z, y)[0]

    return tensor_to_display_img(x_gen)


def load_round_data(log_dir, global_round, noise_dim, img_size, channels, device, train_class_counts):
    server_ckpt_path = os.path.join(log_dir, f"server_checkpoints_{global_round}.pth")
    gan_dir = os.path.join(log_dir, f"local_gans_round_{global_round}")

    if not os.path.exists(server_ckpt_path):
        raise FileNotFoundError(f"Missing server checkpoint: {server_ckpt_path}")
    if not os.path.exists(gan_dir):
        raise FileNotFoundError(f"Missing GAN dir: {gan_dir}")

    server_ckpt = safe_torch_load(server_ckpt_path, device)
    client_label_distributions = server_ckpt.get("client_label_distributions", {})
    label_space_meta = server_ckpt.get("label_space_meta", {})

    gan_paths = sorted(glob.glob(os.path.join(gan_dir, "client_GAN_*_c*.pth")))

    dataset_clients = {}

    for gan_path in gan_paths:
        gan_ckpt = safe_torch_load(gan_path, device)

        client_id = gan_ckpt.get("client_id", get_client_id_from_path(gan_path))
        dataset_name = gan_ckpt["dataset_name"]
        client_name = f"client_{client_id}"
        generator_state = gan_ckpt["generator"]

        if client_id not in client_label_distributions:
            print(f"[Skip] {client_name} not found in client_label_distributions")
            continue

        valid_labels = sorted(client_label_distributions[client_id])
        num_classes = get_num_classes_from_generator_state(generator_state, noise_dim)
        class_names = get_label_space(label_space_meta, dataset_name, client_name, num_classes)

        client_train_counts = train_class_counts.get(dataset_name, {}).get(client_name, {})

        gen = DCGANGenerator(
            num_classes=num_classes,
            noise_dim=noise_dim,
            img_size=img_size,
            channels=channels,
        ).to(device)
        gen.load_state_dict(generator_state)
        gen.eval()

        client_info = {
            "client_id": client_id,
            "client_name": client_name,
            "dataset_name": dataset_name,
            "generator": gen,
            "valid_labels": valid_labels,
            "class_names": class_names,
            "num_classes": num_classes,
            "train_class_counts": client_train_counts,
        }

        if dataset_name not in dataset_clients:
            dataset_clients[dataset_name] = []

        dataset_clients[dataset_name].append(client_info)

    for dataset_name in dataset_clients:
        dataset_clients[dataset_name] = sorted(dataset_clients[dataset_name], key=lambda x: x["client_id"])

    return dataset_clients


def draw_dataset_sheet(dataset_name, client_infos, global_round, output_path, noise_dim, device, classes_per_row=8):
    total_rows = 0
    client_block_rows = []
    height_ratios = []

    for info in client_infos:
        block_rows = max(1, math.ceil(len(info["valid_labels"]) / classes_per_row))
        client_block_rows.append(block_rows)
        total_rows += block_rows + 1
        height_ratios.append(0.9)
        height_ratios.extend([2.4] * block_rows)

    fig = plt.figure(figsize=(2.1 * classes_per_row, sum(height_ratios) + 1.5))
    grid = fig.add_gridspec(total_rows, classes_per_row, height_ratios=height_ratios, hspace=0.65, wspace=0.25)
    fig.suptitle(f"Global Round {global_round} | Dataset: {dataset_name}", fontsize=18, fontweight="bold", y=0.998)

    current_row = 0

    for info, block_rows in zip(client_infos, client_block_rows):
        client_name = info["client_name"]
        valid_labels = info["valid_labels"]
        class_names = info["class_names"]
        generator = info["generator"]
        train_counts = info["train_class_counts"]

        valid_label_text = ", ".join([f"{label_idx}:{class_names[label_idx]}" for label_idx in valid_labels])
        valid_label_text = textwrap.fill(valid_label_text, width=160)
        total_train = sum(int(v) for v in train_counts.values())

        header_ax = fig.add_subplot(grid[current_row, :])
        header_ax.axis("off")
        header_ax.text(0.0, 0.95, f"{client_name} | Total Train: {total_train} | Valid labels ({len(valid_labels)}):\n{valid_label_text}", ha="left", va="top", fontsize=10, fontweight="bold")
        current_row += 1

        for block_row in range(block_rows):
            start_idx = block_row * classes_per_row
            block_labels = valid_labels[start_idx:start_idx + classes_per_row]

            for col_idx in range(classes_per_row):
                ax = fig.add_subplot(grid[current_row, col_idx])
                ax.axis("off")

                if col_idx >= len(block_labels):
                    continue

                label_idx = block_labels[col_idx]
                train_count = train_counts.get(str(label_idx), 0)
                img = generate_one_image(generator, label_idx, noise_dim, device)

                ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
                ax.set_title(f"class {label_idx} | {class_names[label_idx]}\nTrain: {train_count}", fontsize=9)
                ax.axis("off")

            current_row += 1

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="./logs/round_5-45_label_noniid_gan_weight/GeFL_DDPM_baseline_total_gan_noniid")
    parser.add_argument("--output_dir", type=str, default="./plot/Gen_img/gan_label_noniid")
    parser.add_argument("--train_counts_path", type=str, default="./data/raw/splits/mnist_emnist_cifar10_label_noniid_alpha0p1_seed15698.json")
    parser.add_argument("--rounds", type=str, default="5,10,15,20,25")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--gen_noise_dim", type=int, default=128)
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--classes_per_row", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)
    rounds = parse_int_list(args.rounds)
    os.makedirs(args.output_dir, exist_ok=True)
    train_counts = load_train_class_counts(args.train_counts_path)
    for rnd in rounds:
        print("=" * 60)
        print(f"Visualizing Global Round {rnd}")
        print("=" * 60)

        round_output_dir = os.path.join(args.output_dir, f"global_round_{rnd}")
        os.makedirs(round_output_dir, exist_ok=True)

        dataset_clients = load_round_data(
            log_dir=args.log_dir,
            global_round=rnd,
            noise_dim=args.gen_noise_dim,
            img_size=args.img_size,
            channels=args.channels,
            device=device, 
            train_class_counts=train_counts
        )

        for dataset_name, client_infos in dataset_clients.items():
            output_path = os.path.join(round_output_dir, f"{dataset_name}_clients.png")
            draw_dataset_sheet(
                dataset_name=dataset_name,
                client_infos=client_infos,
                global_round=rnd,
                output_path=output_path,
                noise_dim=args.gen_noise_dim,
                device=device,
                classes_per_row=args.classes_per_row
            )


if __name__ == "__main__":
    main()