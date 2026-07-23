"""
Offline SLAM-DUNK label mapping from saved global GANs.

Usage example:
python offline_slam_dunk_mapping.py \
  --log_dir ./logs/your_exp_dir \
  --device cuda:0 \
  --slam_epochs 30 \
  --slam_lr 5e-4 \
  --slam_lambda 0.5 \
  --slam_relation_threshold 0.005 \
  --slam_standalone_margin 2.0 \
  --slam_max_relation_order 2
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import torch
from types import SimpleNamespace
import random
import numpy as np

from utils.nets import DCGANGenerator
from slam_dunk import slam_dunk_label_mapping


class OfflineLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir

    def log(self, msg):
        print(msg)


def dict_to_namespace(d):
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, v)
    return ns


def find_server_checkpoint(log_dir, checkpoint_name=None):
    if checkpoint_name is not None:
        path = checkpoint_name if os.path.isabs(checkpoint_name) else os.path.join(log_dir, checkpoint_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    candidates = []
    for fname in os.listdir(log_dir):
        if fname.startswith("server_") and fname.endswith(".pth"):
            candidates.append(os.path.join(log_dir, fname))

    if not candidates:
        raise FileNotFoundError(f"No server_*.pth checkpoint found under {log_dir}")

    candidates = sorted(candidates, key=os.path.getmtime, reverse=True)
    return candidates[0]


def load_saved_global_gans(log_dir, label_space_meta, exp_conf, device):
    gan_dir = os.path.join(log_dir, "global_gans")
    if not os.path.isdir(gan_dir):
        raise FileNotFoundError(f"global_gans folder not found: {gan_dir}")

    generators = {}
    noise_dim = exp_conf.get("gen_noise_dim", 128)
    img_size = exp_conf.get("img_size", 32)
    channels = exp_conf.get("channels", 3)

    for d_name, class_names in label_space_meta.items():
        gan_path = os.path.join(gan_dir, f"{d_name}_GAN.pth")
        if not os.path.exists(gan_path):
            print(f"[Warning] GAN checkpoint not found for {d_name}: {gan_path}")
            continue

        gen = DCGANGenerator(num_classes=len(class_names), noise_dim=noise_dim, img_size=img_size, channels=channels).to(device)
        ckpt = torch.load(gan_path, map_location=device)
        state_dict = ckpt["generator"] if "generator" in ckpt else ckpt
        gen.load_state_dict(state_dict)
        gen.eval()
        generators[d_name] = gen

    if not generators:
        raise RuntimeError("No global GANs loaded.")

    return generators


def get_saved_gan_images(d_id, l_idx, args=None, gen_dict=None):
    if gen_dict is None or d_id not in gen_dict:
        return None

    gen = gen_dict[d_id]
    device = next(gen.parameters()).device
    n = getattr(args, "slam_samples_per_label", 128)
    noise_dim = getattr(args, "gen_noise_dim", 128)

    z = torch.randn(n, noise_dim, device=device)
    y = torch.full((n,), l_idx, dtype=torch.long, device=device)

    with torch.no_grad():
        imgs = gen(z, y)

    return imgs.detach()


def save_mapping_summary(mapping, label_space_meta, save_path):
    global_to_local = {}

    for d_name, local_map in mapping.items():
        for local_id, global_id in local_map.items():
            global_to_local.setdefault(global_id, [])
            class_name = label_space_meta[d_name][local_id] if local_id < len(label_space_meta[d_name]) else str(local_id)
            global_to_local[global_id].append((d_name, local_id, class_name))

    lines = ["Label Mapping Summary:"]
    for gid in sorted(global_to_local.keys()):
        items = []
        for d_name, local_id, class_name in global_to_local[gid]:
            items.append(f"{d_name}: '{class_name}'")
        lines.append(f"Global ID = {gid:<3}| from: " + ", ".join(items))

    text = "\n".join(lines)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(text)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--checkpoint_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--slam_epochs", type=int, default=20)
    parser.add_argument("--slam_lr", type=float, default=1e-4)
    parser.add_argument("--slam_batch_size", type=int, default=64)
    parser.add_argument("--slam_samples_per_label", type=int, default=64)
    parser.add_argument("--slam_lambda", type=float, default=0.5)
    parser.add_argument("--slam_feat_dim", type=int, default=256)
    parser.add_argument("--slam_relation_threshold", type=float, default=0.0)
    parser.add_argument("--slam_standalone_margin", type=float, default=1.5)
    parser.add_argument("--slam_max_relation_order", type=int, default=2)
    parser.add_argument("--output_name", type=str, default="offline_slam_dunk_mapping.pth")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")
    logger = OfflineLogger(args.log_dir)

    server_ckpt_path = find_server_checkpoint(args.log_dir, args.checkpoint_name)
    logger.log(f"[Offline SLAM-DUNK] Loading server checkpoint: {server_ckpt_path}")

    checkpoint = torch.load(server_ckpt_path, map_location=device)
    label_space_meta = checkpoint["label_space_meta"]
    exp_conf = checkpoint.get("exp_conf", {})
    old_args = checkpoint.get("args", {})

    run_args = dict_to_namespace(old_args if isinstance(old_args, dict) else {})
    run_args.device = str(device)
    run_args.gen_noise_dim = exp_conf.get("gen_noise_dim", getattr(run_args, "gen_noise_dim", 128))
    run_args.slam_epochs = args.slam_epochs
    run_args.slam_lr = args.slam_lr
    run_args.slam_batch_size = args.slam_batch_size
    run_args.slam_samples_per_label = args.slam_samples_per_label
    run_args.slam_lambda = args.slam_lambda
    run_args.slam_feat_dim = args.slam_feat_dim
    run_args.slam_relation_threshold = args.slam_relation_threshold
    run_args.slam_standalone_margin = args.slam_standalone_margin
    run_args.slam_max_relation_order = args.slam_max_relation_order

    generators = load_saved_global_gans(args.log_dir, label_space_meta, exp_conf, device)
    active_datasets = list(generators.keys())
    dataset_label_space_meta = {d_name: label_space_meta[d_name] for d_name in active_datasets}

    logger.log(f"[Offline SLAM-DUNK] Active datasets: {active_datasets}")

    mapping = slam_dunk_label_mapping(
        get_images_func=get_saved_gan_images,
        dataset_ids=active_datasets,
        label_space_meta=dataset_label_space_meta,
        logger=logger,
        args=run_args,
        gen_dict=generators,
    )

    save_path = os.path.join(args.log_dir, args.output_name)
    summary_path = os.path.join(args.log_dir, args.output_name.replace(".pth", "_summary.txt"))

    result = {
        "local_id_to_global_id": mapping,
        "label_space_meta": dataset_label_space_meta,
        "slam_args": vars(args),
        "source_server_checkpoint": server_ckpt_path,
    }

    torch.save(result, save_path)
    save_mapping_summary(mapping, dataset_label_space_meta, summary_path)

    logger.log(f"[Offline SLAM-DUNK] Mapping saved to: {save_path}")
    logger.log(f"[Offline SLAM-DUNK] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
