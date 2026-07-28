import os
import sys
import re
import csv
import glob
import argparse
import torch
from contextlib import redirect_stdout
from datetime import datetime
import random
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.nets import DCGANGenerator, get_heterogeneous_model
from utils.nets import (
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
    evaluate_mapping_results,
    global_to_local_mapping,
    single_direction_label_mapping,
    improve_label_mapping_analyze,
)

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

    ckpt = safe_torch_load(ckpt_path, device)
    return ckpt


def load_global_gans(log_dir, global_round, label_space_meta, args, device):
    generators = {}
    gen_dir = os.path.join(log_dir, f"global_gans_{global_round}")

    if not os.path.exists(gen_dir):
        raise FileNotFoundError(f"Global GAN directory not found: {gen_dir}")

    for d_name, labels in label_space_meta.items():
        gan_path = os.path.join(gen_dir, f"{d_name}_GAN.pth")

        if not os.path.exists(gan_path):
            print(f"[Skip GAN] Not found: {gan_path}")
            continue

        gen = DCGANGenerator(
            num_classes=len(labels),
            noise_dim=args.gen_noise_dim,
            img_size=args.img_size,
            channels=args.channels
        ).to(device)

        checkpoint = safe_torch_load(gan_path, device)
        gen.load_state_dict(checkpoint["generator"])
        gen.eval()

        generators[d_name] = gen
        print(f"[Loaded GAN] {d_name}: {gan_path}")

    return generators


def get_arch_name_from_path(model_path):
    filename = os.path.basename(model_path)
    filename = filename.replace(".pth", "")

    parts = filename.split("_")

    if len(parts) < 5:
        raise ValueError(f"Cannot parse arch name from filename: {filename}")

    return parts[-1]


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


def load_client_models(log_dir, global_round, label_space_meta, global_dim, args, device):
    dataset_clients_dict = {}
    clients_dir = os.path.join(log_dir, f"clients_round_{global_round}_checkpoints")

    if not os.path.exists(clients_dir):
        raise FileNotFoundError(f"Client checkpoint directory not found: {clients_dir}")

    for d_name in label_space_meta.keys():
        pattern = os.path.join(clients_dir, f"client_model_{d_name}_c*_*.pth")
        model_paths = sorted(glob.glob(pattern))

        if len(model_paths) == 0:
            print(f"[Skip Models] No client models for {d_name}")
            continue

        dataset_clients_dict[d_name] = []

        for model_path in model_paths:
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

            dataset_clients_dict[d_name].append(model)

            print(f"[Loaded Client] {os.path.basename(model_path)} -> {model.__class__.__name__}")

        # print(f"[Loaded Clients] {d_name}: {len(dataset_clients_dict[d_name])} models")

    return dataset_clients_dict


def save_mapping_summary(path, mapping, label_space_meta, entropy_ratio, confuse_model_ratio, vote_mode):
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n" + "#" * 50 + "\n")
        f.write(f"Entropy Ratio = {entropy_ratio}\n")
        f.write(f"Confuse Model Ratio = {confuse_model_ratio}\n")
        f.write(f"Vote Mode = {vote_mode}\n")
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

def run_offline(args):
    set_seed(args.seed)

    device = torch.device(args.device)
    rounds = parse_int_list(args.rounds)
    entropy_ratios = parse_float_list(args.entropy_ratios)
    confuse_model_ratios = parse_float_list(args.confuse_model_ratios)
    vote_modes = [x.strip().lower() == "true" for x in args.vote_modes.split(",")]

    os.makedirs(args.output_dir, exist_ok=True)

    run_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_output_dir = os.path.join(args.output_dir, run_time)
    os.makedirs(run_output_dir, exist_ok=False)

    logger = SimpleLogger(run_output_dir)

    csv_dir = os.path.join(run_output_dir, "")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "offline_improve_mapping_acc.csv")
    file_exists = os.path.isfile(csv_path)

    lm_utils.img_num_samples = args.img_num_samples
    lm_utils.feat_gen_noise_dim = args.gen_noise_dim

    for rnd in rounds:
        print("\n" + "=" * 60)
        print(f"Offline Image-BI | Global Round {rnd}")
        print("=" * 60)

        ckpt = load_server_checkpoint(args.log_dir, rnd, device)
        label_space_meta = ckpt["label_space_meta"]
        global_dim = ckpt.get("global_feature_dim", 256)

        generators = load_global_gans(
            log_dir=args.log_dir,
            global_round=rnd,
            label_space_meta=label_space_meta,
            args=args,
            device=device
        )

        dataset_clients_dict = load_client_models(
            log_dir=args.log_dir,
            global_round=rnd,
            label_space_meta=label_space_meta,
            global_dim=global_dim,
            args=args,
            device=device
        )

        active_datasets = []
        dataset_label_space_meta = {}

        for d_name in label_space_meta.keys():
            if d_name in generators and d_name in dataset_clients_dict:
                active_datasets.append(d_name)
                dataset_label_space_meta[d_name] = label_space_meta[d_name]

        if len(active_datasets) == 0:
            print(f"[Skip Round {rnd}] No active datasets.")
            continue

        print(f"Active datasets: {active_datasets}")

        clear_image_caches()

        mapping_log_path = os.path.join(run_output_dir, f"round_{rnd}_improve_mapping_summary.log")

        for entropy_ratio in entropy_ratios:
            for confuse_model_ratio in confuse_model_ratios:
                for vote_mode in vote_modes:
                    print(f"\n[Run] Round {rnd} | Entropy ratio = {entropy_ratio}")

                    mapping = improve_label_mapping_analyze(
                        get_images_func=get_gen_images,
                        dataset_ids=active_datasets,
                        clients_dict=dataset_clients_dict,
                        label_space_meta=dataset_label_space_meta,
                        entropy_ratio=entropy_ratio,
                        confuse_model_ratio=confuse_model_ratio,
                        vote_mode=vote_mode,
                        use_new_entropy_method=args.use_new_entropy_method,
                        logger=logger,
                        args=args,
                        gen_dict=generators
                    )

                    save_mapping_summary(
                        path=mapping_log_path,
                        mapping=mapping,
                        label_space_meta=dataset_label_space_meta,
                        confuse_model_ratio=confuse_model_ratio,
                        vote_mode=vote_mode,
                        entropy_ratio=entropy_ratio
                    )

                    metrics = evaluate_mapping_results(
                        dataset_ids=active_datasets,
                        label_space_meta=dataset_label_space_meta,
                        local_id_to_global_id=mapping
                    )

                    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)

                        if not file_exists:
                            writer.writerow([
                                "global_round", "entropy_ratio", "confuse_model_ratio", "vote_mode",
                                "recall", "specificity", "precision", "average_accuracy", "f1_score", "mcc",
                                "TP", "FP", "TN", "FN"])
                            file_exists = True

                        writer.writerow([
                            rnd, entropy_ratio, confuse_model_ratio, vote_mode,
                            metrics["Recall"], metrics["Specificity"], metrics["Precision"], metrics["AvgAccuracy"], metrics["F1-Score"], metrics["MCC"],
                            metrics["TP"], metrics["FP"], metrics["TN"], metrics["FN"]])

                    print(
                        f"Round {rnd} | entropy ratio {entropy_ratio:.2f} | Confuse {confuse_model_ratio:.2f} | Vote {vote_mode} | "
                        f"Recall {metrics['Recall']:.4f} | "
                        f"Precision {metrics['Precision']:.4f} | "
                        f"F1 {metrics['F1-Score']:.4f} | "
                        f"MCC {metrics['MCC']:.4f}"
                    )

    print("\n" + "=" * 60)
    print(f"Saved CSV: {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, default="./logs/round_5-25_gan_weight/GeFL_DDPM_baseline_total_gan")
    parser.add_argument("--output_dir", type=str, default="./label_mapping/offline_ours_results_analyze")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    
    parser.add_argument("--rounds", type=str, default="5,10,15,20,25")
    # parser.add_argument("--entropy_ratios", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--entropy_ratios", type=str, default="0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0")
    parser.add_argument("--confuse_model_ratios", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--vote_modes", type=str, default="True, False")

    parser.add_argument("--use_new_entropy_method", action="store_true")

    parser.add_argument("--device", type=str, default="cuda:1")

    parser.add_argument("--img_num_samples", type=int, default=8)
    parser.add_argument("--gen_noise_dim", type=int, default=128)
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--channels", type=int, default=3)

    run_offline(parser.parse_args())