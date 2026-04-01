import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import random

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import Logger
from data.datasets import get_raw_dataset_transform, get_readable_class_names
from utils.nets import MLP, CNN, ResNet, BasicBlock, MobileNetV2, MobileNetV3, LeNet, AlexNet, ShuffleNetV2, SqueezeNet
from label_mapping_utils import (
    label_mapping, 
    evaluate_mapping_results, save_mapping_results_to_csv, 
    get_gen_images, get_real_images
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to models")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--seed", type=int, default=15698)
    parser.add_argument("--method", type=str, default="all", choices=['syn', 'real', 'all', 'DI', 'FAST', 'NAYER'], 
                        help="Choose experiment method: 'syn' (Synthetic), 'real' (Real Images), or 'all' (Both)")
    parser.add_argument("--gen_mode", type=str, default="all", choices=['old', 'new', 'all'], 
                        help="")
    args = parser.parse_args()

    logger = Logger(args)
    logger.log(f"Loading checkpoints from {args.model_path}")
    logger.log(f"Method selected: {args.method}")
    save_dir = logger.log_dir

    set_seed(args.seed)
    
    DATASET_META = { 
        'MNIST':  {'in_ch': 3, 'classes': 10,  'size': 32}, 
        'EMNIST': {'in_ch': 3, 'classes': 62,  'size': 32},
        'CIFAR10':  {'in_ch': 3, 'classes': 10,  'size': 32}, 
        'FashionMNIST': {'in_ch': 3, 'classes': 10,  'size': 32},
        'CIFAR100':     {'in_ch': 3, 'classes': 100, 'size': 32},
    }

    DATA_ROOT = './data/raw'

    label_space_meta = {}
    for d_name in DATASET_META.keys():
        label_space_meta[d_name] = get_readable_class_names(d_name, root=DATA_ROOT)

    clients_dict = {}
    for d_name in DATASET_META.keys():
        clients_dict[d_name] = [] 
        meta = DATASET_META[d_name]

        ckpt_folder = os.path.join(args.model_path, "clients_last_round_checkpoints")

        if not os.path.exists(ckpt_folder):
            logger.log(f"Path {ckpt_folder} doesn't exist.")
            continue

        for filename in os.listdir(ckpt_folder):
            if filename.startswith(f"client_model_{d_name}_c") and filename.endswith(".pth"):
                ckpt_path = os.path.join(ckpt_folder, filename)

                arch_name = filename.replace(".pth", "").split("_")[-1]

                if arch_name == 'MLP':
                    model = MLP(in_channels=meta['in_ch'], num_classes=meta['classes'], img_size=meta['size']).to(args.device)
                elif arch_name == 'CNN':
                    model = CNN(in_channels=meta['in_ch'], num_classes=meta['classes']).to(args.device)
                elif arch_name == 'ResNet8':
                    model = ResNet(BasicBlock, [1, 1, 1, 0], in_channels=meta['in_ch'], num_classes=meta['classes']).to(args.device)
                elif arch_name == 'ResNet18':
                    model = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=meta['in_ch'], num_classes=meta['classes']).to(args.device)
                elif arch_name == 'MobileNetV2':
                    model = MobileNetV2(in_channels=meta['in_ch'], num_classes=meta['classes']).to(args.device)
                elif arch_name == 'MobileNetV3':
                    model = MobileNetV3(in_channels=meta['in_ch'], num_classes=meta['classes']).to(args.device)
                elif arch_name == 'LeNet':
                    model = LeNet(in_channels=meta['in_ch'], num_classes=meta['classes']).to(args.device)
                elif arch_name == 'AlexNet':
                    model = AlexNet(in_channels=meta['in_ch'], num_classes=meta['classes']).to(args.device)
                elif arch_name == 'ShuffleNet':
                    model = ShuffleNetV2(in_channels=meta['in_ch'], num_classes=meta['classes']).to(args.device)
                elif arch_name == 'SqueezeNet':
                    model = SqueezeNet(in_channels=meta['in_ch'], num_classes=meta['classes']).to(args.device)
                else:
                    logger.log(f"Warning: Unknown model architecture '{arch_name}' in {filename}. Skipping.")
                    continue

                c_id_str = filename.split('_c')[1].split('_')[0] 
                model.client_id = int(c_id_str)

                model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
                model.eval()
                clients_dict[d_name].append(model)
                logger.log(f"Loaded {arch_name} for {d_name} from {filename}")

        if len(clients_dict[d_name]) == 0:
            logger.log(f"WARNING: No models loaded for dataset: {d_name}.")

    active_datasets = [d for d, models in clients_dict.items() if len(models) > 0]
    clients_dict = {k: clients_dict[k] for k in active_datasets}
    label_space_meta = {k: label_space_meta[k] for k in active_datasets}

    if len(active_datasets) < 2:
        logger.log("Not enough dataset models.")
        return
    
    dynamic_entropy_ratios = [round(x, 2) for x in np.arange(0.10, 1.05, 0.05)]

    if args.method in ['real', 'all']:
        logger.log("\n" + "="*40)
        logger.log(">>> [METHOD: REAL] Starting Real Image Label Mapping...")
        logger.log("="*40)
        
        test_loaders = {}
        for d_name in active_datasets:
            test_dataset = get_raw_dataset_transform(name=d_name, root=DATA_ROOT, train=False)
            test_loaders[d_name] = DataLoader(test_dataset, batch_size=64, shuffle=True)

        # 1. Real Img + Old Entropy
        logger.log("\n>>> Exp 1 (Real): Real Img + Old Entropy <<<")
        for thresh in dynamic_entropy_ratios:
            map_res = label_mapping(
                get_real_images, active_datasets, clients_dict, label_space_meta,
                entropy_ratio=thresh, use_new_entropy_method=False, logger=logger, args=args,
                test_loaders=test_loaders
            )
            metrics = evaluate_mapping_results(active_datasets, label_space_meta, map_res)
            save_mapping_results_to_csv(save_dir, "Real", "real_old_entropy.csv", metrics, ["thresh"], [thresh])

        # 2. Real Img + New Entropy
        logger.log("\n>>> Exp 2 (Real): Real Img + New Entropy <<<")
        for thresh in dynamic_entropy_ratios:
            map_res = label_mapping(
                get_real_images, active_datasets, clients_dict, label_space_meta,
                entropy_ratio=thresh, use_new_entropy_method=True, logger=logger, args=args,
                test_loaders=test_loaders
            )
            metrics = evaluate_mapping_results(active_datasets, label_space_meta, map_res)
            save_mapping_results_to_csv(save_dir, "Real", "real_new_entropy.csv", metrics, ["thresh"], [thresh])

    if args.method in ['syn', 'all', 'DI', 'FAST', 'NAYER']:
        logger.log("\n" + "="*40)
        logger.log(">>> [METHOD: SYNTHETIC] Starting Synthetic Label Mapping...")
        logger.log("="*40)

        if args.method in ['syn', 'all']:
            target_methods = ['DI', 'FAST', 'NAYER']
        else:
            target_methods = [args.method]

        logger.log(f">>> Target Methods to Run: {target_methods}")
        
        # 2. Train New Generators (Mask Filter)
        server_ckpt_path = os.path.join(args.model_path, "server_checkpoints.pth")
        client_label_mask_dict = {}
        if os.path.exists(server_ckpt_path):
            server_ckpt = torch.load(server_ckpt_path, map_location=args.device)
            client_label_distributions = server_ckpt.get('client_label_distributions', {})
            logger.log(">>> [Syn] Loaded client data distributions directly from server checkpoint.")
            
            for ls_id, dataset_clients in clients_dict.items():
                if len(dataset_clients) == 0: continue
                
                num_clients = len(dataset_clients)
                num_classes = len(label_space_meta[ls_id])
                mask_tensor = torch.zeros((num_clients, num_classes), device=args.device)
                
                dataset_clients_sorted = sorted(dataset_clients, key=lambda m: m.client_id)
                
                for idx, client_model in enumerate(dataset_clients_sorted):
                    c_id = client_model.client_id 
                    
                    if c_id in client_label_distributions:
                        valid_labels = client_label_distributions[c_id]
                        mask_tensor[idx, valid_labels] = 1.0
                        
                client_label_mask_dict[ls_id] = mask_tensor
        else:
            logger.log(f">>> [!] Checkpoint not found at {server_ckpt_path}")

        for method_name in target_methods:
            logger.log(f"\n{'#'*40}")
            logger.log(f">>> [Syn] Running Method: {method_name}")
            logger.log(f"{'#'*40}")

            # if method_name == 'FAST':
            #     from generator_trainer import train_generators_FAST as trainer_func
            # elif method_name == 'NAYER':
            #     from generator_trainer import train_generators_NAYER as trainer_func
            # elif method_name == 'DI': 
            #     from generator_trainer import train_generators_DeepInversion as trainer_func

            if method_name == 'FAST':
                from generator_trainer_2 import train_generators_FAST as trainer_func
            elif method_name == 'NAYER':
                from generator_trainer_2 import train_generators_NAYER as trainer_func
            elif method_name == 'DI': 
                from generator_trainer_2 import train_generators_DeepInversion as trainer_func

            student_models_old = {}
            student_models_new = {}

            for d_name in active_datasets:
                meta = DATASET_META[d_name]
                student_old = MobileNetV2(in_channels=meta['in_ch'], num_classes=meta['classes']).to(args.device)
                student_new = MobileNetV2(in_channels=meta['in_ch'], num_classes=meta['classes']).to(args.device)
                
                student_models_old[d_name] = student_old
                student_models_new[d_name] = student_new

            if args.gen_mode in ['old', 'all']:
                logger.log(f"\n>>> Syn-{method_name} Training Old Generators...")
                gens_old = trainer_func(
                    clients_dict, label_space_meta, DATASET_META, args.device, logger, 
                    use_new_gen_method=False, client_label_mask_dict=None, 
                    student_model=student_models_old
                )

                # 1: Old Gen + Old Entropy 
                for t in dynamic_entropy_ratios:
                    logger.log(f"Exp: Old Gen + Old Ent ({t})")
                    res = label_mapping(get_gen_images, active_datasets, clients_dict, label_space_meta, t, False, logger, args=args, gen_dict=gens_old)
                    metrics = evaluate_mapping_results(active_datasets, label_space_meta, res)
                    save_mapping_results_to_csv(save_dir, method_name, "syn_oldGen_oldEnt.csv", metrics, ["thresh"], [t])

                # 3: Old Gen + New Entropy
                for t in dynamic_entropy_ratios:
                    logger.log(f"Exp: Old Gen + New Ent ({t})")
                    res = label_mapping(get_gen_images, active_datasets, clients_dict, label_space_meta, t, True, logger, args=args, gen_dict=gens_old)
                    metrics = evaluate_mapping_results(active_datasets, label_space_meta, res)
                    save_mapping_results_to_csv(save_dir, method_name, "syn_oldGen_newEnt.csv", metrics, ["thresh"], [t])

            elif args.gen_mode in ['new', 'all']:
                logger.log(f"\n>>> Syn-{method_name} Training New Generators...")
                gens_new = trainer_func(
                    clients_dict, label_space_meta, DATASET_META, args.device, logger, 
                    use_new_gen_method=True, client_label_mask_dict=client_label_mask_dict,
                    student_model=student_models_new
                )

                # 2: New Gen + Old Entropy
                for t in dynamic_entropy_ratios:
                    logger.log(f"Exp (Syn): New Gen + Old Ent ({t})")
                    res = label_mapping(get_gen_images, active_datasets, clients_dict, label_space_meta, t, False, logger, args=args, gen_dict=gens_new)
                    metrics = evaluate_mapping_results(active_datasets, label_space_meta, res)
                    save_mapping_results_to_csv(save_dir, method_name, "syn_newGen_oldEnt.csv", metrics, ["thresh"], [t])

                # 4: New Gen + New Entropy
                for t in dynamic_entropy_ratios:
                    logger.log(f"Exp (Syn): New Gen + New Ent ({t})")
                    res = label_mapping(get_gen_images, active_datasets, clients_dict, label_space_meta, t, True, logger, args=args, gen_dict=gens_new)
                    metrics = evaluate_mapping_results(active_datasets, label_space_meta, res)
                    save_mapping_results_to_csv(save_dir, method_name, "syn_newGen_newEnt.csv", metrics, ["thresh"], [t])

            # # 1: Old Gen + Old Entropy 
            # for t in dynamic_entropy_ratios:
            #     logger.log(f"Exp: Old Gen + Old Ent ({t})")
            #     res = label_mapping(get_gen_images, active_datasets, clients_dict, label_space_meta, t, False, logger, args=args, gen_dict=gens_old)
            #     metrics = evaluate_mapping_results(active_datasets, label_space_meta, res)
            #     save_mapping_results_to_csv(save_dir, method_name, "syn_oldGen_oldEnt.csv", metrics, ["thresh"], [t])

            # # 2: New Gen + Old Entropy
            # for t in dynamic_entropy_ratios:
            #     logger.log(f"Exp (Syn): New Gen + Old Ent ({t})")
            #     res = label_mapping(get_gen_images, active_datasets, clients_dict, label_space_meta, t, False, logger, args=args, gen_dict=gens_new)
            #     metrics = evaluate_mapping_results(active_datasets, label_space_meta, res)
            #     save_mapping_results_to_csv(save_dir, method_name, "syn_newGen_oldEnt.csv", metrics, ["thresh"], [t])

            # # 3: Old Gen + New Entropy
            # for t in dynamic_entropy_ratios:
            #     logger.log(f"Exp: Old Gen + New Ent ({t})")
            #     res = label_mapping(get_gen_images, active_datasets, clients_dict, label_space_meta, t, True, logger, args=args, gen_dict=gens_old)
            #     metrics = evaluate_mapping_results(active_datasets, label_space_meta, res)
            #     save_mapping_results_to_csv(save_dir, method_name, "syn_oldGen_newEnt.csv", metrics, ["thresh"], [t])

            # # 4: New Gen + New Entropy
            # for t in dynamic_entropy_ratios:
            #     logger.log(f"Exp (Syn): New Gen + New Ent ({t})")
            #     res = label_mapping(get_gen_images, active_datasets, clients_dict, label_space_meta, t, True, logger, args=args, gen_dict=gens_new)
            #     metrics = evaluate_mapping_results(active_datasets, label_space_meta, res)
            #     save_mapping_results_to_csv(save_dir, method_name, "syn_newGen_newEnt.csv", metrics, ["thresh"], [t])


if __name__ == "__main__":
    main()