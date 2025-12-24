import os
import torch

from data.datasets import get_readable_class_names
from models.hetero_model import get_heterogeneous_model
from client import Client
from utils.plotting import plot_accuracy_curves

def save_gen_model(self, args, rnd, logger):
    save_path = os.path.join(logger.get_log_dir(), 'checkpoint.pth')
    logger.log(f"Saving checkpoint (Round {rnd}) to {save_path} ...")

    checkpoint = {
        'args': self.args, 
        'round': rnd, 
        
        'label_to_global_id': self.label_to_global_id,
        'global_id_to_label': self.global_id_to_label,
        'label_space_meta': self.label_space_meta,
        
        'generator': None,
        'shared_classifier': None,
        'dataset_classifiers': {}, 
        'optimizer_g': None
    }

    if self.generator is not None:
        checkpoint['generator'] = self.generator.state_dict()
        
    if self.shared_classifier is not None:
        checkpoint['shared_classifier'] = self.shared_classifier.state_dict()

    for ls_id, clf in self.global_classifiers.items():
        checkpoint['dataset_classifiers'][ls_id] = clf.state_dict()

    torch.save(checkpoint, save_path)
    print(f"[Server] Models saved to {save_path}")

def record_round_acc(rnd, client, loss, acc, id_to_dataset, round_acc_dataset, logger): 
    d_name = id_to_dataset[client.client_id]
    round_acc_dataset[d_name].append(acc)
    if rnd == 0: 
        logger.log(f"Round 0 | Client {client.client_id} | Acc: {acc:.2f}%", print_to_console=False)
    else:
        logger.log(f"Round {rnd} | Client {client.client_id} | Model: ({client.model_name}) | Loss: {loss:.4f} | Acc: {acc:.2f}%")

    return round_acc_dataset

def record_plot_total_acc(rnd, round_acc_dataset, history, logger, args, mode=""):
    log_msg_parts = []
    for d_name, acc_list in round_acc_dataset.items():
        avg = sum(acc_list) / len(acc_list)
        history['train_detail'][d_name].append(avg)
        log_msg_parts.append(f"{d_name}: {avg:.2f}%")

    if rnd == 0:
        full_log_msg = f"Round 0 Init Acc | " + " | ".join(log_msg_parts)
    else:
        full_log_msg = f"Round {rnd} Train Acc | " + " | ".join(log_msg_parts)
    logger.log(full_log_msg)

    if args.no_plot_log: 
        plot_accuracy_curves(history, mode=mode, save_dir=logger.get_log_dir(), args=args)

    return history

def initialize_training_clients(all_client_data_loaders, dataset_meta, model_list, args, data_root):
    print("Initializing Training Clients...")
    train_clients = []  # 存所有訓練client object
    id_to_dataset = {}  # 存{client_id: dataset_name}的mapping, 用來統計每個資料集的準確率
    client_id_counter = 0   # FL架構下所有client的編號（所有資料集共用）

    for d_name, loaders_list in all_client_data_loaders.items():
        d_meta = dataset_meta.get(d_name)
        full_class_names = get_readable_class_names(d_name, data_root)
        # 該資料集訓練client數量 = 該資料集下所有client (len(loaders_list)) - new client數量 
        num_train_client = len(loaders_list) - args.num_new_clients

        # 逐一跑過這個資料集下的每個client的train set和test set
        for idx, loader_dict in enumerate(loaders_list):
            # idx是訓練client id的時候再初始化Client object
            # loader_dict: train dataloader + test dataloader
            if idx < num_train_client:
                train_loader = loader_dict['train']
                test_loader = loader_dict['test']

                # 根據client id分配模型 (Model Heterogeneity)
                model_arch_id = idx % 10
                model = get_heterogeneous_model(
                    client_id=model_arch_id,
                    in_channels=d_meta['in_ch'],
                    num_classes=d_meta['classes'],
                    img_size=d_meta['size'],
                    global_dim=args.global_feature_dim
                )

                # 初始化Client object
                client = Client(
                    client_id=client_id_counter,
                    args=args,
                    train_dataset=train_loader.dataset,
                    test_dataset=test_loader.dataset,
                    model=model,
                    class_names=full_class_names, 
                    model_name=model_list[model_arch_id]
                )

                train_clients.append(client)
                # 在id_to_dataset裡記下這個client是哪個資料集
                id_to_dataset[client_id_counter] = d_name

            client_id_counter += 1

    print(f"Total Training Clients Initialized: {len(train_clients)}")

    return train_clients, id_to_dataset

def initialize_new_clients(all_client_data_loaders, dataset_meta, model_list, args, data_root):
    print("Initializing New Clients...")
    new_clients = []
    id_to_dataset = {}  
    client_id_counter = 0   

    for d_name, loaders_list in all_client_data_loaders.items():
        d_meta = dataset_meta.get(d_name)
        full_class_names = get_readable_class_names(d_name, data_root)
        num_train_client = len(loaders_list) - args.num_new_clients

        for idx, loader_dict in enumerate(loaders_list):
            if idx >= num_train_client:
                train_loader = loader_dict['train']
                test_loader = loader_dict['test']

                client = Client(
                    client_id=client_id_counter,
                    args=args,
                    train_dataset=train_loader.dataset,
                    test_dataset=test_loader.dataset,
                    class_names=full_class_names
                )

                new_clients.append(client)
                id_to_dataset[client_id_counter] = d_name

            client_id_counter += 1

    print(f"Total New Clients Initialized: {len(new_clients)}")

    return new_clients, id_to_dataset