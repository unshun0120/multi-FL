import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict

from data.datasets import get_readable_class_names
from models.hetero_model import get_heterogeneous_model
from client import Client
from utils.plotting import plot_accuracy_curves

def get_ood_soft_label(args, batch_size, num_classes, threshold, max_iter=1000):
    device = args.device
    
    perfect_soft_label = torch.ones(num_classes, device=device) / num_classes

    soft_labels = []
    for _ in range(batch_size):
        SL_found = False
        for i in range(max_iter):
            random_logits = torch.randn(num_classes, device=device)
            random_soft_label = F.softmax(random_logits, dim=0)

            kl_div = F.kl_div(perfect_soft_label.log(), random_soft_label, reduction='sum')
            
            if kl_div.item() <= threshold:
                soft_labels.append(random_soft_label)
                SL_found = True
                break
        
        if not SL_found:
            soft_labels.append(perfect_soft_label)
    
    soft_labels = torch.stack(soft_labels, dim=0)
    
    return soft_labels


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


def distribute_to_clients(server, clients):
    print("[Server] Distributing generator and update classifier...")
    for client in clients:
        global_clf_weight = server.get_global_classifier(client.class_names)
        client.update_local_model(global_clf_weight)


def server_update(server, client_uploads, logger):
    print(f"[Server] Aggregating...")
    server.aggregate_clients(client_uploads, logger)

    print("[Server] Training Generator...")
    server.train_generator(logger)

    print("[Server] Training Shared Classifier...")
    server.train_global_shared_classifier(logger)


def client_local_training(clients, id_to_dataset, round_acc, rnd, logger):
    client_uploads = []
    
    for client in tqdm(clients, desc="Local Training"):
        loss = client.local_train()
        acc = client.test()

        # 給server的東西
        payload = {
            'client_id': client.client_id,
            'class_names': client.class_names,
            'classifier_state_dict': client.model.classifier.state_dict()
        }
        client_uploads.append(payload)

        round_acc = record_round_acc(rnd, client, loss, acc, id_to_dataset, round_acc, logger)

    return client_uploads, round_acc


def record_round_acc(rnd, client, loss, acc, id_to_dataset, round_acc_dataset, logger): 
    d_name = id_to_dataset[client.client_id]
    round_acc_dataset[d_name].append(acc)
    logger.log(f"Round {rnd} | Client {client.client_id} | Model: ({client.model_name}) | Loss: {loss:.4f} | Acc: {acc:.2f}%")

    return round_acc_dataset


def final_round_acc(rnd, round_acc_dataset, history, logger, args, mode=""):
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


def initial_evaluation(clients, id_to_dataset, logger):
    round_acc = defaultdict(list)
    for client in tqdm(clients, desc="Initial Testing", ncols=100):
        acc = client.test()
        round_acc = record_round_acc(0, client, 0, acc, id_to_dataset, round_acc, logger)
    
    return round_acc


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
