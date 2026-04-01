import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.datasets import get_readable_class_names
from utils.nets import get_heterogeneous_model


def initialize_new_clients(Client, all_client_data_loaders, dataset_meta, model_list, data_root, global_registry, args, logger, **kwargs):
    print("Initializing New Clients...")
    global_feature_dim = kwargs.get('global_feature_dim', 128)

    new_clients = []
    client_id_counter = 0   
    heterogeneous = kwargs.get('heterogeneous', False)

    for d_name, loaders_list in all_client_data_loaders.items():
        d_meta = dataset_meta.get(d_name)
        class_name_set = get_readable_class_names(d_name, data_root)
        num_train_client = len(loaders_list) - args.num_new_clients

        for idx, loader_dict in enumerate(loaders_list):
            if idx >= num_train_client:
                train_loader = loader_dict['train']
                test_loader = loader_dict['test']

                if heterogeneous: 
                    model_arch_id = idx % 10
                    model = get_heterogeneous_model(
                        node_id=model_arch_id,
                        in_channels=d_meta['in_ch'],
                        num_classes=d_meta['classes'],
                        img_size=d_meta['size'],
                        global_dim=global_feature_dim
                    )
                else:
                    model_arch_id = 0
                    model = get_heterogeneous_model(
                        node_id=model_arch_id,
                        in_channels=d_meta['in_ch'],
                        num_classes=d_meta['classes'],
                        img_size=d_meta['size'],
                        global_dim=global_feature_dim
                    )

                client = Client(
                    node_id=client_id_counter,
                    args=args,
                    dataset_name=d_name, 
                    train_loader=train_loader,
                    test_loader=test_loader,
                    public_train_loaders=None, 
                    public_test_loaders=None,
                    model=model,
                    class_name_set=sorted(class_name_set), 
                    model_name=model_list[model_arch_id],
                    global_registry=global_registry,
                    logger=logger,
                    **kwargs
                )

                new_clients.append(client)

            client_id_counter += 1

    print(f"Total New Clients Initialized: {len(new_clients)}")

    return new_clients


def evaluate_client(client, args):
    """Evaluate client model on test set"""
    client.model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in client.test_loader:
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            _, logits = client.model(imgs)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total if total > 0 else 0.0