import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict

from data.datasets import get_readable_class_names
from models.hetero_model import get_heterogeneous_model
from client import Client
from utils.plotting import plot_accuracy_curves



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