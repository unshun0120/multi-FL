import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from data.datasets import get_readable_class_names
from utils.nets import get_heterogeneous_model
from utils.nets import TwinBranchNets

def initialize_training_clients(Client, all_client_data_loaders, dataset_meta, model_list, args, data_root, logger, **exp_conf):
    print("Creating and Initializing Training Clients...")

    global_feature_dim = exp_conf.get('global_feature_dim', 128)
    heterogeneous = exp_conf.get('heterogeneous', False)

    train_clients = []  
    client_id = 0
    
    for d_name, loaders_list in all_client_data_loaders.items():
        d_meta = dataset_meta.get(d_name)
        class_name_set = get_readable_class_names(d_name, data_root)
        num_train_client = len(loaders_list) - args.num_new_clients

        for idx, loader_dict in enumerate(loaders_list):
            # loader_dict = train dataloader + test dataloader
            if idx < num_train_client:
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
                    model_arch_id = 1
                    model = get_heterogeneous_model(
                        node_id=model_arch_id,
                        in_channels=d_meta['in_ch'],
                        num_classes=d_meta['classes'],
                        img_size=d_meta['size'],
                        global_dim=global_feature_dim
                    )

                if args.algorithm == 'FedTED' or args.algorithm == 'FedTED_dir' or args.algorithm == 'FedTED_DDPM' or args.algorithm == 'FedTED_DDPM_2':
                    model = TwinBranchNets(model)

                client = Client(
                    node_id=client_id,
                    args=args,
                    dataset_name = d_name, 
                    train_loader=train_loader,
                    test_loader=test_loader,
                    model=model,
                    class_name_set=sorted(class_name_set),
                    model_name=model_list[model_arch_id],
                    logger=logger,
                    **exp_conf
                )

                train_clients.append(client)

            client_id += 1

    print(f"Total Training Clients Initialized: {len(train_clients)}")

    return train_clients

def train_model(model, train_loader, optimizer, loss_fn, epochs, device):
    model.to(device)
    model.train()
    loss_metric = []

    for epoch in range(epochs):
        total_loss, num_samples = 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            feat, logits = model(imgs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss
            num_samples += labels.size(0)

        total_loss /= num_samples
        loss_metric.append(total_loss)

    model.to('cpu')
    torch.cuda.empty_cache()

    avg_loss = sum(loss_metric) / len(loss_metric)
    return avg_loss


def evaluate_model(model, test_loader, metric_type, device):
    model.eval()
    model.to(device)

    accuracy, num_samples = 0, 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            feat, logits = model(imgs)
            # loss = loss_fn(logits, labels)

            if metric_type == 'accuracy':
                predicted = logits.argmax(dim=1, keepdim=True)
                accuracy += predicted.eq(labels.view_as(predicted)).sum().item()
            elif metric_type == 'mse':
                accuracy += F.mse_loss(logits, labels, reduction='sum').item()
            
            num_samples += labels.size(0)

    accuracy = accuracy / num_samples

    model.to('cpu')
    torch.cuda.empty_cache()

    return accuracy


def freeze(model, freeze_name=None):
    """freeze model parameters"""
    set_requires_grad(model, freeze_name, False)

def unfreeze(model, unfreeze_name=None):
    """unfreeze model parameters"""
    set_requires_grad(model, unfreeze_name, True)

def set_requires_grad(model, param_name=None, requires_grad=False):
    for name, param in model.named_parameters():
        if param_name is None:
            param.requires_grad = requires_grad
        else:
            if param_name in name:
                param.requires_grad = requires_grad

