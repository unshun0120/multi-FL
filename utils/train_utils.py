"""Base train functions for reusing
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from data.datasets import get_readable_class_names
from utils.nets import get_heterogeneous_model
from utils.nets import TwinBranchNets


def initialize_training_clients(Client, all_client_data_loaders, dataset_meta, model_list, args, data_root, global_registry, logger, **kwargs):
    print("Creating and Initializing Training Clients...")

    global_feature_dim = kwargs.get('global_feature_dim', 128)

    train_clients = []  
    client_id_counter = 0   

    for d_name, loaders_list in all_client_data_loaders.items():
        d_meta = dataset_meta.get(d_name)
        class_name_set = get_readable_class_names(d_name, data_root)
        num_train_client = len(loaders_list) - args.num_new_clients

        for idx, loader_dict in enumerate(loaders_list):
            # loader_dict = train dataloader + test dataloader
            if idx < num_train_client:
                train_loader = loader_dict['train']
                test_loader = loader_dict['test']

                model_arch_id = idx % 10
                model = get_heterogeneous_model(
                    node_id=model_arch_id,
                    in_channels=d_meta['in_ch'],
                    num_classes=d_meta['classes'],
                    img_size=d_meta['size'],
                    global_dim=global_feature_dim
                )

                if args.algorithm == 'FedTED':
                    model = TwinBranchNets(model)

                client = Client(
                    node_id=client_id_counter,
                    args=args,
                    dataset_name = d_name, 
                    train_loader=train_loader,
                    test_loader=test_loader,
                    model=model,
                    class_name_set=sorted(class_name_set), 
                    model_name=model_list[model_arch_id],
                    global_registry=global_registry,
                    logger=logger,
                    **kwargs
                )

                train_clients.append(client)

            client_id_counter += 1
    
    print(f"Total Training Clients Initialized: {len(train_clients)}")

    return train_clients

def train_model(model: nn.Module, train_loader: DataLoader,
                optimizer=None, criterion=nn.CrossEntropyLoss(),
                epochs=1, device='cpu'):
    """

    """
    # rationalization proposal
    assert epochs > 0
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # step 1. model init
    model.to(device)
    model.train()
    # step 2. train loop
    loss_metric = []  # to record avg loss
    for epoch in range(epochs):
        # init loss value
        loss_value, num_samples = 0, 0
        # one epoch train
        for i, (x, y) in enumerate(train_loader):
            # put tensor into same device
            x, y = x.to(device), y.to(device)
            # calculate loss
            feat, y_ = model(x)
            loss = criterion(y_, y)
            # backward & step optim
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # get loss valur of current bath
            loss_value += loss.item()
            num_samples += y.size(0)

        # Use mean loss value of each epoch as metric
        # Just a reference value, not precise. If you want precise, dataloader should set `drop_last = True`.
        loss_value = loss_value / num_samples
        loss_metric.append(loss_value)

    # step 3. release gpu resource
    model.to('cpu')
    torch.cuda.empty_cache()

    avg_loss = sum(loss_metric) / len(loss_metric)
    return avg_loss


def evaluate_model(model: nn.Module, test_loader, criterion=nn.CrossEntropyLoss(),
                   metric_type='accuracy', device='cpu', release=True):
    """
    """

    # rationalization proposal
    assert metric_type in ['accuracy', 'mse']

    # init model with eval mode
    model.eval()
    model.to(device)

    # init metric and loss value
    loss_value, accuracy, num_samples = 0, 0, 0

    # test by test loader
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            # forward
            _, y_ = model(x)
            # record loss value
            loss_value += criterion(y_, y).item()
            # metric correct
            if metric_type == 'accuracy':
                predicted = y_.argmax(dim=1, keepdim=True)
                accuracy += predicted.eq(y.view_as(predicted)).sum().item()
            elif metric_type == 'mse':
                accuracy += F.mse_loss(y_, y, reduction='sum').item()
            num_samples += y.size(0)

    # cal metric
    loss_value = loss_value / len(test_loader)
    accuracy = accuracy / num_samples

    # release gpu resource
    if release:
        model.to('cpu')
        torch.cuda.empty_cache()

    return accuracy, loss_value


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