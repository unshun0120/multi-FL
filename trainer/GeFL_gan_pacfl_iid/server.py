"""
Server for GeFL gan pacfl noniid
"""

import torch
import copy
from collections import OrderedDict, defaultdict
import os
import csv
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import random
from torch.optim import *
from torch.utils.data import TensorDataset, DataLoader

from trainer.BaseFL.server import Server as BaseServer
from utils.plotting import plot_accuracy_curves
from utils.nets import ResNet, BasicBlock, DCGANGenerator
from utils.pacfl_utils import calculating_adjacency, hierarchical_clustering
from label_mapping.label_mapping_utils import (
    label_mapping, evaluate_mapping_results, 
    feature_bi_direction_label_mapping, single_direction_label_mapping,
    get_gen_images, global_to_local_mapping, clear_image_caches,
    image_cosine_similarity_mapping, missing_link_label_mapping
)
from utils.nets import ResNet, BasicBlock

class Server(BaseServer):
    def __init__(self, **kwargs):
        super(Server, self).__init__(**kwargs)

        self.global_gen_states = {}
        self.global_dis_states = {}
        self.model = None

        self.aggregate_method = self.args.aggregate_method
        self.client_groups = defaultdict(list)
        self.group_label_space_meta = {}

    def initialize_client_groups(self):
        self.client_groups = defaultdict(list)
        self.group_label_space_meta = {}

        if self.aggregate_method == "dataset_name":
            self.logger.log("[Server] Grouping clients by dataset name ...")

            for client in self.clients:
                client.group_name = client.dataset_name
                self.client_groups[client.group_name].append(client)

        elif self.aggregate_method == "pacfl":
            self.logger.log("[Server] Grouping clients by PACFL ...")

            n_basis = 3
            cluster_alpha = 15
            U_clients = []

            for client in self.clients:
                local_images = []
                count = 0

                for imgs, _ in client.train_loader:
                    local_images.append(imgs.detach().cpu().numpy())
                    count += imgs.size(0)

                    if count >= 500:
                        break

                local_ds = np.concatenate(local_images, axis=0)[:500]
                local_ds = local_ds.reshape(local_ds.shape[0], -1).T
                local_ds = (local_ds * 0.5) + 0.5

                u_temp, _, _ = np.linalg.svd(local_ds, full_matrices=False)
                u_temp = u_temp / np.linalg.norm(u_temp, ord=2, axis=0)
                U_clients.append(copy.deepcopy(u_temp[:, :n_basis]))

            adj_mat = calculating_adjacency(list(range(len(self.clients))), U_clients)
            clusters_idx = hierarchical_clustering(copy.deepcopy(adj_mat), thresh=cluster_alpha, linkage="average")

            for cluster_id, client_indices in enumerate(clusters_idx):
                group_name = f"Cluster_{cluster_id}"

                for idx in client_indices:
                    client = self.clients[idx]
                    client.group_name = group_name
                    self.client_groups[group_name].append(client)

        for group_name, group_clients in self.client_groups.items():
            datasets = {client.dataset_name for client in group_clients}
            self.group_label_space_meta[group_name] = list(group_clients[0].class_name_set)
            client_ids = [client.id for client in group_clients]
            self.logger.log(f"{group_name}: clients={client_ids} | datasets={datasets}")
        
    def run(self):
        self.logger.log("")
        self.logger.log("=" * 50)
        self.logger.log(f"Start {self.global_rounds} rounds training by {self.algorithm}")

        self.initialize_client_groups()

        for r in range(self.global_rounds):
            self.glob_iter = r

            self.sample_clients()
            self.distribute_model()
            self.local_update()

            # if (r + 1) % self.test_interval == 0:
            #     self.evaluate_private()

            self.aggregate()

            if (r+1) % 5 == 0:
                self.save_model(r+1) 

        # self.save_model(r + 1)
        # plot_accuracy_curves(self.dataset_acc_history, self.logger.log_dir, self.args, self.global_rounds, self.dirichlet_alpha)


    def get_all_clients_averaged_features(self):
        dataset_features = {}

        for client in self.clients:
            d_id = f"client_{client.id}"

            if d_id not in dataset_features:
                dataset_features[d_id] = {}

            client_avg_feats = client.get_avg_features()

            for lbl, feat in client_avg_feats.items():
                if lbl not in dataset_features[d_id]:
                    dataset_features[d_id][lbl] = []

                dataset_features[d_id][lbl].append(feat)

        for d_id in dataset_features:
            for lbl in dataset_features[d_id]:
                dataset_features[d_id][lbl] = torch.cat(dataset_features[d_id][lbl], dim=0)

        return dataset_features
    

    def aggregate(self):
        groups = defaultdict(list)

        for client in self.selected_clients:
            groups[client.group_name].append(client)

        for group_name, group_clients in groups.items():
            datasets = {client.dataset_name for client in group_clients}
            self.logger.log(f"[Aggregate] {group_name}: {len(group_clients)} clients | datasets={datasets}")

            gen_msg_list = [
                (client.num_samples, client.generator.state_dict())
                for client in group_clients
            ]
            self.global_gen_states[group_name] = self.aggregate_weights(gen_msg_list)

            dis_msg_list = [
                (client.num_samples, client.discriminator.state_dict())
                for client in group_clients
            ]
            self.global_dis_states[group_name] = self.aggregate_weights(dis_msg_list)

        if (self.glob_iter + 1) == self.start_mapping_epoch: 

            clear_image_caches()
            dataset_clients_dict = {}   
            active_datasets = []
            dataset_label_space_meta = {}
            generators = {}
            valid_labels_dict = {}

            for group_name, group_clients in self.client_groups.items():
                if group_name not in self.global_gen_states:
                    self.logger.log(f"[Mapping] Skip {group_name}: no aggregated GAN state.")
                    continue

                dataset_clients_dict[group_name] = [
                    client.model.to(self.device)
                    for client in group_clients
                ]

                dataset_label_space_meta[group_name] = self.group_label_space_meta[group_name]

                group_valid_labels = set()
                for client in group_clients:
                    for _, labels in client.train_loader:
                        group_valid_labels.update(labels.tolist())

                valid_labels_dict[group_name] = sorted(group_valid_labels)

                num_local_classes = len(self.group_label_space_meta[group_name])

                gen = DCGANGenerator(
                    num_classes=num_local_classes,
                    noise_dim=self.exp_conf.get("gen_noise_dim", 128),
                    img_size=self.exp_conf.get("img_size", 32),
                    channels=self.exp_conf.get("channels", 3)
                ).to(self.device)

                gen.load_state_dict(self.global_gen_states[group_name])
                gen.eval()

                generators[group_name] = gen
                active_datasets.append(group_name)

                datasets = {client.dataset_name for client in group_clients}

            if not active_datasets:
                return 
            
            use_new_ent = self.exp_conf.get('use_new_entropy_method', False)
            entropy_ratio = self.exp_conf.get('entropy_ratio', 1.0) 
            cs_threshold = self.exp_conf.get('cs_threshold', 1.0) 
            missing_threshold = self.exp_conf.get('missing_threshold', 1.0)

            mapping = None

            if self.args.label_mapping == 'image-bi':
                mapping = label_mapping(
                    get_images_func=get_gen_images, 
                    dataset_ids=active_datasets,    
                    clients_dict=dataset_clients_dict,
                    label_space_meta=dataset_label_space_meta,
                    entropy_ratio=entropy_ratio,
                    use_new_entropy_method=use_new_ent, 
                    logger=self.logger, 
                    args=self.args,
                    gen_dict=generators,
                    valid_labels_dict=valid_labels_dict
                )
                global_map = global_to_local_mapping(mapping, logger=self.logger, label_space_meta=dataset_label_space_meta)
                self.local_id_to_global_id = mapping

            elif self.args.label_mapping == 'image-single':
                mapping = single_direction_label_mapping(
                    get_images_func=get_gen_images,  
                    dataset_ids=active_datasets,
                    clients_dict=dataset_clients_dict,
                    label_space_meta=dataset_label_space_meta,
                    entropy_ratio=entropy_ratio, 
                    use_new_entropy_method=use_new_ent,
                    logger=self.logger,
                    args=self.args,
                    gen_dict=generators,
                    valid_labels_dict=valid_labels_dict
                )
                global_map = global_to_local_mapping(mapping, logger=self.logger, label_space_meta=dataset_label_space_meta)
                self.local_id_to_global_id = mapping

            elif self.args.label_mapping == 'feature-bi':
                dataset_feats = self.get_all_clients_averaged_features()
                mapping = feature_bi_direction_label_mapping(
                    dataset_features_dict=dataset_feats,
                    dataset_ids=active_datasets,
                    clients_dict=dataset_clients_dict,
                    label_space_meta=dataset_label_space_meta,
                    entropy_ratio=entropy_ratio, 
                    use_new_entropy_method=use_new_ent,
                    logger=self.logger,
                    valid_labels_dict=valid_labels_dict
                )
                global_map = global_to_local_mapping(mapping, logger=self.logger, label_space_meta=dataset_label_space_meta)
                self.local_id_to_global_id = mapping

            elif self.args.label_mapping == 'image-cs':
                mapping = image_cosine_similarity_mapping(
                    get_images_func=get_gen_images,
                    dataset_ids=active_datasets,
                    label_space_meta=dataset_label_space_meta,
                    cs_threshold=cs_threshold,
                    logger=self.logger,
                    args=self.args,
                    gen_dict=generators,
                    valid_labels_dict=valid_labels_dict
                )
                global_map = global_to_local_mapping(mapping, logger=self.logger, label_space_meta=dataset_label_space_meta)
                self.local_id_to_global_id = mapping

            elif self.args.label_mapping == 'missing_link':
                mapping = missing_link_label_mapping(
                    get_images_func=get_gen_images,
                    dataset_ids=active_datasets,
                    clients_dict=dataset_clients_dict,
                    label_space_meta=dataset_label_space_meta,
                    missing_threshold=missing_threshold,
                    logger=self.logger, 
                    args=self.args,
                    gen_dict=generators,
                    valid_labels_dict=valid_labels_dict
                )
                global_map = global_to_local_mapping(mapping, logger=self.logger, label_space_meta=dataset_label_space_meta)
                self.local_id_to_global_id = mapping

            elif self.args.label_mapping == 'class_name':
                mapping = {}
                class_name_to_global_id = {}
                next_global_id = 0

                for d_name in active_datasets:
                    mapping[d_name] = {}

                    for local_id, class_name in enumerate(dataset_label_space_meta[d_name]):
                        if d_name in valid_labels_dict and local_id not in valid_labels_dict[d_name]:
                            continue

                        if class_name not in class_name_to_global_id:
                            class_name_to_global_id[class_name] = next_global_id
                            next_global_id += 1

                        mapping[d_name][local_id] = class_name_to_global_id[class_name]

                self.local_id_to_global_id = mapping

                global_to_local_mapping(
                    mapping,
                    logger=self.logger,
                    label_space_meta=dataset_label_space_meta
                )

            elif self.args.label_mapping == 'independent':
                self.local_id_to_global_id = {}
                current_gid = 0

                for group_name in active_datasets:
                    self.local_id_to_global_id[group_name] = {}

                    for local_id in valid_labels_dict[group_name]:
                        self.local_id_to_global_id[group_name][local_id] = current_gid
                        current_gid += 1

                global_to_local_mapping(
                    self.local_id_to_global_id,
                    logger=self.logger,
                    label_space_meta=dataset_label_space_meta
                )

            elif self.args.label_mapping == 'identical':
                self.identical_label_mapping()

        if (self.glob_iter + 1) >= self.start_mapping_epoch: 
            self.train_global_inference_model()
            self.test_global_inference_model()
    
    def train_global_inference_model(self):
        all_global_ids = set()
        for d_name, mapping in self.local_id_to_global_id.items():
            for l_id, g_id in mapping.items():
                all_global_ids.add(g_id)
        all_global_ids = list(all_global_ids)
        num_global_classes = len(all_global_ids)
        
        if num_global_classes == 0:
            self.logger.log("Warning: No mappings found via local_id_to_global_id. Skipping global model training.")
            return

        if self.model is None:
            self.logger.log(f"Initializing Global Inference Model with {num_global_classes} classes...")
            self.model = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=3, num_classes=num_global_classes, global_dim=256)
            
            optim_name = self.exp_conf.get('global_model_optim', 'Adam')
            optim_lr = self.exp_conf.get('global_model_optim_lr', 1e-3)
            self.global_model_optimizer = eval(optim_name)(self.model.parameters(), optim_lr)

        self.model.to(self.device)
        self.model.train()

        criterion = nn.CrossEntropyLoss()
        
        img_size = (self.exp_conf.get('channels', 3), self.exp_conf.get('img_size', 32), self.exp_conf.get('img_size', 32))
        
        dataset_x = []
        dataset_y = []
        noise_dim = self.exp_conf.get("gen_noise_dim", 128)

        for group_name, mapping in self.local_id_to_global_id.items():
            if group_name not in self.global_gen_states:
                continue

            num_local_classes = len(self.group_label_space_meta[group_name])

            gen = DCGANGenerator(
                num_classes=num_local_classes,
                noise_dim=noise_dim,
                img_size=self.exp_conf.get("img_size", 32),
                channels=self.exp_conf.get("channels", 3)
            ).to(self.device)

            gen.load_state_dict(self.global_gen_states[group_name])
            gen.eval()

            self.logger.log(f"[Global Train] Generating images from {group_name}")

            for local_id, global_id in mapping.items():
                z = torch.randn(self.global_samples_per_class, noise_dim, device=self.device)
                y_local = torch.full((self.global_samples_per_class,), local_id, dtype=torch.long, device=self.device)

                with torch.no_grad():
                    x_gen = gen(z, y_local)

                dataset_x.extend(x_gen.detach().cpu())
                dataset_y.extend([global_id] * self.global_samples_per_class)

        if not dataset_x:
            self.logger.log("No synthetic data generated")
            return

        dataset_x = torch.stack(dataset_x)
        dataset_y = torch.tensor(dataset_y, dtype=torch.long)
        
        train_dataset = TensorDataset(dataset_x, dataset_y)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in tqdm(range(self.global_model_epochs), colour="green", ncols=100):
            epoch_loss = 0
            for gen_imgs, y_batch in train_loader:
                gen_imgs, y_batch = gen_imgs.to(self.device), y_batch.to(self.device)
                
                self.global_model_optimizer.zero_grad()
                _, logits = self.model(gen_imgs)
                loss = criterion(logits, y_batch)
                loss.backward()
                self.global_model_optimizer.step()
                
                epoch_loss += loss.item()
            
            self.logger.log(f"Epoch {epoch} Loss: {epoch_loss / len(train_loader):.4f}")
        
    def distribute_model(self):
        for client in self.selected_clients:
            group_name = client.group_name

            if group_name in self.global_gen_states and group_name in self.global_dis_states:
                client.generator.load_state_dict(self.global_gen_states[group_name])
                client.discriminator.load_state_dict(self.global_dis_states[group_name])


    def aggregate_weights(self, weights_list):
        """
        FedAvg aggregation for Generator
        """
        total_samples = sum([w[0] for w in weights_list])
        avg_params = OrderedDict()
        
        for name in weights_list[0][1].keys():
            avg_params[name] = torch.zeros_like(weights_list[0][1][name], dtype=torch.float32)
            
            for num_samples, params in weights_list:
                avg_params[name] += params[name] * (num_samples / total_samples)
                
        return avg_params


    def test_global_inference_model(self, epoch=None):
        if self.model is None:
            return
        
        self.model.eval()
        self.model.to(self.device)
        
        dataset_correct = defaultdict(int)
        dataset_total = defaultdict(int)

        class_correct = defaultdict(lambda: defaultdict(int))
        class_total = defaultdict(lambda: defaultdict(int))

        with torch.no_grad():
            for client in self.clients:
                mapping_name = client.group_name
                d_name = client.dataset_name

                if mapping_name not in self.local_id_to_global_id:
                    continue

                local_to_global_map = self.local_id_to_global_id[mapping_name]

                for x, y in client.test_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)

                    _, logits = self.model(x)
                    preds = torch.argmax(logits, dim=1)

                    y_np = y.cpu().numpy()
                    y_global = []
                    valid_indices = []

                    for i, label in enumerate(y_np):
                        label = int(label)

                        if label in local_to_global_map:
                            y_global.append(local_to_global_map[label])
                            valid_indices.append(i)

                    if len(valid_indices) == 0:
                        continue

                    y_global_tensor = torch.tensor(y_global, dtype=torch.long, device=self.device)
                    valid_preds = preds[valid_indices]
                    correct_mask = valid_preds == y_global_tensor

                    dataset_correct[d_name] += correct_mask.sum().item()
                    dataset_total[d_name] += len(valid_indices)

                    for valid_idx, original_idx in enumerate(valid_indices):
                        local_label = int(y_np[original_idx])
                        class_total[d_name][local_label] += 1

                        if correct_mask[valid_idx].item():
                            class_correct[d_name][local_label] += 1

        total_correct = 0
        total_samples = 0

        for d_name in dataset_total.keys():
            correct = dataset_correct[d_name]
            total = dataset_total[d_name]
            total_correct += correct
            total_samples += total
            
            acc = (correct / total) * 100.0 if total > 0 else 0.0
            self.logger.log(f"[Server] Global Inference Model Acc ({d_name}) (Round {self.glob_iter + 1}): {acc:.2f}% (Tested on {total} samples)")
            
            csv_path = os.path.join(self.logger.log_dir, f"global_model_acc_{d_name}.csv")
            file_exists = os.path.isfile(csv_path)
            
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    #writer.writerow(["Round", "Accuracy"])
                    writer.writerow(["Round", "Epoch", "Accuracy"])
                #writer.writerow([self.glob_iter + 1, round(acc, 2)])
                writer.writerow([
                    self.glob_iter + 1,
                    epoch,
                    round(acc, 2)
                ])

        mix_acc = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
        self.logger.log(f"[Server] Global Inference Model Acc (Mix) (Round {self.glob_iter + 1}): {mix_acc:.2f}% (Tested on {total_samples} samples)")
        
        csv_path_mix = os.path.join(self.logger.log_dir, "global_model_acc_mix.csv")
        file_exists_mix = os.path.isfile(csv_path_mix)
        
        with open(csv_path_mix, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists_mix:
                #writer.writerow(["Round", "Accuracy"])
                writer.writerow(["Round", "Epoch", "Accuracy"])
            #writer.writerow([self.glob_iter + 1, round(mix_acc, 2)])
            writer.writerow([
                self.glob_iter + 1,
                epoch,
                round(mix_acc, 2)
            ])

        csv_path_class = os.path.join(self.logger.log_dir, "global_model_class_acc.csv")
        file_exists_class = os.path.isfile(csv_path_class)
        with open(csv_path_class, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists_class:
                writer.writerow(["Round", "Dataset", "Local_Class", "Accuracy"])
            
            for d_name in class_total.keys():
                for local_lbl in sorted(class_total[d_name].keys()):
                    tot = class_total[d_name][local_lbl]
                    cor = class_correct[d_name][local_lbl]
                    cls_acc = (cor / tot) * 100.0 if tot > 0 else 0.0
                    writer.writerow([self.glob_iter + 1, d_name, local_lbl, round(cls_acc, 2)])


    def save_model(self, global_round=0, fname='checkpoints.pth'):
        self.logger.log("Saving checkpoints ...")
        client_label_distributions = {}
        client_label_space_meta = {}
        client_metadata = {}

        for client in self.clients:
            unique_labels = set()

            for _, labels in client.train_loader:
                unique_labels.update(labels.tolist())

            valid_labels = sorted(unique_labels)
            client_name = f"client_{client.id}"

            client_label_distributions[client.id] = valid_labels
            client_label_space_meta[client_name] = list(client.class_name_set)

            client_metadata[client.id] = {
                "client_id": client.id,
                "dataset_name": client.dataset_name,
                "group_name": client.group_name,
                "valid_labels": valid_labels,
                "class_names": list(client.class_name_set),
                "model_name": client.model_name
            }

        group_metadata = {}

        for group_name, group_clients in self.client_groups.items():
            group_metadata[group_name] = {
                "group_name": group_name,
                "client_ids": [client.id for client in group_clients],
                "datasets": sorted({client.dataset_name for client in group_clients}),
                "label_space_meta": self.group_label_space_meta.get(group_name, [])
            }


        checkpoint = {
            'client_label_distributions': client_label_distributions,
            'global_registry': self.local_id_to_global_id,
            'label_space_meta': client_label_space_meta,
            "group_label_space_meta": self.group_label_space_meta,
            "client_groups": {client.id: client.group_name for client in self.clients},
            "client_metadata": client_metadata,
            "group_metadata": group_metadata,
            'global_feature_dim': self.global_feature_dim,
            'exp_conf': self.exp_conf,
            'args': {
                'num_train_mnist': self.args.num_train_mnist,
                'num_train_emnist': self.args.num_train_emnist,
                'num_train_fashionmnist': self.args.num_train_fashionmnist,
                'num_train_cifar10': self.args.num_train_cifar10,
                'num_train_cifar100': self.args.num_train_cifar100,
                'num_new_clients': self.args.num_new_clients,
                'seed': self.args.seed,
                'device': str(self.args.device),
                'algorithm': self.args.algorithm,
            },
        }

        server_save_path = os.path.join(self.logger.log_dir, 'server_'+'checkpoints'+f'_{global_round}.pth')
        torch.save(checkpoint, server_save_path)
        self.logger.log(f"[Server] Checkpoint saved to {server_save_path}")

        global_gan_dir = os.path.join(self.logger.log_dir, f"global_gans_round_{global_round}")
        os.makedirs(global_gan_dir, exist_ok=True)

        for group_name, gen_state in self.global_gen_states.items():
            group_clients = self.client_groups.get(group_name, [])
            client_ids = [client.id for client in group_clients]
            datasets = sorted({
                client.dataset_name
                for client in group_clients
            })

            global_gan_checkpoint = {
                "global_round": global_round,
                "aggregate_method": self.aggregate_method,
                "group_name": group_name,
                "client_ids": client_ids,
                "datasets": datasets,
                "label_space_meta": self.group_label_space_meta.get(
                    group_name,
                    []
                ),
                "generator": gen_state,
                "discriminator": self.global_dis_states.get(group_name)
            }

            global_gan_path = os.path.join(
                global_gan_dir,
                f"global_GAN_{group_name}.pth"
            )

            torch.save(global_gan_checkpoint, global_gan_path)

            self.logger.log(
                f"[Server] Global GAN saved: "
                f"group={group_name} | datasets={datasets} | "
                f"clients={client_ids} | path={global_gan_path}"
            )

        local_gan_dir  = os.path.join(self.logger.log_dir, f"local_gans_round_{global_round}")
        os.makedirs(local_gan_dir , exist_ok=True)

        for client in self.clients:
            local_gan_checkpoint = {
                "global_round": global_round,
                "client_id": client.id,
                "dataset_name": client.dataset_name,
                "group_name": client.group_name,
                "valid_labels": client_label_distributions[client.id],
                "class_names": list(client.class_name_set),
                "generator": client.generator.state_dict(),
                "discriminator": client.discriminator.state_dict()
            }

            local_gan_path = os.path.join(
                local_gan_dir,
                f"local_GAN_{client.group_name}_"
                f"{client.dataset_name}_c{client.id}.pth"
            )

            torch.save(local_gan_checkpoint, local_gan_path)

        client_model_dir = os.path.join(self.logger.log_dir, f"clients_round_{global_round}_checkpoints")
        os.makedirs(client_model_dir, exist_ok=True)

        for client in self.clients:
            arch_name = client.model_name

            local_model_checkpoint = {
                "global_round": global_round,
                "client_id": client.id,
                "dataset_name": client.dataset_name,
                "group_name": client.group_name,
                "valid_labels": client_label_distributions[client.id],
                "class_names": list(client.class_name_set),
                "model_name": arch_name,
                "model": client.model.state_dict()
            }

            client_model_path = os.path.join(
                client_model_dir,
                f"client_model_{client.group_name}_"
                f"{client.dataset_name}_c{client.id}_{arch_name}.pth"
            )

            torch.save(local_model_checkpoint, client_model_path)