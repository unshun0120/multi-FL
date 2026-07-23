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
            #self.distribute_model()
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

        if (self.glob_iter + 1) == self.start_mapping_epoch: 

            clear_image_caches()
            dataset_clients_dict = {}   
            active_datasets = []
            dataset_label_space_meta = {}
            generators = {}
            valid_labels_dict = {}
            
            for client in self.clients: 
                d_name = f"client_{client.id}"

                dataset_clients_dict[d_name] = [client.model.to(self.device)]
                dataset_label_space_meta[d_name] = list(client.class_name_set)

                client.generator.to(self.device)
                client.generator.eval()
                generators[d_name] = client.generator

                unique_labels = set()
                for _, labels in client.train_loader:
                    unique_labels.update(labels.tolist())

                valid_labels_dict[d_name] = sorted(unique_labels)
                active_datasets.append(d_name)

                self.logger.log(f"{d_name} | group={client.group_name} | dataset={client.dataset_name} | labels={valid_labels_dict[d_name]}")

            active_datasets = list(dataset_clients_dict.keys())

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

                for group_name, group_clients in self.client_groups.items():
                    group_valid_labels = set()

                    for client in group_clients:
                        client_name = f"client_{client.id}"

                        if valid_labels_dict is not None and client_name in valid_labels_dict:
                            group_valid_labels.update(valid_labels_dict[client_name])
                        else:
                            group_valid_labels.update(range(len(dataset_label_space_meta[client_name])))

                    group_local_to_global = {}

                    for local_id in sorted(group_valid_labels):
                        group_local_to_global[local_id] = current_gid
                        current_gid += 1

                    for client in group_clients:
                        client_name = f"client_{client.id}"
                        self.local_id_to_global_id[client_name] = {}

                        if valid_labels_dict is not None and client_name in valid_labels_dict:
                            client_valid_labels = valid_labels_dict[client_name]
                        else:
                            client_valid_labels = range(len(dataset_label_space_meta[client_name]))

                        for local_id in client_valid_labels:
                            if local_id in group_local_to_global:
                                self.local_id_to_global_id[client_name][local_id] = group_local_to_global[local_id]

                    self.logger.log(
                        f"[Independent] {group_name} | "
                        f"clients={[client.id for client in group_clients]} | "
                        f"global_ids={group_local_to_global}"
                    )

                global_to_local_mapping(
                    self.local_id_to_global_id,
                    logger=self.logger,
                    label_space_meta=dataset_label_space_meta
                )

                for client in self.clients:
                    if client.dataset_name in ["MNIST", "EMNIST"]:
                        client_name = f"client_{client.id}"
                        self.logger.log(f"[Check Mapping] {client.dataset_name} | {client_name} | local 0 -> global {self.local_id_to_global_id[client_name].get(0)}")

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

        # for group_name, group_clients in self.client_groups.items(): 
        #     self.logger.log(f"[Global Train] Generating images from {group_name}")
        #     for client in group_clients:
        #         d_name = f"client_{client.id}"
        #         mapping = self.local_id_to_global_id.get(d_name, {})
        #         if not mapping: continue
        #         valid_labels = set()
        #         for _, labels in client.train_loader:
        #             valid_labels.update(labels.tolist())
        #         gen = client.generator.to(self.device)
        #         gen.eval()
        #         for local_id, global_id in mapping.items():
        #             if local_id not in valid_labels: continue
        #             z = torch.randn(self.global_samples_per_class, noise_dim, device=self.device)
        #             y_local = torch.full((self.global_samples_per_class,), local_id, dtype=torch.long, device=self.device)
        #             with torch.no_grad():
        #                 x_gen = gen(z, y_local)
        #             for i in range(self.global_samples_per_class):
        #                 dataset_x.append(x_gen[i].cpu())
        #                 dataset_y.append(global_id)

        for group_name, group_clients in self.client_groups.items():
            self.logger.log(f"[Global Train] Generating images from {group_name}")

            sources_by_global_id = defaultdict(list)

            for client in group_clients:
                d_name = f"client_{client.id}"
                mapping = self.local_id_to_global_id.get(d_name, {})

                if not mapping:
                    continue

                valid_labels = set()
                for _, labels in client.train_loader:
                    valid_labels.update(labels.tolist())

                client.generator.to(self.device)
                client.generator.eval()

                for local_id, global_id in mapping.items():
                    if local_id in valid_labels:
                        sources_by_global_id[global_id].append((client, local_id))

            group_generated_count = 0

            for global_id, sources in sources_by_global_id.items():
                num_sources = len(sources)
                samples_per_source = self.global_samples_per_class // num_sources
                remainder = self.global_samples_per_class % num_sources

                for source_idx, (client, local_id) in enumerate(sources):
                    num_samples = samples_per_source + (1 if source_idx < remainder else 0)

                    if num_samples == 0:
                        continue

                    gen = client.generator
                    z = torch.randn(num_samples, noise_dim, device=self.device)
                    y_local = torch.full((num_samples,), local_id, dtype=torch.long, device=self.device)

                    with torch.no_grad():
                        x_gen = gen(z, y_local)

                    dataset_x.extend(x_gen.detach().cpu())
                    dataset_y.extend([global_id] * num_samples)
                    group_generated_count += num_samples

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


    # def test_global_inference_model(self, epoch=None):
    #     if self.model is None:
    #         return
        
    #     self.model.eval()
    #     self.model.to(self.device)
        
    #     dataset_correct = defaultdict(int)
    #     dataset_total = defaultdict(int)

    #     class_correct = defaultdict(lambda: defaultdict(int))
    #     class_total = defaultdict(lambda: defaultdict(int))

    #     with torch.no_grad():
    #         for client in self.clients:
    #             mapping_name = f"client_{client.id}"
    #             d_name = client.dataset_name

    #             if mapping_name not in self.local_id_to_global_id:
    #                 continue

    #             local_to_global_map = self.local_id_to_global_id[mapping_name]

    #             for x, y in client.test_loader:
    #                 x = x.to(self.device)
    #                 y = y.to(self.device)

    #                 _, logits = self.model(x)
    #                 preds = torch.argmax(logits, dim=1)

    #                 y_np = y.cpu().numpy()
    #                 y_global = []
    #                 valid_indices = []

    #                 for i, label in enumerate(y_np):
    #                     label = int(label)

    #                     if label in local_to_global_map:
    #                         y_global.append(local_to_global_map[label])
    #                         valid_indices.append(i)

    #                 if len(valid_indices) == 0:
    #                     continue

    #                 y_global_tensor = torch.tensor(y_global, dtype=torch.long, device=self.device)
    #                 valid_preds = preds[valid_indices]
    #                 correct_mask = valid_preds == y_global_tensor

    #                 dataset_correct[d_name] += correct_mask.sum().item()
    #                 dataset_total[d_name] += len(valid_indices)

    #                 for valid_idx, original_idx in enumerate(valid_indices):
    #                     local_label = int(y_np[original_idx])
    #                     class_total[d_name][local_label] += 1

    #                     if correct_mask[valid_idx].item():
    #                         class_correct[d_name][local_label] += 1

    #     total_correct = 0
    #     total_samples = 0

    #     for d_name in dataset_total.keys():
    #         correct = dataset_correct[d_name]
    #         total = dataset_total[d_name]
    #         total_correct += correct
    #         total_samples += total
            
    #         acc = (correct / total) * 100.0 if total > 0 else 0.0
    #         self.logger.log(f"[Server] Global Inference Model Acc ({d_name}) (Round {self.glob_iter + 1}): {acc:.2f}% (Tested on {total} samples)")
            
    #         csv_path = os.path.join(self.logger.log_dir, f"global_model_acc_{d_name}.csv")
    #         file_exists = os.path.isfile(csv_path)
            
    #         with open(csv_path, mode='a', newline='') as f:
    #             writer = csv.writer(f)
    #             if not file_exists:
    #                 #writer.writerow(["Round", "Accuracy"])
    #                 writer.writerow(["Round", "Epoch", "Accuracy"])
    #             #writer.writerow([self.glob_iter + 1, round(acc, 2)])
    #             writer.writerow([
    #                 self.glob_iter + 1,
    #                 epoch,
    #                 round(acc, 2)
    #             ])

    #     mix_acc = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    #     self.logger.log(f"[Server] Global Inference Model Acc (Mix) (Round {self.glob_iter + 1}): {mix_acc:.2f}% (Tested on {total_samples} samples)")
        
    #     csv_path_mix = os.path.join(self.logger.log_dir, "global_model_acc_mix.csv")
    #     file_exists_mix = os.path.isfile(csv_path_mix)
        
    #     with open(csv_path_mix, mode='a', newline='') as f:
    #         writer = csv.writer(f)
    #         if not file_exists_mix:
    #             #writer.writerow(["Round", "Accuracy"])
    #             writer.writerow(["Round", "Epoch", "Accuracy"])
    #         #writer.writerow([self.glob_iter + 1, round(mix_acc, 2)])
    #         writer.writerow([
    #             self.glob_iter + 1,
    #             epoch,
    #             round(mix_acc, 2)
    #         ])

    #     csv_path_class = os.path.join(self.logger.log_dir, "global_model_class_acc.csv")
    #     file_exists_class = os.path.isfile(csv_path_class)
    #     with open(csv_path_class, mode='a', newline='') as f:
    #         writer = csv.writer(f)
    #         if not file_exists_class:
    #             writer.writerow(["Round", "Dataset", "Local_Class", "Accuracy"])
            
    #         for d_name in class_total.keys():
    #             for local_lbl in sorted(class_total[d_name].keys()):
    #                 tot = class_total[d_name][local_lbl]
    #                 cor = class_correct[d_name][local_lbl]
    #                 cls_acc = (cor / tot) * 100.0 if tot > 0 else 0.0
    #                 writer.writerow([self.glob_iter + 1, d_name, local_lbl, round(cls_acc, 2)])


    def test_global_inference_model(self, epoch=None):
        if self.model is None:
            return
        
        self.model.eval()
        self.model.to(self.device)
        
        dataset_correct = defaultdict(int)
        dataset_total = defaultdict(int)

        class_correct = defaultdict(lambda: defaultdict(int))
        class_total = defaultdict(lambda: defaultdict(int))

        dataset_to_mapping = {}

        for group_name, group_clients in self.client_groups.items():
            datasets = {client.dataset_name for client in group_clients}

            if len(datasets) != 1:
                raise ValueError(
                    f"{group_name} contains multiple datasets: {datasets}. "
                    f"Cannot build one public-test mapping."
                )

            d_name = next(iter(datasets))

            if d_name not in dataset_to_mapping:
                dataset_to_mapping[d_name] = {}

            for client in group_clients:
                client_name = f"client_{client.id}"

                if client_name not in self.local_id_to_global_id:
                    continue

                client_mapping = self.local_id_to_global_id[client_name]

                for local_id, global_id in client_mapping.items():
                    if local_id in dataset_to_mapping[d_name]:
                        old_global_id = dataset_to_mapping[d_name][local_id]

                        if old_global_id != global_id:
                            raise ValueError(
                                f"Inconsistent mapping for {d_name} local label {local_id}: "
                                f"{old_global_id} vs {global_id} from {client_name}"
                            )

                    dataset_to_mapping[d_name][local_id] = global_id

        for d_name, mapping in dataset_to_mapping.items():
            self.logger.log(
                f"[Public Test Mapping] {d_name}: "
                f"{dict(sorted(mapping.items()))}"
            )

        with torch.no_grad():
            for d_name, test_loader in self.public_test_loader.items():
                if d_name not in dataset_to_mapping:
                    self.logger.log(f"[Test] Skip {d_name}: no client-level mapping found.")
                    continue

                local_to_global_map = dataset_to_mapping[d_name]

                for x, y in test_loader:
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

                    y_global_tensor = torch.tensor(
                        y_global,
                        dtype=torch.long,
                        device=self.device
                    )

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

        dataset_classifiers = {}
        for ls_id, model_dict in self.global_models.items():
            if 'classifier' in model_dict:
                dataset_classifiers[ls_id] = model_dict['classifier'].state_dict()

        client_label_distributions = {}
        for client in self.clients:
            unique_labels = set()
            for _, labels in client.train_loader:
                unique_labels.update(labels.tolist())
            client_label_distributions[client.id] = list(unique_labels)

        client_label_space_meta = {f"client_{client.id}": list(client.class_name_set) for client in self.clients}

        checkpoint = {
            'client_label_distributions': client_label_distributions,
            'global_registry': self.local_id_to_global_id,
            'label_space_meta': client_label_space_meta,
            "client_groups": {client.id: client.group_name for client in self.clients},
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

        gen_dir = os.path.join(self.logger.log_dir, f"local_gans_round_{global_round}")
        os.makedirs(gen_dir, exist_ok=True)

        for client in self.clients:
            gan_checkpoint = {
                "global_round": global_round,
                "client_id": client.id,
                "dataset_name": client.dataset_name,
                "group_name": client.group_name,
                "generator": client.generator.state_dict(),
                "discriminator": client.discriminator.state_dict()
            }

            gan_save_path = os.path.join(gen_dir, f"client_GAN_{client.dataset_name}_c{client.id}.pth")
            torch.save(gan_checkpoint, gan_save_path)
            self.logger.log(f"[Server] Local GAN saved: {gan_save_path}")

        clients_dir = os.path.join(self.logger.log_dir, f'clients_round_{global_round}_checkpoints')
        os.makedirs(clients_dir, exist_ok=True)

        for client in self.clients:
            arch_name = getattr(client, 'model_name', 'Unknown')
            client_path = os.path.join(clients_dir, f'client_model_{client.dataset_name}_c{client.id}_{arch_name}.pth')
            torch.save(client.model.state_dict(), client_path)
            
        self.logger.log(f"[Server] All {len(self.clients)} clients saved in {clients_dir}/")