import csv

import torch
from torch.nn import *
from torch.optim import *
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
from collections import OrderedDict
import copy
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import math

from .client import Node
from utils.plotting import plot_accuracy_curves
from utils.nets import ConditionalGenerator, ConditionalImageGenerator, Classifier, DCGANGenerator, ContextUnet, DDPM
from utils.loss import Gen_DiversityLoss, total_variation_loss, BNSM_Hook, get_bn_loss
from utils.train_utils import evaluate_model
from utils.pacfl_utils import calculating_adjacency, hierarchical_clustering
from label_mapping.label_mapping_utils import (
    label_mapping, 
    global_to_local_mapping, 
    get_gen_images, 
    get_real_images
)
#from utils.plotting import plot_accuracy_curves

class Server(Node):
    def __init__(self, clients, **exp_conf):
        super(Server, self).__init__(**exp_conf)

        self.exp_conf = exp_conf.get('exp_conf', {})

        # experiment config
        self.sample_frac = exp_conf.get('sample_frac', 1.0) 
        self.global_rounds = exp_conf.get('global_rounds', 100)
        self.metric_type = exp_conf.get('metric_type', 'accuracy')
        self.global_feature_dim = exp_conf.get('global_feature_dim', 256)
        self.index_matching = exp_conf.get('index_matching', 'ours')

        self.global_model_epochs = exp_conf.get('global_model_epochs', 1) 
        self.global_model_optim_name =  exp_conf.get('global_model_optim', 'Adam')
        self.global_model_optim_lr = exp_conf.get('global_model_optim_lr', 1e-3)
        self.global_samples_per_class = exp_conf.get('global_samples_per_class', 1)

        # initial variables
        self.clients = clients
        self.num_clients = len(clients)
        self.selected_clients_ids = []
        self.selected_clients = []

        # Value: {'feature_extractor': state_dict, 'classifier': state_dict}
        self.global_models = {}
        self.global_ddpm_states = {} 

        # accuracy
        self.dataset_acc_history = defaultdict(list)

    def run(self):
        self.logger.log("")
        self.logger.log("=" * 50)
        self.logger.log(f"Start {self.global_rounds} rounds training by {self.algorithm}")

        for r in range(self.global_rounds):
            self.glob_iter = r

            self.sample_clients()
            self.distribute_model()
            self.local_update()

            if (r + 1) % self.test_interval == 0:
                self.evaluate_private()
                #self.record_metric()

            self.aggregate()

        # self.save_metric()

        self.save_model()
        plot_accuracy_curves(self.dataset_acc_history, self.logger.log_dir, self.args, self.global_rounds, self.dirichlet_alpha)


    def sample_clients(self):
        """Select some fraction of all clients."""
        # sample clients randomly
        num_sampled_clients = max(int(self.sample_frac * self.num_clients), 1)
        self.selected_clients_ids = sorted(np.random.choice(range(self.num_clients),
                                                            size=num_sampled_clients,
                                                            replace=False).tolist())
        
        self.logger.log(f'Selected client ids: {self.selected_clients_ids}')

        self.selected_clients = [self.clients[idx] for idx in self.selected_clients_ids]

        for client in self.selected_clients:
            client.glob_iter = self.glob_iter

    def distribute_model(self):
        for client in self.selected_clients:
            if self.dataset_name not in self.global_models:
                continue

            global_part = self.global_models[self.dataset_name]

            if 'classifier' in global_part:
                client.model.classifier.load_state_dict(global_part['classifier'].state_dict())

            if self.heterogeneous is False and 'feature_extractor' in global_part:
                client.model.feature_extractor.load_state_dict(global_part['feature_extractor'].state_dict())


    def local_update(self):
        self.logger.log(f"--- Round {self.glob_iter + 1} ---")
        for client in tqdm(self.selected_clients):
            client.update()


    def aggregate(self):
        if self.aggregate_method == 'pacfl':
            self.logger.log("[Server] Aggregating by PACFL ...")
            n_basis = 3
            cluster_alpha = 15
            
            U_clients = []
            for client in self.selected_clients:
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
                u_temp = u_temp / np.linalg.norm(u_temp, ord=2, axis=0) # Normalize
                U_clients.append(copy.deepcopy(u_temp[:, 0:n_basis]))

            adj_mat = calculating_adjacency(list(range(len(self.selected_clients))), U_clients)
            clusters_idx = hierarchical_clustering(copy.deepcopy(adj_mat), thresh=cluster_alpha, linkage='average')
            
            groups = defaultdict(list)
            for cluster_id, client_indices in enumerate(clusters_idx):
                cluster_name = f"Cluster_{cluster_id}"
                
                ds_in_cluster = set()
                for idx in client_indices:
                    client = self.selected_clients[idx]
                    client.group_name = cluster_name
                    groups[cluster_name].append(client)
                    ds_in_cluster.add(client.dataset_name)
                    
                self.logger.log(f'   -> {cluster_name}: {len(client_indices)} Users | Datasets included: {ds_in_cluster}')

                if cluster_name not in self.label_space_meta:
                    c_set = set()
                    for c in groups[cluster_name]:
                        c_set.update(c.class_name_set)
                    self.label_space_meta[cluster_name] = list(c_set)

        elif self.aggregate_method == 'dataset_name':
            self.logger.log("[Server] Aggregating by Dataset Name ...")
            groups = defaultdict(list)
            for client in self.selected_clients:
                d_name = client.dataset_name  
                client.group_name = d_name
                groups[d_name].append(client)

                if d_name not in self.label_space_meta:
                    self.label_space_meta[d_name] = client.class_name_set

        print(f"[Server] Aggregating from {len(self.selected_clients)} clients (grouped by {len(groups)} datasets)...") 
        
        if self.index_matching == 'ddpm':
            for d_name, group_clients in groups.items():
                if hasattr(group_clients[0], 'ddpm') and group_clients[0].ddpm is not None:
                    ddpm_msg_list = [
                        (client.num_samples, client.ddpm.state_dict())
                        for client in group_clients
                    ]
                    w_ddpm = self.avg_weights(ddpm_msg_list)
                    self.global_ddpm_states[d_name] = w_ddpm

        # for d_name, group_clients in groups.items():
        #     if d_name not in self.global_models:
        #         self.global_models[d_name] = {}

        #     if self.heterogeneous:
        #         # aggregate clients generic classifier
        #         msg_list = [(client.num_samples, client.model.classifier.state_dict())
        #                     for client in group_clients]
        #         w_cls = self.avg_weights(msg_list)

        #         num_classes = w_cls['weight'].shape[0]
        #         input_dim = w_cls['weight'].shape[1]

        #         cls_model = Classifier(input_dim, num_classes).to(self.device)
        #         cls_model.load_state_dict(w_cls)
        #         cls_model.eval()

        #         self.global_models[d_name]['classifier'] = cls_model
        #     else:
        #     # if not heterogeneous, aggregate feature_extractor of clients
        #         msg_list = [(client.num_samples, client.model.state_dict())
        #                     for client in group_clients]
        #         w_global = self.avg_weights(msg_list)

        #         full_model = copy.deepcopy(group_clients[0].model).to(self.device)
        #         full_model.load_state_dict(w_global)
        #         full_model.eval()
                
        #         self.global_models[d_name]['full_model'] = full_model
                
        #         self.global_models[d_name]['classifier'] = full_model.classifier
        #         self.global_models[d_name]['feature_extractor'] = full_model.feature_extractor

        if (self.glob_iter + 1) == self.start_mapping_epoch: 
            if self.index_matching == 'class_name':
                self.class_name_label_mapping()
            elif self.index_matching == 'real_image':
                self.real_img_label_mapping()
            elif self.index_matching == 'gan':
                self.gan_label_mapping()
            elif self.index_matching == 'ddpm':
                self.ddpm_label_mapping()
            elif self.index_matching == 'independent':
                self.independent_label_mapping()
            elif self.index_matching == 'identical':
                self.identical_label_mapping()

    def class_name_label_mapping(self):
        self.logger.log("[Server] Performing Name-Based Label Mapping...")
        
        all_class_names = set()
        for d_name in self.label_space_meta:
             for name in self.label_space_meta[d_name]:
                 all_class_names.add(name)
        
        sorted_names = sorted(list(all_class_names))
        name_to_gid = {name: idx for idx, name in enumerate(sorted_names)}
        
        self.local_id_to_global_id = {}
        
        for d_name in self.label_space_meta:
            self.local_id_to_global_id[d_name] = {}
            current_names = self.label_space_meta[d_name]
            for l_id, name in enumerate(current_names):
                if name in name_to_gid:
                     self.local_id_to_global_id[d_name][l_id] = name_to_gid[name]

        self.logger.log(f"[Server] Name-based mapping completed. Found {len(sorted_names)} unique global classes.")
        
        self.logger.log("\n=========================================================================================================")
        self.logger.log(f"{'Global ID':<10} | {'Class Name':<20} | {'Mapped Datasets (Local ID)'}")
        self.logger.log("---------------------------------------------------------------------------------------------------------")
        
        gid_to_sources = defaultdict(list)
        for d_name, mapping in self.local_id_to_global_id.items():
            for l_id, gid in mapping.items():
                gid_to_sources[gid].append(f"{d_name}({l_id})")
                
        for gid in range(len(sorted_names)):
            c_name = sorted_names[gid]
            sources = ", ".join(gid_to_sources[gid])
            self.logger.log(f"{gid:<10} | {c_name:<20} | {sources}")
        self.logger.log("=========================================================================================================\n")
        

    def real_img_label_mapping(self):
        self.logger.log("[Server] Performing Real Image Label Mapping (Client-Level)...")

        dataset_clients_dict = defaultdict(list)
        active_dataset_ids = []
        dataset_test_loaders = {}

        for client in self.selected_clients:
            d_name = client.dataset_name
            #d_name = getattr(client, 'group_name', client.dataset_name) 
            dataset_clients_dict[d_name].append(client.model.to(self.device).eval())
            
            if d_name not in active_dataset_ids:
                active_dataset_ids.append(d_name)
                dataset_test_loaders[d_name] = client.test_loader

        dataset_label_space_meta = {d_name: self.label_space_meta[d_name] for d_name in active_dataset_ids}

        entropy_ratio = self.exp_conf.get('entropy_ratio', 1.0) 
        use_new_entropy_method = self.exp_conf.get('use_new_entropy_method', False)

        res_mapping = label_mapping(
            get_images_func=get_real_images,
            dataset_ids=active_dataset_ids,          
            clients_dict=dataset_clients_dict,       
            label_space_meta=dataset_label_space_meta,
            entropy_ratio=entropy_ratio,
            use_new_entropy_method=use_new_entropy_method,
            logger=self.logger,
            valid_labels_dict=None,                  
            args=self.args,
            test_loaders=dataset_test_loaders
        )

        self.local_id_to_global_id = res_mapping
        global_to_local_mapping(self.local_id_to_global_id, logger=self.logger, label_space_meta=dataset_label_space_meta)

    def gan_label_mapping(self):
        self.logger.log("[Server] Performing GAN Generator Label Mapping (Client-Level)...")
        
        gens_dict = {} 

        dataset_clients_dict = defaultdict(list)
        active_dataset_ids = []
        
        for client in self.selected_clients:
            #d_name = client.dataset_name
            d_name = getattr(client, 'group_name', client.dataset_name)
            dataset_clients_dict[d_name].append(client.model.to(self.device).eval())
            if d_name not in active_dataset_ids:
                active_dataset_ids.append(d_name)

        dataset_label_space_meta = {d_name: self.label_space_meta[d_name] for d_name in active_dataset_ids}

        gens_dict = {} 
        
        for d_name in active_dataset_ids:
            if not hasattr(self, 'global_gen_states') or d_name not in self.global_gen_states:
                self.logger.log(f"[Server] WARNING: global_gen_states for {d_name} not found! Skipping...")
                continue
                
            num_classes = len(dataset_label_space_meta[d_name])
            gen = DCGANGenerator(
                num_classes=num_classes, 
                noise_dim=self.exp_conf.get('gen_noise_dim', 128), 
                img_size=self.exp_conf.get('img_size', 32), 
                channels=self.exp_conf.get('channels', 3)
            ).to(self.device)
            
            gen.load_state_dict(self.global_gen_states[d_name])
            gen.eval()
            gens_dict[d_name] = gen

        entropy_ratio = self.exp_conf.get('entropy_ratio', 1.0)
        use_new_entropy_method = self.exp_conf.get('use_new_entropy_method', False)

        res_mapping = label_mapping(
            get_images_func=get_gen_images,
            dataset_ids=active_dataset_ids,    
            clients_dict=dataset_clients_dict,
            label_space_meta=dataset_label_space_meta,
            entropy_ratio=entropy_ratio,
            use_new_entropy_method=use_new_entropy_method,
            logger=self.logger,
            valid_labels_dict=None,
            args=self.args,
            gen_dict=gens_dict
        )

        self.local_id_to_global_id = res_mapping
        global_to_local_mapping(self.local_id_to_global_id, logger=self.logger, label_space_meta=dataset_label_space_meta)

    
    def ddpm_label_mapping(self):
        self.logger.log("[Server] Performing DDPM Label Mapping...")
        
        gens_dict = {} 

        dataset_clients_dict = defaultdict(list)
        active_dataset_ids = []
        
        for client in self.selected_clients:
            d_name = client.dataset_name
            #d_name = getattr(client, 'group_name', client.dataset_name)
            dataset_clients_dict[d_name].append(client.model.to(self.device).eval())
            if d_name not in active_dataset_ids:
                active_dataset_ids.append(d_name)

        dataset_label_space_meta = {d_name: self.label_space_meta[d_name] for d_name in active_dataset_ids}

        ddpm_dict = {}
        for d_name in active_dataset_ids:
            if d_name in self.global_ddpm_states:
                nn_model = ContextUnet(
                    in_channels=self.exp_conf.get('channels', 3), 
                    n_feat=self.exp_conf.get('n_feat', 64), 
                    n_classes=len(self.label_space_meta[d_name])
                ).to(self.device)
                
                ddpm = DDPM(
                    nn_model=nn_model, 
                    betas=(1e-4, 0.02), 
                    n_T=1000, 
                    device=self.device, 
                    drop_prob=0.1
                ).to(self.device)

                ddpm.load_state_dict(self.global_ddpm_states[d_name])
                ddpm.eval()
                ddpm_dict[d_name] = ddpm

        entropy_ratio = self.exp_conf.get('entropy_ratio', 1.0)
        use_new_entropy_method = self.exp_conf.get('use_new_entropy_method', False)

        res_mapping = label_mapping(
            get_images_func=get_gen_images,
            dataset_ids=active_dataset_ids,    
            clients_dict=dataset_clients_dict,
            label_space_meta=dataset_label_space_meta,
            entropy_ratio=entropy_ratio,
            use_new_entropy_method=use_new_entropy_method,
            logger=self.logger,
            valid_labels_dict=None,
            args=self.args,
            gen_dict=ddpm_dict
        )

        self.local_id_to_global_id = res_mapping
        global_to_local_mapping(self.local_id_to_global_id, logger=self.logger, label_space_meta=dataset_label_space_meta)
    
    def independent_label_mapping(self):
        self.logger.log("[Server] Performing Independent Label Mapping (All Different)...")
        self.local_id_to_global_id = {}
        current_gid = 0
        for d_name, class_names in self.label_space_meta.items():
            self.local_id_to_global_id[d_name] = {}
            for l_id in range(len(class_names)):
                self.local_id_to_global_id[d_name][l_id] = current_gid
                current_gid += 1

        global_to_local_mapping(self.local_id_to_global_id, logger=self.logger, label_space_meta=self.label_space_meta)


    def identical_label_mapping(self):
        self.logger.log("[Server] Performing Identical Label Mapping (All Same by Local ID)...")
        self.local_id_to_global_id = {}
        for d_name, class_names in self.label_space_meta.items():
            self.local_id_to_global_id[d_name] = {}
            for l_id in range(len(class_names)):
                self.local_id_to_global_id[d_name][l_id] = l_id
                
        global_to_local_mapping(self.local_id_to_global_id, logger=self.logger, label_space_meta=self.label_space_meta)


    @staticmethod
    def avg_weights(nk_and_wk):
        """
        n_k_and_weights: [..., (n_k, w_k), ....], where n_k is the number of samples w_k is weight.
        """
        averaged_weights = OrderedDict()

        n_sum = sum([n_k for n_k, _ in nk_and_wk])
        for i, (n_k, w_k) in enumerate(nk_and_wk):
            for key in w_k.keys():
                averaged_weights[key] = n_k / n_sum * w_k[key] if i == 0 \
                    else averaged_weights[key] + n_k / n_sum * w_k[key]
        return averaged_weights


    def evaluate_private(self):
        acc_list, loss_list = [], []
        dataset_accs = defaultdict(list)

        for client in self.clients:
            p_acc = evaluate_model(client.model, client.test_loader, self.metric_type, self.device)
            client.round_test_acc = p_acc
            acc_list.append(p_acc)
            #loss_list.append(p_loss)

            dataset_accs[client.dataset_name].append(p_acc)

            self.logger.log(f"Round {self.glob_iter + 1} | Client {client.id} | Model: ({client.model_name}) | Acc: {client.round_test_acc*100:.2f}%")

        #self.p_acc, self.p_loss = np.mean(acc_list), np.mean(loss_list)
        self.p_acc = np.mean(acc_list)

        for d_name, accs in dataset_accs.items():
            self.dataset_acc_history[d_name].append(np.mean(accs) * 100.0)
    
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
                #d_name = client.dataset_name
                d_name = getattr(client, 'group_name', client.dataset_name)
                if d_name not in self.local_id_to_global_id:
                    continue
                    
                local_to_global_map = self.local_id_to_global_id[d_name]

                for x, y in client.test_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)

                    _, logits = self.model(x)
                    preds = torch.argmax(logits, dim=1)

                    y_np = y.cpu().numpy()
                    y_global = []
                    valid_indices = []

                    for i, label in enumerate(y_np):
                        if label in local_to_global_map:
                            y_global.append(local_to_global_map[label])
                            valid_indices.append(i)

                    if len(valid_indices) > 0:
                        y_global_tensor = torch.tensor(y_global, dtype=torch.long, device=self.device)
                        valid_preds = preds[valid_indices]

                        correct_mask = (valid_preds == y_global_tensor)
                        dataset_correct[d_name] += (valid_preds == y_global_tensor).sum().item()
                        dataset_total[d_name] += len(valid_indices)

                        for valid_idx, original_idx in enumerate(valid_indices):
                            local_label = y_np[original_idx]
                            is_correct = correct_mask[valid_idx].item()
                            
                            class_total[d_name][local_label] += 1
                            if is_correct:
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

        # with open(csv_path_mix, mode='a', newline='') as f:
        #     writer = csv.writer(f)
        #     if not file_exists_mix:
        #         writer.writerow(["Round", "Epoch", "Accuracy"])

        #     writer.writerow([
        #         self.glob_iter + 1,
        #         epoch,
        #         round(mix_acc, 2)
        #     ])

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


    def save_model(self, fname='checkpoints.pth'):
        self.logger.log("Saving checkpoints ...")

        client_label_distributions = {}
        for client in self.clients:
            unique_labels = set()
            for _, labels in client.train_loader:
                unique_labels.update(labels.tolist())
            client_label_distributions[client.id] = list(unique_labels)

        checkpoint = {
            'generator': self.global_gen.state_dict() if hasattr(self, 'global_gen') else None,
            'client_label_distributions': client_label_distributions,
            'global_registry': self.local_id_to_global_id,
            'label_space_meta': self.label_space_meta,
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

        server_save_path = os.path.join(self.logger.log_dir, 'config_'+fname)
        torch.save(checkpoint, server_save_path)
        self.logger.log(f"[Server] Checkpoint saved to {server_save_path}")

        clients_dir = os.path.join(self.logger.log_dir, f'clients_last_round_checkpoints')
        os.makedirs(clients_dir, exist_ok=True)

        for client in self.clients:
            arch_name = getattr(client, 'model_name', 'Unknown')
            client_path = os.path.join(clients_dir, f'client_model_{client.dataset_name}_c{client.id}_{arch_name}.pth')
            torch.save(client.model.state_dict(), client_path)
            
        self.logger.log(f"[Server] All {len(self.clients)} clients saved in {clients_dir}/")
