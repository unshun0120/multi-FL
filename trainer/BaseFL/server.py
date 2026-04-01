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
from utils.nets import ConditionalGenerator, ConditionalImageGenerator, Classifier
from utils.loss import Gen_DiversityLoss, total_variation_loss, BNSM_Hook, get_bn_loss
from utils.train_utils import evaluate_model
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

        self.global_model_epochs = exp_conf.get('global_full_model_epochs', 1) 
        self.global_model_optim_name =  exp_conf.get('global_full_model_optim', 'Adam')
        self.global_model_optim_lr = exp_conf.get('global_full_model_optim_lr', 1e-3)
        self.global_model_optimizer = eval(self.global_model_optim_name)(self.model.parameters(), self.global_model_optim_lr)


        # initial variables
        self.clients = clients
        self.num_clients = len(clients)
        self.selected_clients_ids = []
        self.selected_clients = []

        # Value: {'feature_extractor': state_dict, 'classifier': state_dict}
        self.global_models = {}

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
        groups = defaultdict(list)
        for client in self.selected_clients:
            d_name = client.dataset_name  
            groups[d_name].append(client)

            if d_name not in self.label_space_meta:
                self.label_space_meta[d_name] = client.class_name_set

        print(f"[Server] Aggregating from {len(self.selected_clients)} clients (grouped by {len(groups)} datasets)...") 
        
        for d_name, group_clients in groups.items():
            if d_name not in self.global_models:
                self.global_models[d_name] = {}

            if self.heterogeneous:
                # aggregate clients generic classifier
                msg_list = [(client.num_samples, client.model.classifier.state_dict())
                            for client in group_clients]
                w_cls = self.avg_weights(msg_list)

                num_classes = w_cls['weight'].shape[0]
                input_dim = w_cls['weight'].shape[1]

                cls_model = Classifier(input_dim, num_classes).to(self.device)
                cls_model.load_state_dict(w_cls)
                cls_model.eval()

                self.global_models[d_name]['classifier'] = cls_model
            else:
            # if not heterogeneous, aggregate feature_extractor of clients
                msg_list = [(client.num_samples, client.model.state_dict())
                            for client in group_clients]
                w_global = self.avg_weights(msg_list)

                full_model = copy.deepcopy(group_clients[0].model).to(self.device)
                full_model.load_state_dict(w_global)
                full_model.eval()
                
                self.global_models[d_name]['full_model'] = full_model
                
                self.global_models[d_name]['classifier'] = full_model.classifier
                self.global_models[d_name]['feature_extractor'] = full_model.feature_extractor

        if (self.glob_iter + 1) == self.start_mapping_epoch: 
            if self.index_matching == 'class_name':
                self.perform_name_based_mapping()
            else:
                self.test_discover_mappings_with_real_images()


    def perform_name_based_mapping(self):
        self.logger.log("[Server] Performing Name-Based Label Mapping...")
        
        # 1. Collect all unique class names across all clients
        all_class_names = set()
        for d_name in self.label_space_meta:
             for name in self.label_space_meta[d_name]:
                 all_class_names.add(name)
        
        # 2. Assign Global IDs
        sorted_names = sorted(list(all_class_names))
        name_to_gid = {name: idx for idx, name in enumerate(sorted_names)}
        
        # 3. Build local_id_to_global_id
        self.local_id_to_global_id = {}
        
        for d_name in self.label_space_meta:
            self.local_id_to_global_id[d_name] = {}
            current_names = self.label_space_meta[d_name]
            for l_id, name in enumerate(current_names):
                if name in name_to_gid:
                     self.local_id_to_global_id[d_name][l_id] = name_to_gid[name]

        self.logger.log(f"[Server] Name-based mapping completed. Found {len(sorted_names)} unique global classes.")
        
        # Log details similar to the visual method format
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
        

    def test_discover_mappings_with_real_images(self):
        self.logger.log("[Server] Testing Cross-Dataset Label Mappings with REAL images ...")

        dataset_clients_map = defaultdict(list)
        for client in self.selected_clients:
            dataset_clients_map[client.dataset_name].append(client)
            
        dataset_ids = list(dataset_clients_map.keys())
        entropy_threshold = 2.0

        def get_real_images(loader, target_label, num_samples=32):
            images_collected = []
            for imgs, labels in loader:
                mask = (labels == target_label)
                valid_imgs = imgs[mask]
                images_collected.append(valid_imgs)
                if sum(x.size(0) for x in images_collected) >= num_samples:
                    break
            
            if len(images_collected) == 0:
                return None
            
            images_cat = torch.cat(images_collected, dim=0)[:num_samples]
            return images_cat.to(self.device)

        for i, src_id in enumerate(dataset_ids):
            for j, tgt_id in enumerate(dataset_ids):
                if i == j: continue 

                tgt_clients = [c for c in self.selected_clients if c.dataset_name == tgt_id]
                src_clients = [c for c in self.selected_clients if c.dataset_name == src_id]

                src_names = self.label_space_meta[src_id]
                tgt_names = self.label_space_meta[tgt_id]
                
                src_loader = self.test_loader[src_id] 
                tgt_loader = self.test_loader[tgt_id]

                for label_idx, label_name in enumerate(src_names):
                    img_src_real = get_real_images(src_loader, label_idx, num_samples=32)
                    if img_src_real is None:
                        continue

                    with torch.no_grad():
                        avg_probs_tgt = 0
                        for c in tgt_clients:
                            c.model.to(self.device)
                            c.model.eval()
                            _, logits = c.model(img_src_real)
                            avg_probs_tgt += F.softmax(logits, dim=1).mean(dim=0)
                        avg_probs_tgt /= len(tgt_clients)

                        entropy_tgt = -torch.sum(avg_probs_tgt * torch.log(avg_probs_tgt + 1e-8)).item()
                        pred_tgt_idx = torch.argmax(avg_probs_tgt).item()
                        pred_tgt_name = tgt_names[pred_tgt_idx]
                        tgt_conf = avg_probs_tgt[pred_tgt_idx].item()

                        if entropy_tgt > entropy_threshold:
                            self.logger.log(f"{src_id}:'{label_name}' -> {tgt_id} | predict:'{pred_tgt_name}' | Entropy:{entropy_tgt:.2f} ⚠️ [Filtered: High Uncertainty]")
                            continue
                        else: 
                            self.logger.log(f"{src_id}:'{label_name}' -> {tgt_id} | predict:'{pred_tgt_name}' | Entropy:{entropy_tgt:.2f}")

                        img_tgt_real = get_real_images(tgt_loader, pred_tgt_idx, num_samples=32)
                        if img_tgt_real is None:
                            continue

                        avg_probs_src = 0
                        for c in src_clients:
                            c.model.to(self.device)
                            c.model.eval()
                            _, logits = c.model(img_tgt_real)
                            avg_probs_src += F.softmax(logits, dim=1).mean(dim=0)
                        avg_probs_src /= len(src_clients)

                        entropy_src = -torch.sum(avg_probs_src * torch.log(avg_probs_src + 1e-8)).item()
                        pred_src_cycle_idx = torch.argmax(avg_probs_src).item()
                        pred_src_cycle_name = src_names[pred_src_cycle_idx]
                        conf_src = avg_probs_src[pred_src_cycle_idx].item()

                        if entropy_src > entropy_threshold:
                            match_symbol = "❓" 
                            self.logger.log(f"  -> Check -> {src_id} | Predict:'{pred_src_cycle_name}' | {match_symbol} | Entropy:{entropy_src:.2f} ⚠️ [Filtered: Weak Cycle-Back]")
                        else:
                            match_symbol = "✅" if pred_src_cycle_idx == label_idx else "❌"
                            self.logger.log(f"  -> Check -> {src_id} | Predict:'{pred_src_cycle_name}' | {match_symbol} | Entropy:{entropy_src:.2f}")

                        if pred_src_cycle_idx == label_idx:
                            self.register_mapping(src_id, label_idx, tgt_id, pred_tgt_idx)

        for d_id in self.label_space_meta.keys():
             if d_id not in self.local_id_to_global_id:
                 self.local_id_to_global_id[d_id] = {}
             
             class_names = self.label_space_meta[d_id]
             
             for l_id in range(len(class_names)):
                 if l_id not in self.local_id_to_global_id[d_id]:
                     current_max = -1
                     for d in self.local_id_to_global_id:
                         if self.local_id_to_global_id[d]:
                             current_max = max(current_max, max(self.local_id_to_global_id[d].values()))
                     new_gid = current_max + 1
                     
                     self.local_id_to_global_id[d_id][l_id] = new_gid
                     
                     l_name = class_names[l_id]                

    def register_mapping(self, d1, l1, d2, l2):
        gid = None
        
        if d1 in self.local_id_to_global_id and l1 in self.local_id_to_global_id[d1]:
             gid = self.local_id_to_global_id[d1][l1]
        
        if gid is None and d2 in self.local_id_to_global_id and l2 in self.local_id_to_global_id[d2]:
             gid = self.local_id_to_global_id[d2][l2]
             
        if gid is None:
             current_max = -1
             for d in self.local_id_to_global_id:
                 if self.local_id_to_global_id[d]:
                     current_max = max(current_max, max(self.local_id_to_global_id[d].values()))
             gid = current_max + 1
             
        if d1 not in self.local_id_to_global_id: self.local_id_to_global_id[d1] = {}
        self.local_id_to_global_id[d1][l1] = gid
        
        if d2 not in self.local_id_to_global_id: self.local_id_to_global_id[d2] = {}
        self.local_id_to_global_id[d2][l2] = gid


    def img_train_dataset_generators(self):
        self.logger.log("[Server] Training Per-Dataset Generators ...")

        feat_gen_noise_dim = 128
        div_loss_fn = Gen_DiversityLoss(metric='l1').to(self.device)

        gen_epochs = 1500
        gen_lr = 2e-3
        fedted_beta = 0.0
        bn_weight = 0.05
        tv_weight = 0.0
        
        for ls_id, model_dict in self.global_models.items():
            dataset_clients = [c for c in self.selected_clients if c.dataset_name == ls_id]
            if len(dataset_clients) == 0:
                continue

            class_names = self.label_space_meta[ls_id]
            num_local_classes = len(class_names)

            if 'generator' not in model_dict:
                gen = ConditionalImageGenerator(
                    num_classes=num_local_classes, 
                    noise_dim=feat_gen_noise_dim,
                    img_channels=1,
                    img_size=28   
                ).to(self.device)
                model_dict['generator'] = gen
                model_dict['gen_optimizer'] = torch.optim.Adam(gen.parameters(), lr=gen_lr)
            
            generator = model_dict['generator']
            optimizer = model_dict['gen_optimizer']

            for c in dataset_clients:
                c.model.to(self.device)
                c.model.eval() 
                for param in c.model.parameters():      
                    param.requires_grad = False

            generator.train()

            batch_size = 64
            epoch_loss_tracker = []

            for epoch in tqdm(range(gen_epochs), colour='blue', ncols=100):
                class_order = torch.randperm(num_local_classes).tolist()

                #for c_idx in class_order:
                for _ in range(10):
                    batch_labels = torch.randint(0, num_local_classes, (batch_size,)).to(self.device)
                    z_noise = torch.randn(batch_size, feat_gen_noise_dim).to(self.device)

                    optimizer.zero_grad()
                    
                    gen_imgs = generator(z_noise, batch_labels)
                    gen_imgs.retain_grad()
                    
                    div_loss = div_loss_fn(gen_imgs.view(batch_size, -1), z_noise)
                    
                    total_cls_loss = 0.0
                    total_bn_loss = 0.0

                    for client in dataset_clients:
                        bn_hooks = []
                        for module in client.model.modules():
                            if isinstance(module, torch.nn.BatchNorm2d):
                                bn_hooks.append(BNSM_Hook(module))
                        
                        _, logits = client.model(gen_imgs)
                        
                        total_cls_loss += F.cross_entropy(logits, batch_labels)
                        
                        total_bn_loss += get_bn_loss(bn_hooks)
                        for hook in bn_hooks:
                            hook.close()

                    cls_loss = total_cls_loss / len(dataset_clients)
                    bn_loss = total_bn_loss / len(dataset_clients)

                    tv_loss = total_variation_loss(gen_imgs) 

                    loss = cls_loss + (fedted_beta * div_loss) + (tv_weight * tv_loss) + (bn_weight * bn_loss)
                    loss.backward()
                    
                    optimizer.step()
                    
                    epoch_loss_tracker.append(loss.item())

            avg_loss = sum(epoch_loss_tracker) / len(epoch_loss_tracker)
            self.logger.log(f"     Dataset [{ls_id}] Generator | Epochs: {gen_epochs} | Final Loss: {avg_loss:.4f}")

            generator.eval()
            with torch.no_grad():
                save_dir = self.logger.log_dir

                num_samples_per_class = 1 
                #num_classes_to_plot = min(10, num_local_classes) # 畫 0~9
                num_classes_to_plot = num_local_classes

                sample_labels = torch.arange(num_classes_to_plot).repeat(num_samples_per_class).to(self.device)
                sample_z = torch.randn(num_samples_per_class * num_classes_to_plot, feat_gen_noise_dim).to(self.device)
                sample_imgs = generator(sample_z, sample_labels)
                sample_imgs = (sample_imgs * 0.5 + 0.5).clamp(0, 1).cpu().numpy()

                cols = min(10, num_classes_to_plot)
                rows = math.ceil(num_classes_to_plot / cols)

                fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
                
                if rows * cols == 1:
                    axes_flat = [axes]
                else:
                    axes_flat = axes.flatten()
                
                for idx in range(num_classes_to_plot):
                    ax = axes_flat[idx]
                    
                    img = sample_imgs[idx].squeeze() 
                    ax.imshow(img, cmap='gray')
                    
                    label_name = str(class_names[idx])
                    ax.set_title(label_name, fontsize=12)
                    ax.axis('off') 

                for idx in range(num_classes_to_plot, len(axes_flat)):
                    axes_flat[idx].axis('off')

                plt.tight_layout()
                save_path = os.path.join(save_dir, f'{ls_id}_generated_samples.png')
                plt.savefig(save_path)
                plt.close() 
                self.logger.log(f"     -> Sample images with labels saved to {save_path}")

        self.logger.log("[Server] Per-Dataset Generators trained.")

        for ls_id, model_dict in self.global_models.items():
            dataset_clients = [c for c in self.selected_clients if c.dataset_name == ls_id]
            for c in dataset_clients:
                for param in c.model.parameters():
                    param.requires_grad = True
        
        self.img_discover_label_mappings()


    def img_discover_label_mappings(self):
        self.logger.log("[Server] Discovering Cross-Dataset Label Mappings ...")

        dataset_ids = list(self.global_models.keys())
        feat_gen_noise_dim = 128

        for i, src_id in enumerate(dataset_ids):
            for j, tgt_id in enumerate(dataset_ids):
                if i == j: continue 

                tgt_clients = [c for c in self.selected_clients if c.dataset_name == tgt_id]
                src_clients = [c for c in self.selected_clients if c.dataset_name == src_id]

                src_group = self.global_models[src_id]
                tgt_group = self.global_models[tgt_id]

                src_gen = src_group['generator']; src_model = src_group['full_model']
                tgt_gen = tgt_group['generator']; tgt_model = tgt_group['full_model']

                src_names = self.label_space_meta[src_id]
                tgt_names = self.label_space_meta[tgt_id]

                src_gen.eval(); src_model.eval()
                tgt_gen.eval(); tgt_model.eval()

                for label_idx, label_name in enumerate(src_names):
                    num_samples = 32
                    z = torch.randn(num_samples, feat_gen_noise_dim).to(self.device)
                    label_tensor = torch.tensor([label_idx] * num_samples).to(self.device)

                    with torch.no_grad():
                        img_src = src_gen(z, label_tensor)

                        avg_probs_tgt = 0
                        for c in tgt_clients:
                            c.model.eval()
                            _, logits = c.model(img_src)
                            avg_probs_tgt += F.softmax(logits, dim=1).mean(dim=0)
                        avg_probs_tgt /= len(tgt_clients)

                        entropy_tgt = -torch.sum(avg_probs_tgt * torch.log(avg_probs_tgt + 1e-8)).item()

                        pred_tgt_idx = torch.argmax(avg_probs_tgt).item()

                        pred_tgt_name = tgt_names[pred_tgt_idx]

                        tgt_conf = avg_probs_tgt[pred_tgt_idx].item()

                        self.logger.log(f"\n{src_id}:'{label_name}' -> {tgt_id} | Predict:'{pred_tgt_name}'| Entropy:{entropy_tgt:.2f}")

                        z_cycle = torch.randn(num_samples, feat_gen_noise_dim).to(self.device)
                        label_tensor_tgt = torch.tensor([pred_tgt_idx] * num_samples).to(self.device)
                        
                        img_tgt = tgt_gen(z_cycle, label_tensor_tgt)

                        avg_probs_src = 0
                        for c in src_clients:
                            c.model.eval()
                            _, logits = c.model(img_tgt)
                            avg_probs_src += F.softmax(logits, dim=1).mean(dim=0)
                        avg_probs_src /= len(src_clients)

                        entropy_src = -torch.sum(avg_probs_src * torch.log(avg_probs_src + 1e-8)).item()

                        pred_src_cycle_idx = torch.argmax(avg_probs_src).item()

                        pred_src_cycle_name = src_names[pred_src_cycle_idx]

                        conf_src = avg_probs_src[pred_src_cycle_idx].item()

                        match_symbol = "✅" if pred_src_cycle_idx == label_idx else "❌"

                        self.logger.log(f"  -> Check -> {src_id} | Predict:'{pred_src_cycle_name}' | {match_symbol} | Entropy:{entropy_src:.2f}")


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
    

    def save_model(self, fname='checkpoints.pth'):
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

        server_save_path = os.path.join(self.logger.log_dir, 'server_'+fname)
        torch.save(checkpoint, server_save_path)
        self.logger.log(f"[Server] Checkpoint saved to {server_save_path}")

        clients_dir = os.path.join(self.logger.log_dir, f'clients_last_round_checkpoints')
        os.makedirs(clients_dir, exist_ok=True)

        for client in self.clients:
            arch_name = getattr(client, 'model_name', 'Unknown')
            client_path = os.path.join(clients_dir, f'client_model_{client.dataset_name}_c{client.id}_{arch_name}.pth')
            torch.save(client.model.state_dict(), client_path)
            
        self.logger.log(f"[Server] All {len(self.clients)} clients saved in {clients_dir}/")
