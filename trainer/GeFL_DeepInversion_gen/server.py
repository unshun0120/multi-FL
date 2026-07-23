"""
Server for GeFL (Shared DeepInversion Generator across ALL datasets)
"""

import torch
import copy
from collections import OrderedDict, defaultdict, deque
import os
import csv
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random
import math
from torch.optim import *
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import make_grid, save_image

from trainer.BaseFL.server import Server as BaseServer
from utils.plotting import plot_accuracy_curves
from utils.nets import ResNet, BasicBlock, ConditionalImageGenerator

from label_mapping.label_mapping_utils import (
    label_mapping, evaluate_mapping_results,
    feature_bi_direction_label_mapping, single_direction_label_mapping,
    get_gen_images, global_to_local_mapping, clear_image_caches,
    image_cosine_similarity_mapping,
)

from utils.DFKD_utils import (
    KLDiv, JSDiv, evaluate_student_model,
    jitter_and_flip, get_image_prior_losses, DeepInversionHook,
)

from trainer.GeFL_DeepInversion_gen.mapping_res import get_mapping

GEN_CONFIG = {
    'img_num_samples': 64,
    'feat_gen_noise_dim': 128,
    'student_lr': 1e-3,
    'kd_steps': 1,
}

di_weight = {
    'epochs': 1,
    'g_steps': 15000,
    # 'lr': 0.02,
    'lr': 1e-3,
    'ce': 1.0,
    'bn': 0.05,
    'tv': 0.005,
    'l2': 0.0,
    'adv': 0.0,
}


class Server(BaseServer):
    def __init__(self, **kwargs):
        super(Server, self).__init__(**kwargs)

        self.generator = None
        self.model = None

    def run(self):
        self.logger.log("")
        self.logger.log("=" * 50)
        self.logger.log(f"Start {self.global_rounds} rounds training by {self.algorithm}")

        for r in range(self.global_rounds):
            self.glob_iter = r
            self.sample_clients()
            self.distribute_model()
            self.local_update()
            self.aggregate()

        self.save_model()

    def distribute_model(self):
        pass  

    def aggregate(self):
        groups = defaultdict(list)
        for client in self.selected_clients:
            d_name = client.dataset_name
            groups[d_name].append(client)
            if d_name not in self.label_space_meta:
                self.label_space_meta[d_name] = client.class_name_set

        print(f"[Server] Aggregating from {len(self.selected_clients)} clients "
              f"(grouped by {len(groups)} datasets)...")

        if (self.glob_iter + 1) >= self.start_mapping_epoch:
            clear_image_caches()

            if self.args.label_mapping == 'class_name':
                self.class_name_label_mapping()
            elif self.args.label_mapping == 'independent':
                self.independent_label_mapping()
            elif self.args.label_mapping == 'identical':
                self.identical_label_mapping()

            elif self.args.label_mapping == 'ours_5':
                self.local_id_to_global_id = get_mapping("ours_5")
            elif self.args.label_mapping == 'ours_10':
                self.local_id_to_global_id = get_mapping("ours_10")
            elif self.args.label_mapping == 'ours_15':
                self.local_id_to_global_id = get_mapping("ours_15")
            elif self.args.label_mapping == 'ours_20':
                self.local_id_to_global_id = get_mapping("ours_20")
            elif self.args.label_mapping == 'ours_25':
                self.local_id_to_global_id = get_mapping("ours_25")

            elif self.args.label_mapping == 'ours_5_noEntropy':
                self.local_id_to_global_id = get_mapping("ours_5_noEntropy")

            elif self.args.label_mapping == 'cs_mapping_25':
                self.local_id_to_global_id = get_mapping("cs_mapping_25")
            elif self.args.label_mapping == 'feature_mapping_25':
                self.local_id_to_global_id = get_mapping("feature_mapping_25")
            elif self.args.label_mapping == 'single_mapping_25':
                self.local_id_to_global_id = get_mapping("single_mapping_25")

            elif self.args.label_mapping == 'slam_dunk_mapping_5':
                self.local_id_to_global_id = get_mapping("slam_dunk_mapping_5")
            elif self.args.label_mapping == 'slam_dunk_mapping_10':
                self.local_id_to_global_id = get_mapping("slam_dunk_mapping_10")
            elif self.args.label_mapping == 'slam_dunk_mapping_15':
                self.local_id_to_global_id = get_mapping("slam_dunk_mapping_15")
            elif self.args.label_mapping == 'slam_dunk_mapping_20':
                self.local_id_to_global_id = get_mapping("slam_dunk_mapping_20")
            elif self.args.label_mapping == 'slam_dunk_mapping_25':
                self.local_id_to_global_id = get_mapping("slam_dunk_mapping_25")

            self.logger.log(f"[Server] Label mapping done. "
                            f"num_global_classes = {self._count_global_classes()}")

            all_client_models = []
            for client in self.clients:
                m = client.model.to(self.device)
                m.dataset_name = client.dataset_name  
                all_client_models.append(m)

            self.generator = self._train_shared_generator_DeepInversion(
                all_client_models=all_client_models,
                local_id_to_global_id=self.local_id_to_global_id,
                device=self.device,
            )

            self._draw_global_samples()

        # if (self.glob_iter + 1) >= self.start_mapping_epoch:
        #     self.train_global_inference_model()
        #     self.test_global_inference_model()

    def _count_global_classes(self):
        gids = set()
        for mapping in self.local_id_to_global_id.values():
            gids.update(mapping.values())
        return len(gids)

    def _train_shared_generator_DeepInversion(self, all_client_models, local_id_to_global_id, device):
    
        self.logger.log("[Server] Training SHARED generator via DeepInversion (all datasets)...")
        self.logger.log("--- DeepInversion Configuration ---")
        for k, v in di_weight.items():
            self.logger.log(f"    {k}: {v}")
        self.logger.log("-----------------------------------")

        gid_to_clients = defaultdict(list)
        for client_model in all_client_models:
            d_name = client_model.dataset_name
            if d_name not in local_id_to_global_id:
                continue
            for local_id, global_id in local_id_to_global_id[d_name].items():
                gid_to_clients[global_id].append((client_model, local_id))

        num_global_classes = max(gid_to_clients.keys()) + 1 if gid_to_clients else 0
        if num_global_classes == 0:
            self.logger.log("[Server] No global classes found, skipping generator training.")
            return None

        generator = ConditionalImageGenerator(
            num_classes=num_global_classes,
            noise_dim=GEN_CONFIG['feat_gen_noise_dim'],
            img_channels=3,
            img_size=32,
        ).to(device)

        generator_optimizer = torch.optim.Adam(
            generator.parameters(),
            lr=di_weight['lr'],
            betas=[0.5, 0.99],
        )

        model_training_modes = {}
        for m in all_client_models:
            model_training_modes[id(m)] = m.training
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

        bn_hooks_per_model = {}
        for m in all_client_models:
            hooks = []
            for module in m.modules():
                if hasattr(module, 'inplace'):
                    module.inplace = False
                if isinstance(module, nn.BatchNorm2d):
                    hooks.append(DeepInversionHook(module))
            bn_hooks_per_model[id(m)] = hooks

        epoch_loss_tracker = []

        batch_size = 64
        all_global_ids = list(gid_to_clients.keys())

        for epoch in tqdm(range(di_weight['epochs']), colour='blue', ncols=100,
                        desc="SharedGen"):

            generator.train()
            epoch_loss = 0.0
            valid_steps = 0

            for it in tqdm(range(di_weight['g_steps']), leave=False):

                sampled_gids = [random.choice(all_global_ids) for _ in range(batch_size)]
                batch_global_labels = torch.tensor(sampled_gids, dtype=torch.long, device=device)

                noise = torch.randn(
                    batch_size,
                    GEN_CONFIG['feat_gen_noise_dim'],
                    device=device,
                )

                inputs = generator(noise, batch_global_labels)

                generator_optimizer.zero_grad()

                batch_loss = 0.0
                clients_contributed = 0

                for client_model in all_client_models:
                    d_name = client_model.dataset_name
                    if d_name not in local_id_to_global_id:
                        continue

                    g2l = {g: l for l, g in local_id_to_global_id[d_name].items()}

                    valid_positions = []
                    local_labels_for_valid = []

                    for pos, gid in enumerate(sampled_gids):
                        if gid in g2l:
                            valid_positions.append(pos)
                            local_labels_for_valid.append(g2l[gid])

                    if not valid_positions:
                        continue

                    sub_imgs = inputs[valid_positions]
                    sub_labels = torch.tensor(local_labels_for_valid,
                                            dtype=torch.long, device=device)

                    preds = client_model(sub_imgs)
                    logits_t = preds[1] if isinstance(preds, tuple) else preds

                    cls_loss = F.cross_entropy(logits_t, sub_labels)

                    bn_loss = 0.0
                    hooks = bn_hooks_per_model[id(client_model)]
                    if hooks and di_weight['bn'] != 0:
                        bn_loss = sum(h.r_feature for h in hooks) / len(hooks)

                    tv_loss = get_image_prior_losses(sub_imgs)
                    l2_loss = torch.norm(sub_imgs, 2)

                    client_loss = (di_weight['ce'] * cls_loss +
                                di_weight['bn'] * bn_loss +
                                di_weight['tv'] * tv_loss +
                                di_weight['l2'] * l2_loss)

                    batch_loss += client_loss
                    clients_contributed += 1

                if clients_contributed > 0:
                    avg_loss = batch_loss / clients_contributed
                    avg_loss.backward()
                    generator_optimizer.step()

                    epoch_loss += avg_loss.item()
                    valid_steps += 1

            avg_epoch_loss = epoch_loss / max(1, valid_steps)
            epoch_loss_tracker.append(avg_epoch_loss)

            self.logger.log(f"[Server] Generator Epoch {epoch + 1}/{di_weight['epochs']} "
                            f"| Loss: {avg_epoch_loss:.4f}")

        avg_loss_all = sum(epoch_loss_tracker) / max(1, len(epoch_loss_tracker))
        self.logger.log(f"[Server] Shared Generator | Epochs: {di_weight['epochs']} "
                        f"| Avg Loss: {avg_loss_all:.4f}")

        for hooks in bn_hooks_per_model.values():
            for h in hooks:
                if hasattr(h, 'close'):   h.close()
                elif hasattr(h, 'remove'): h.remove()

        for m in all_client_models:
            for p in m.parameters():
                p.requires_grad = True
            m.train(model_training_modes[id(m)])

        return generator


    def _sample_global_images(self, global_id, num_samples):
        if self.generator is None:
            return None

        generator_mode = self.generator.training
        self.generator.eval()

        with torch.no_grad():
            batch_global_labels = torch.full(
                (num_samples,),
                global_id,
                dtype=torch.long,
                device=self.device,
            )

            noise = torch.randn(
                num_samples,
                GEN_CONFIG['feat_gen_noise_dim'],
                device=self.device,
            )

            inputs = self.generator(noise, batch_global_labels)

        self.generator.train(generator_mode)

        return inputs.cpu()
    

    def _draw_global_samples(self):
        """Save a grid image for each global_id to inspect generation quality."""
        if self.generator is None:
            return

        draw_dir = os.path.join(
            self.logger.log_dir,
            'gen_samples',
            f'round_{self.glob_iter + 1}_{self.args.label_mapping}'
        )
        os.makedirs(draw_dir, exist_ok=True)

        def safe_text(text):
            text = str(text)
            text = text.replace(" ", "")
            text = text.replace("-", "")
            text = text.replace("/", "_")
            return text

        def format_class_name(class_name):
            class_name = str(class_name)
            if class_name.isdigit():
                return f"digit{class_name}"
            return safe_text(class_name)

        global_id_to_names = defaultdict(list)

        for d_name, mapping in self.local_id_to_global_id.items():
            for local_id, global_id in mapping.items():
                if d_name in self.label_space_meta and local_id < len(self.label_space_meta[d_name]):
                    class_name = self.label_space_meta[d_name][local_id]
                else:
                    class_name = str(local_id)

                dataset_name = safe_text(d_name)
                class_name = format_class_name(class_name)

                global_id_to_names[global_id].append(f"{dataset_name}_{class_name}")

        all_global_ids = set()
        for mapping in self.local_id_to_global_id.values():
            all_global_ids.update(mapping.values())

        for gid in sorted(all_global_ids):
            pool = self._sample_global_images(
                global_id=gid,
                num_samples=GEN_CONFIG['img_num_samples'],
            )

            if pool is None or len(pool) == 0:
                continue

            imgs = pool.clamp(-1.0, 1.0)
            imgs = (imgs + 1.0) / 2.0

            ncols = min(8, len(imgs))
            grid = make_grid(imgs, nrow=ncols, padding=2)

            name_parts = global_id_to_names.get(gid, [f"global{gid}"])
            file_name = f"{gid}_" + "_".join(name_parts) + ".pdf"

            save_path = os.path.join(draw_dir, file_name)
            save_image(grid, save_path)

        self.logger.log(f"[Server] Generated sample grids saved to {draw_dir}")


    def train_global_inference_model(self):
        all_global_ids = set()
        for mapping in self.local_id_to_global_id.values():
            all_global_ids.update(mapping.values())
        all_global_ids = list(all_global_ids)
        num_global_classes = len(all_global_ids)

        if num_global_classes == 0:
            self.logger.log("Warning: No mappings found. Skipping global model training.")
            return

        if self.model is None:
            self.logger.log(f"Initializing Global Inference Model with {num_global_classes} classes...")
            self.model = ResNet(BasicBlock, [2, 2, 2, 2],
                                in_channels=3, num_classes=num_global_classes, global_dim=256)
            optim_name = self.exp_conf.get('global_model_optim', 'Adam')
            optim_lr   = self.exp_conf.get('global_model_optim_lr', 1e-3)
            self.global_model_optimizer = eval(optim_name)(self.model.parameters(), optim_lr)

        self.model.to(self.device)
        self.model.train()

        criterion = nn.CrossEntropyLoss()
        dataset_x, dataset_y = [], []

        for d_name, mapping in self.local_id_to_global_id.items():
            for local_id, global_id in mapping.items():
                x_gen = self._sample_global_images(
                    global_id=global_id,
                    num_samples=self.global_samples_per_class,
                )

                if x_gen is None or len(x_gen) == 0:
                    continue

                for i in range(self.global_samples_per_class):
                    dataset_x.append(x_gen[i])
                    dataset_y.append(global_id)

        if not dataset_x:
            self.logger.log("Warning: No synthetic data generated.")
            return

        dataset_x = torch.stack(dataset_x)
        dataset_y = torch.tensor(dataset_y, dtype=torch.long)

        train_loader = DataLoader(TensorDataset(dataset_x, dataset_y),
                                  batch_size=self.batch_size, shuffle=True)

        for epoch in tqdm(range(50), colour="green", ncols=100):
            self.model.train()
            epoch_loss = 0.0
            for gen_imgs, y_batch in train_loader:
                gen_imgs, y_batch = gen_imgs.to(self.device), y_batch.to(self.device)
                self.global_model_optimizer.zero_grad()
                _, logits = self.model(gen_imgs)
                loss = criterion(logits, y_batch)
                loss.backward()
                self.global_model_optimizer.step()
                epoch_loss += loss.item()

            self.test_global_inference_model(epoch=epoch + 1)

            self.logger.log(f"Epoch {epoch} Loss: {epoch_loss / len(train_loader):.4f}")

    def aggregate_weights(self, weights_list):
        total_samples = sum(w[0] for w in weights_list)
        avg_params = OrderedDict()
        for name in weights_list[0][1].keys():
            avg_params[name] = torch.zeros_like(weights_list[0][1][name], dtype=torch.float32)
            for num_samples, params in weights_list:
                avg_params[name] += params[name] * (num_samples / total_samples)
        return avg_params

    def save_model(self, fname='checkpoints.pth'):
        self.logger.log("Saving checkpoints ...")

        client_label_distributions = {}
        for client in self.clients:
            unique_labels = set()
            for _, labels in client.train_loader:
                unique_labels.update(labels.tolist())
            client_label_distributions[client.id] = list(unique_labels)

        checkpoint = {
            'client_label_distributions': client_label_distributions,
            'global_registry': getattr(self, 'local_id_to_global_id', {}),
            'label_space_meta': self.label_space_meta,
            'global_feature_dim': getattr(self, 'global_feature_dim', 256),
            'exp_conf': self.exp_conf,
            'args': {
                'num_train_mnist':        self.args.num_train_mnist,
                'num_train_emnist':       self.args.num_train_emnist,
                'num_train_fashionmnist': self.args.num_train_fashionmnist,
                'num_train_cifar10':      self.args.num_train_cifar10,
                'num_train_cifar100':     self.args.num_train_cifar100,
                'num_new_clients':        self.args.num_new_clients,
                'seed':                   self.args.seed,
                'device':                 str(self.args.device),
                'algorithm':              self.args.algorithm,
            },
        }

        server_save_path = os.path.join(self.logger.log_dir, 'server_' + fname)
        torch.save(checkpoint, server_save_path)
        self.logger.log(f"[Server] Checkpoint saved to {server_save_path}")

        if self.model is not None:
            torch.save(self.model.state_dict(),
                       os.path.join(self.logger.log_dir, 'server_global_model.pth'))

        clients_dir = os.path.join(self.logger.log_dir, 'clients_last_round_checkpoints')
        os.makedirs(clients_dir, exist_ok=True)
        for client in self.clients:
            arch_name = getattr(client, 'model_name', 'Unknown')
            client_path = os.path.join(
                clients_dir,
                f'client_model_{client.dataset_name}_c{client.id}_{arch_name}.pth')
            torch.save(client.model.state_dict(), client_path)

        self.logger.log(f"[Server] All {len(self.clients)} clients saved in {clients_dir}/")
