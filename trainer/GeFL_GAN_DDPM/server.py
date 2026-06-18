import torch
import copy
from collections import OrderedDict, defaultdict
import os
import csv
import numpy as np
import torchvision.utils as vutils

from trainer.BaseFL.server import Server as BaseServer
from utils.plotting import plot_accuracy_curves
from utils.nets import DCGANGenerator, ContextUnet, DDPM
from label_mapping.label_mapping_utils import label_mapping, evaluate_mapping_results, get_gen_images, get_real_images, clear_image_caches

class Server(BaseServer):
    def __init__(self, **kwargs):
        super(Server, self).__init__(**kwargs)

        self.global_gan_gen_states = {}
        self.global_gan_dis_states = {}

        self.global_ddpm_states = {}

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

            if (r+1) % 5 == 0:
                #self.evaluate_mapping(r + 1)
                self.visualize_synthetic_images(current_round=(r+1))

            #self.evaluate_mapping(r + 1)

            if r+1 == 40:
                self.save_model() 

        # self.save_metric()

        self.save_model()
        plot_accuracy_curves(self.dataset_acc_history, self.logger.log_dir, self.args, self.global_rounds, self.dirichlet_alpha)


    def aggregate(self):
        groups = defaultdict(list)
        for client in self.selected_clients:
            d_name = client.dataset_name  
            groups[d_name].append(client)

            if d_name not in self.label_space_meta:
                self.label_space_meta[d_name] = client.class_name_set

        print(f"[Server] Aggregating from {len(self.selected_clients)} clients (grouped by {len(groups)} datasets)...")

        for d_name, group_clients in groups.items():
            self.global_gan_gen_states[d_name] = self.aggregate_weights([(c.num_samples, c.generator.state_dict()) for c in group_clients])
            self.global_gan_dis_states[d_name] = self.aggregate_weights([(c.num_samples, c.discriminator.state_dict()) for c in group_clients])
            self.global_ddpm_states[d_name] = self.aggregate_weights([(c.num_samples, c.ddpm.state_dict()) for c in group_clients])

    def evaluate_mapping(self, current_round):
        clear_image_caches()

        self.logger.log(f"[Server] Evaluating Label Mapping for all thresholds at Round {current_round}...")

        dataset_clients_dict = {}
        active_datasets = []
        dataset_label_space_meta = {}
        
        for client in self.clients: 
            d_name = client.dataset_name
            
            if d_name not in dataset_clients_dict:
                dataset_clients_dict[d_name] = []
                active_datasets.append(d_name)
                dataset_label_space_meta[d_name] = self.label_space_meta[d_name]
                
            dataset_clients_dict[d_name].append(client.model.to(self.device))

        gan_dict = {}
        ddpm_dict = {}
        for d_name in active_datasets:
            if d_name in self.global_gan_gen_states:
                gan = DCGANGenerator(
                    num_classes=len(self.label_space_meta[d_name]), noise_dim=self.exp_conf.get('gen_noise_dim', 128),
                    img_size=self.exp_conf.get('img_size', 32), channels=self.exp_conf.get('channels', 3)
                ).to(self.device)
                gan.load_state_dict(self.global_gan_gen_states[d_name])
                gan.eval()
                gan_dict[d_name] = gan

            if d_name in self.global_ddpm_states:
                nn_model = ContextUnet(
                    in_channels=self.exp_conf.get('channels', 3), 
                    n_feat=self.exp_conf.get('n_feat', 64), 
                    n_classes=len(self.label_space_meta[d_name])
                ).to(self.device)
                
                ddpm = DDPM(
                    nn_model=nn_model, 
                    betas=(1e-4, 0.02), 
                    n_T=400, 
                    device=self.device, 
                    drop_prob=0.1
                ).to(self.device)

                ddpm.load_state_dict(self.global_ddpm_states[d_name])
                ddpm.eval()
                ddpm_dict[d_name] = ddpm

        if not active_datasets:
            return 

        test_loaders_dict = {}
        for client in self.clients:
            if client.dataset_name not in test_loaders_dict:
                test_loaders_dict[client.dataset_name] = getattr(client, 'test_loader', client.train_loader)

        dynamic_entropy_ratios = [round(x, 2) for x in np.arange(0.10, 1.05, 0.05)]
        use_new_entropy_method = self.exp_conf.get('use_new_entropy_method', False)
        
        csv_dir = os.path.join(self.logger.log_dir, "mapping_results")
        os.makedirs(csv_dir, exist_ok=True)
        csv_filename = os.path.join(csv_dir, f"{self.algorithm}_mapping_acc_per_round.csv")
        file_exists = os.path.isfile(csv_filename)

        class DummyLogger:
            def log(self, msg): pass

        self.label_mapping_evaluation("GAN", get_gen_images, {'gen_dict': gan_dict}, dataset_clients_dict, dataset_label_space_meta, active_datasets, dynamic_entropy_ratios, current_round, DummyLogger())
        self.label_mapping_evaluation("DDPM", get_gen_images, {'gen_dict': ddpm_dict}, dataset_clients_dict, dataset_label_space_meta, active_datasets, dynamic_entropy_ratios, current_round, DummyLogger())
        self.label_mapping_evaluation("Real", get_real_images, {'test_loaders': test_loaders_dict}, dataset_clients_dict, dataset_label_space_meta, active_datasets, dynamic_entropy_ratios, current_round, DummyLogger())

    def label_mapping_evaluation(self, model_type, get_images_func, img_kwargs, dataset_clients_dict, dataset_label_space_meta, active_datasets, thresholds, round_num, logger):
        if not img_kwargs:
            return
            
        csv_dir = os.path.join(self.logger.log_dir, "mapping_results")
        os.makedirs(csv_dir, exist_ok=True)
        csv_filename = os.path.join(csv_dir, f"{self.algorithm}_{model_type}_mapping_acc_per_round.csv")
        file_exists = os.path.isfile(csv_filename)

        with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['global_round', 'entropy_ratio', 'recall', 'specificity', 'precision', 'average_accuracy', 'f1_score', 'mcc', 'TP', 'FP', 'TN', 'FN'])

            for thresh in thresholds:
                print(thresh)
                res_mapping = label_mapping(
                    get_images_func=get_images_func, 
                    dataset_ids=active_datasets,    
                    clients_dict=dataset_clients_dict,
                    label_space_meta=dataset_label_space_meta,
                    entropy_ratio=thresh,
                    use_new_entropy_method=self.exp_conf.get('use_new_entropy_method', False),
                    logger=logger, 
                    args=self.args,
                    **img_kwargs
                )
                metrics = evaluate_mapping_results(active_datasets, dataset_label_space_meta, res_mapping)
                writer.writerow([
                    round_num, thresh, 
                    metrics['Recall'], metrics['Specificity'], metrics['Precision'], 
                    metrics['AvgAccuracy'], metrics['F1-Score'], metrics['MCC'],
                    metrics['TP'], metrics['FP'], metrics['TN'], metrics['FN']
                ])

        
    def distribute_model(self):
        for client in self.selected_clients:
            d_name = client.dataset_name
            if d_name in self.global_gan_gen_states:
                client.generator.load_state_dict(self.global_gan_gen_states[d_name])
                client.discriminator.load_state_dict(self.global_gan_dis_states[d_name])
            if d_name in self.global_ddpm_states:
                client.ddpm.load_state_dict(self.global_ddpm_states[d_name])

    def visualize_synthetic_images(self, current_round, save_dir_name="synthetic_images"):
        """
        """
            
        self.logger.log(f"[Server] Generating synthetic images for Round {current_round}...")
        
        save_dir = os.path.join(self.logger.log_dir, save_dir_name)
        os.makedirs(save_dir, exist_ok=True)
        
        noise_dim = self.exp_conf.get('gen_noise_dim', 128)
        img_size = self.exp_conf.get('img_size', 32)
        channels = self.exp_conf.get('channels', 3)
        n_feat = self.exp_conf.get('n_feat', 64)
        
        images_per_class = 1
        
        for d_name in self.global_gan_gen_states.keys():
            class_names = self.label_space_meta[d_name]
            num_classes = len(class_names)
            
            y_tensor = torch.arange(num_classes).repeat_interleave(images_per_class).to(self.device)
            total_samples = num_classes * images_per_class
            
            generator = DCGANGenerator(
                num_classes=num_classes, noise_dim=noise_dim, img_size=img_size, channels=channels
            ).to(self.device)
            generator.load_state_dict(self.global_gan_gen_states[d_name])
            generator.eval()
            
            with torch.no_grad():
                z = torch.randn(total_samples, noise_dim, device=self.device)
                gan_imgs = generator(z, y_tensor)
                
                gan_save_path = os.path.join(save_dir, f"Round_{current_round:03d}_{d_name}_GAN.png")
                vutils.save_image(gan_imgs, gan_save_path, nrow=images_per_class, normalize=True, value_range=(-1, 1))
            
            unet = ContextUnet(in_channels=channels, n_feat=n_feat, n_classes=num_classes)
            ddpm = DDPM(
                nn_model=unet, betas=(1e-4, 0.02), n_T=400, device=self.device, drop_prob=0.1
            ).to(self.device)
            ddpm.load_state_dict(self.global_ddpm_states[d_name])
            ddpm.eval()
            
            with torch.no_grad():
                ddpm_imgs, _ = ddpm.sample(
                    n_sample=total_samples, 
                    size=(channels, img_size, img_size), 
                    device=self.device, 
                    guide_w=0.0
                )
                
                ddpm_save_path = os.path.join(save_dir, f"Round_{current_round:03d}_{d_name}_DDPM.png")
                vutils.save_image(ddpm_imgs, ddpm_save_path, nrow=images_per_class, normalize=True, value_range=(-1, 1))
                
        self.logger.log(f"[Server] Synthetic images saved to {save_dir}/")

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

        gan_dir = os.path.join(self.logger.log_dir, 'global_gans')
        os.makedirs(gan_dir, exist_ok=True)
        
        for d_name in self.global_gan_gen_states.keys():
            gan_checkpoint = {
                'generator': self.global_gan_gen_states[d_name]
            }
            gan_save_path = os.path.join(gan_dir, f'{d_name}_GAN.pth')
            torch.save(gan_checkpoint, gan_save_path)
            self.logger.log(f"[Server] Global GAN for {d_name} saved to {gan_save_path}")

        ddpm_dir = os.path.join(self.logger.log_dir, 'global_ddpms')
        os.makedirs(ddpm_dir, exist_ok=True)
        
        for d_name in self.global_ddpm_states.keys():
            ddpm_checkpoint = {
                'ddpm': self.global_ddpm_states[d_name]
            }
            ddpm_save_path = os.path.join(ddpm_dir, f'{d_name}_DDPM.pth')
            torch.save(ddpm_checkpoint, ddpm_save_path)
            self.logger.log(f"[Server] Global DDPM for {d_name} saved to {ddpm_save_path}")


        clients_dir = os.path.join(self.logger.log_dir, f'clients_last_round_checkpoints')
        os.makedirs(clients_dir, exist_ok=True)

        for client in self.clients:
            arch_name = getattr(client, 'model_name', 'Unknown')
            client_path = os.path.join(clients_dir, f'client_model_{client.dataset_name}_c{client.id}_{arch_name}.pth')
            torch.save(client.model.state_dict(), client_path)
            
        self.logger.log(f"[Server] All {len(self.clients)} clients saved in {clients_dir}/")