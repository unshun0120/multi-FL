import torch
import copy
from collections import OrderedDict, defaultdict
import os
import csv
import numpy as np

from trainer.BaseFL.server import Server as BaseServer
from utils.plotting import plot_accuracy_curves
from label_mapping.label_mapping_utils import (
    label_mapping, 
    evaluate_mapping_results, 
    get_gen_images, 
    get_real_images, 
    clear_image_caches
)

class Server(BaseServer):
    def __init__(self, **kwargs):
        super(Server, self).__init__(**kwargs)

        self.global_gen_states = {}
        self.global_dis_states = {}


    def run(self):
        self.logger.log("")
        self.logger.log("=" * 50)
        self.logger.log(f"Start {self.global_rounds} rounds training by {self.algorithm}")

        for r in range(self.global_rounds):
            self.glob_iter = r

            self.sample_clients()
            #self.distribute_model()
            self.local_update()

            #self.evaluate_mapping()

            if (r + 1) % self.test_interval == 0:
                self.evaluate_private()
                #self.record_metric()

            #self.aggregate()

            if r+1 == 40:
                self.save_model() 

        # self.save_metric()

        self.save_model()
        plot_accuracy_curves(self.dataset_acc_history, self.logger.log_dir, self.args, self.global_rounds, self.dirichlet_alpha)

    def evaluate_mapping(self):
        current_round = self.glob_iter + 1
        self.logger.log(f"--- Local Client Label Mapping Evaluation (Round {current_round}) ---")
        
        local_clients_dict = {}
        local_label_space_meta = {}
        gan_dict = {}
        test_loaders_dict = {}
        active_local_ids = []
        
        for client in self.clients:
            c_name = f"c{client.id}_{client.dataset_name}"
            
            local_clients_dict[c_name] = [client.model]
            
            local_label_space_meta[c_name] = self.label_space_meta.get(client.dataset_name, getattr(client, 'class_name_set', []))
            
            gan_dict[c_name] = client.generator
            test_loaders_dict[c_name] = getattr(client, 'test_loader', client.train_loader)
            
            active_local_ids.append(c_name)

        if not active_local_ids:
            return 
            
        dynamic_entropy_ratios = [round(x, 2) for x in np.arange(0.10, 1.05, 0.05)]
        
        class DummyLogger:
            def log(self, msg): pass

        clear_image_caches()
        
        self.label_mapping_evaluation("GAN", get_gen_images, {'gen_dict': gan_dict}, local_clients_dict, local_label_space_meta, active_local_ids, dynamic_entropy_ratios, current_round, DummyLogger())
        self.label_mapping_evaluation("Real", get_real_images, {'test_loaders': test_loaders_dict}, local_clients_dict, local_label_space_meta, active_local_ids, dynamic_entropy_ratios, current_round, DummyLogger())

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

        gan_dir = os.path.join(self.logger.log_dir, f'client_gans_{self.glob_iter+1}')
        os.makedirs(gan_dir, exist_ok=True)
        
        for client in self.clients:
            gan_checkpoint = {
                'generator': client.generator.state_dict(),
                'discriminator': client.discriminator.state_dict()
            }
            gan_save_path = os.path.join(gan_dir, f'client_c{client.id}_{client.dataset_name}_GAN.pth')
            torch.save(gan_checkpoint, gan_save_path)
            
        clients_dir = os.path.join(self.logger.log_dir, f'clients_last_round_checkpoints')
        os.makedirs(clients_dir, exist_ok=True)

        for client in self.clients:
            arch_name = getattr(client, 'model_name', 'Unknown')
            client_path = os.path.join(clients_dir, f'client_model_{client.dataset_name}_c{client.id}_{arch_name}.pth')
            torch.save(client.model.state_dict(), client_path)
            
        self.logger.log(f"[Server] All {len(self.clients)} clients saved in {clients_dir}/")