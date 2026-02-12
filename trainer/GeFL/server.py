import torch
import copy
from collections import OrderedDict

from trainer.BaseFL.server import Server as BaseServer
from utils.nets import CCVAE

class Server(BaseServer):
    def __init__(self, clients, metric_type='accuracy', logger=None, **kwargs):
        super(Server, self).__init__(clients, metric_type, logger, **kwargs)
        self.algorithm_name = "GeFL"

        # Initialize Global Generator
        # We assume args has image info
        self.global_generator = CCVAE(
            num_classes=len(self.clients[0].class_names),
            latent_size=getattr(self.args, 'latent_size', 16),
            img_size=getattr(self.args, 'img_size', 32),
            channels=getattr(self.args, 'channels', 3)
        ).to(self.args.device)

    def train(self):
        """
        Main Loop for GeFL Server.
        Overrides BaseFL train loop to handle Generator aggregation.
        """
        self.logger.info(f"Start Training {self.algorithm_name}...")
        
        for round_idx in range(self.global_rounds):
            self.glob_current_iter = round_idx
            self.logger.info(f"--- Global Round {round_idx+1}/{self.global_rounds} ---")
            
            # 1. Client Selection
            self.selected_clients = self.select_clients() # Implemented in BaseServer
            
            # 2. Distribute Generator to Clients
            global_gen_state = self.global_generator.state_dict()
            for client in self.selected_clients:
                client.set_generator(copy.deepcopy(global_gen_state))
            
            # 3. Local Training
            client_gen_weights = []
            
            for client in self.selected_clients:
                # Client returns dict with 'generator_weights'
                result = client.train() 
                client_gen_weights.append((result['num_samples'], result['generator_weights']))
            
            # 4. Aggregate Generator (FedAvg)
            new_gen_weights = self.aggregate_weights(client_gen_weights)
            self.global_generator.load_state_dict(new_gen_weights)
            
            # 5. Evaluation (Evaluate clients' heterogeneous models)
            if round_idx % self.args.eval_interval == 0:
                self.evaluate_clients()

    def aggregate_weights(self, weights_list):
        """
        FedAvg aggregation for Generator.
        weights_list: list of (num_samples, state_dict)
        """
        total_samples = sum([w[0] for w in weights_list])
        avg_params = OrderedDict()
        
        for name in weights_list[0][1].keys():
            avg_params[name] = torch.zeros_like(weights_list[0][1][name], dtype=torch.float32)
            
            for num_samples, params in weights_list:
                avg_params[name] += params[name] * (num_samples / total_samples)
                
        return avg_params

    def evaluate_clients(self):
        """Evaluate each client logic."""
        accuracies = []
        for client in self.clients: # Evaluate all clients or selected? Usually all.
            # BaseFL client might have evaluate method
            # Assuming client.model is the target model
            acc, loss = client.evaluate() # BaseFL client.evaluate()
            accuracies.append(acc)
            
        avg_acc = sum(accuracies) / len(accuracies)
        self.logger.info(f"Round {self.glob_current_iter} Avg Accuracy: {avg_acc:.2f}%")