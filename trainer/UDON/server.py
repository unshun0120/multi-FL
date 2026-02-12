import torch
import copy
from trainer.BaseFL.server import Server as BaseServer

class Server(BaseServer):
    def __init__(self, public_dataloader=None, **kwargs):
        super(Server, self).__init__(**kwargs)
        self.public_dataloader = public_dataloader
        
        # Identify layers to aggregate vs private
        # We only aggregate Backbone and Universal Projection
        self.shared_layers = ['backbone', 'universal_projection']

    def aggregate(self, local_weights):
        """
        Custom aggregation: Only aggregate shared layers.
        local_weights: list of state_dicts from clients.
        """
        # Initialize averaged weights with the structure of the first client's weights
        avg_weights = copy.deepcopy(self.model.state_dict())
        
        # Calculate dataset size ratios for weighted average
        total_samples = sum([n for n, _ in local_weights])
        
        # Iterate over all parameters
        for key in avg_weights.keys():
            # Check if this parameter belongs to shared layers
            if any(shared_name in key for shared_name in self.shared_layers):
                
                weighted_sum = 0
                for n_samples, client_state in local_weights:
                    weighted_sum += client_state[key] * n_samples
                
                avg_weights[key] = weighted_sum / total_samples
            else:
                # Keep server's current weight (or don't update) for private layers
                # For private layers, usually they are not overwritten by server in next round
                pass
                
        self.model.load_state_dict(avg_weights)
        
        # Optional: Train on Public Dataset (UDON feature)
        if self.public_dataloader:
            self.train_on_public_dataset()

    def train_on_public_dataset(self):
        """
        Train the aggregated global model (Universal parts) on public dataset.
        This aligns with 'Universal Embedding' idea where server has some 'Universal' data.
        """
        print("Server: Training on Public Dataset...")
        self.model.train()
        
        # We might only want to update backbone/universal proj, 
        # but we need a classifier head for public dataset.
        # This assumes the 'student_classifier' can handle public classes 
        # or we add a specific 'public_head' to the model.
        # For simplicity here, assume public data matches student classifier dimension (or pre-training logic).
        
        for epoch in range(1): # One epoch fine-tune
            for x, y in self.public_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                
                outputs = self.model(x)
                # Use Universal Student Logits
                logits = outputs['universal_student_logits']
                
                loss = self.loss_fn(logits, y)
                loss.backward()
                self.optimizer.step()

    def distribute_model(self):
        """
        Distribute only shared layers to clients.
        Clients keep their private layers (Teacher, Classifiers).
        """
        global_state = self.model.state_dict()
        
        for client in self.clients:
            client_state = client.model.state_dict()
            
            new_state = {}
            for key in client_state.keys():
                if any(shared_name in key for shared_name in self.shared_layers):
                    # Update shared parts from server
                    new_state[key] = global_state[key]
                else:
                    # Keep client's private parts
                    new_state[key] = client_state[key]
            
            client.model.load_state_dict(new_state)