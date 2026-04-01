import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import *
import torch.nn.functional as F

from utils.train_utils import train_model, freeze, unfreeze
from utils.nets import TwinBranchNets, ConditionalGenerator
from trainer.BaseFL.client import Client as BaseClient

class Client(BaseClient):
    def __init__(self, fedted_lambda1=1.0, fedted_lambda2=1.0, **kwargs):
        super(Client, self).__init__(**kwargs)

        self.lambda1, self.lambda2 = fedted_lambda1, fedted_lambda2

        self.optim_name =  kwargs.get('optim', 'Adam')
        self.optim_lr = kwargs.get('optim_lr', 1e-3)

        self.distill_lr = kwargs.get('distill_lr', 1e-3)
        self.distill_optimizer = eval(self.optim_name)(
            list(self.model.feature_extractor.parameters()) + list(self.model.adapter.parameters()),
            self.distill_lr)

        # optimizer for feature_extractor
        self.optimizer_fe = eval(self.optim_name)(
            filter(lambda p: p.requires_grad, self.model.feature_extractor.parameters()),
            self.optim_lr)
        
        # optimizer for classifiers
        unfreeze(self.model)
        freeze(self.model.feature_extractor)
        self.optimizer_cls = eval(self.optim_name)(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            self.optim_lr)
        unfreeze(self.model)

        self.global_feat_gen = None
        self.feat_gen_noise_dim = kwargs.get('feat_gen_noise_dim', 128) 

        self.distill_epochs = kwargs.get('distill_epochs', 1) 
        self.kd_loss_fn = nn.MSELoss() 

        self.prox_z = None
        self.prox_y = None


    def local_fine_tune(self):
        # one shot fine tune to match feature extractor and local classifier
        self.model.use_twin = True
        BaseClient.update(self)
        self.model.use_twin = False

    def update(self, epochs=1):
        # 1. distill feature extractor by generator
        if (self.glob_iter + 1) >= self.start_mapping_epoch: 
            if self.global_feat_gen is not None:
                self.distill_feature_extractor()
            else:
                pass

        # 2. decouple train feature extractor and classifier
        self.update_twin_branch(epochs)

    def distill_feature_extractor(self):
        self.model.to(self.device)
        self.model.train()

        self.global_feat_gen.to(self.device)
        self.global_feat_gen.eval()

        feature_extractor = self.model.feature_extractor
        adapter = self.model.adapter
        unfreeze(feature_extractor)
        unfreeze(adapter)

        for epoch in range(self.distill_epochs):
            for x, y in self.train_loader:
                if y.size(0) == 1: continue  # generator used a bn

                x, y = x.to(self.device), y.to(self.device)

                global_y_list = [self.local_id_to_global_id[label.item()] for label in y]
                global_y = torch.tensor(global_y_list, dtype=torch.long).to(self.device)

                with torch.no_grad():
                    z_noise = torch.randn(y.size(0), self.feat_gen_noise_dim).to(self.device)
                    z_teacher = self.global_feat_gen(z_noise, global_y)

                native_feat = feature_extractor(x)           
                native_feat = torch.flatten(native_feat, 1)  
                z_student = adapter(native_feat)

                loss = self.kd_loss_fn(z_student, z_teacher)

                self.distill_optimizer.zero_grad()
                loss.backward()
                self.distill_optimizer.step()

        self.model.to('cpu')
        torch.cuda.empty_cache()

    def update_twin_branch(self, epochs):
        # step 1. model init
        self.model.to(self.device)

        # to facility use
        feature_extractor = self.model.feature_extractor
        adapter = self.model.adapter
        classifier_g = self.model.classifier
        classifier_p = self.model.twin_classifier

        # step 2. decouple train

        # 1. train feature extractor
        self.model.train()
        freeze(self.model)
        classifier_p.eval()
        classifier_g.eval()
        
        unfreeze(feature_extractor)
        unfreeze(adapter)

        for epoch in range(epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                label_counts = self.count_labels(y)
                label_prob = torch.tensor(label_counts).to(self.device) / sum(label_counts)

                native_feat = feature_extractor(x)
                native_feat = torch.flatten(native_feat, 1)  # (B, 64)
                z = adapter(native_feat)
                y_g = classifier_g(z) * label_prob
                y_p = classifier_p(z)

                # c. calculate loss
                loss_g = self.local_loss_fn(y_g, y)
                loss_p = self.local_loss_fn(y_p, y)
                loss = loss_g + loss_p

                # d. backward & step optim
                self.optimizer_fe.zero_grad()
                loss.backward()
                self.optimizer_fe.step()

        # 2. train generic and personalized branch in multitask way
        self.model.train()
        unfreeze(self.model)
        feature_extractor.eval()
        adapter.eval()
        freeze(feature_extractor)
        freeze(adapter)

        for epoch in range(epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                label_counts = self.count_labels(y)
                label_prob = torch.tensor(label_counts).to(self.device) / sum(label_counts)

                with torch.no_grad():
                    native_feat = feature_extractor(x)
                    native_feat = torch.flatten(native_feat, 1)
                    z = adapter(native_feat)

                #z = feature_extractor(x)
                y_g = classifier_g(z) * label_prob
                y_p = classifier_p(z)

                # c. calculate loss
                loss_g = self.local_loss_fn(y_g, y)
                loss_p = self.local_loss_fn(y_p, y)
                loss_norm = self.norm_loss_fn(classifier_g, classifier_p)
                loss = loss_p + self.lambda1*loss_g + self.lambda2*loss_norm

                self.optimizer_cls.zero_grad()
                loss.backward()
                self.optimizer_cls.step()

        # step 3. release gpu resource
        self.model.to('cpu')
        torch.cuda.empty_cache()

    def count_labels(self, y):
        label_counts = [0] * self.local_num_classes
        for i in range(self.local_num_classes):
            idx = torch.nonzero(y == i).view(-1)
            label_counts[i] += len(idx)
        return label_counts

    @staticmethod
    def norm_loss_fn(model1, model2):
        m1_params = [param.view(-1) for param in model1.parameters()]
        m2_params = [param.view(-1) for param in model2.parameters()]

        m1_params = torch.cat(m1_params, dim=0)
        m2_params = torch.cat(m2_params, dim=0)

        assert m1_params.size(0) == m2_params.size(0)

        loss = F.mse_loss(m1_params, m2_params)
        return loss

