import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm

from models.hetero_model import ConditionalGenerator, Classifier


class Server:
    def __init__(self, args):
        self.args = args
        self.device = args.device 

        if args.gen_optim == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.client_lr, weight_decay=1e-4)
        else:    
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.client_lr, momentum=0.9)

        self.global_classifiers = {}
        self.label_to_global_id = {} # {'dog': 0, 'cat': 1, ...}
        self.global_id_to_label = {} # {0: 'dog', 1: 'cat', ...}
        self.label_space_meta = {} # { 'ls_id': ['dog', 'cat', ...] }

        self.generator = None
        self.shared_classifier = None

    def train_global_shared_classifier(self, logger):
        num_global_classes = len(self.label_to_global_id)
        if self.shared_classifier is None:
            print(f"[Server] Initializing Shared Classifier for {num_global_classes} classes...")
            self.shared_classifier = Classifier(
                input_dim=self.args.global_feature_dim,
                num_classes=num_global_classes
            ).to(self.device)

        self.generator.eval()       
        self.shared_classifier.train() 

        all_global_ids = list(self.global_id_to_label.keys())

        logger.log(f"[Server] Shared Classifier Training ({self.args.server_epochs} epochs)...")
        for epoch in range(self.args.server_epochs):
            epoch_loss = 0
            random.shuffle(all_global_ids)

            for i in range(0, len(all_global_ids), self.args.batch_size):
                batch_ids = all_global_ids[i : i + self.args.batch_size]
                current_bs = len(batch_ids)
                
                labels_input = torch.tensor(batch_ids).to(self.device)
                z = torch.randn(current_bs, self.args.noise_dim).to(self.device)
                
                with torch.no_grad():
                    gen_feat = self.generator(z, labels_input)
                
                self.optimizer.zero_grad()
                
                logits = self.shared_classifier(gen_feat)
                loss = F.cross_entropy(logits, labels_input)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            logger.log(f"  [Shared Classifier] Epoch {epoch+1} Loss: {epoch_loss:.4f}")

    def train_generator(self, logger):
        if self.generator is None:
            print(f"[Server] Initializing Conditional Generator...")
            self.generator = ConditionalGenerator(
                num_global_classes=len(self.label_to_global_id),
                noise_dim=self.args.noise_dim,
                output_dim=self.args.global_feature_dim 
            ).to(self.device)

        logger.log("[Server] Training Generator...")
        self.generator.train()

        all_global_ids = list(self.global_id_to_label.keys())

        for epoch in tqdm(range(self.args.server_gen_epochs), colour="blue"):
            epoch_loss = 0
            random.shuffle(all_global_ids)
            
            for i in range(0, len(all_global_ids), self.args.batch_size):
                batch_ids = all_global_ids[i : i + self.args.batch_size]
                curr_batch = len(batch_ids)

                labels_input = torch.tensor(batch_ids).to(self.device)
                z = torch.randn(curr_batch, self.args.noise_dim).to(self.device)
                gen_feat = self.generator(z, labels_input)
                
                batch_loss_expert = 0
                batch_loss_observer = 0
                
                self.optimizer.zero_grad()

                for ls_id, classifier in self.global_classifiers.items():
                    classifier.eval()
                    logits = classifier(gen_feat)

                    group_class_names = self.label_space_meta[ls_id]

                    # 看這個classifier認不認識batch裡的global_id
                    target_list = []
                    for gid in batch_ids:
                        g_name = self.global_id_to_label[gid]
                        if g_name in group_class_names:
                            # 認識 -> 存他資料集原本的local id
                            target_list.append(group_class_names.index(g_name))
                        else:
                            # 不認識 -> 存-1
                            target_list.append(-1)

                    target_tensor = torch.tensor(target_list).to(self.device)
                    # boolean tensor
                    mask_expert = (target_tensor != -1)

                    # expert (classification loss) 
                    if mask_expert.any():
                        # 取mask=true的算loss
                        loss_ce = F.cross_entropy(logits[mask_expert], target_tensor[mask_expert])
                        batch_loss_expert += loss_ce

                    # observer (logit distillation)
                    if (~mask_expert).any():
                        # 取mask=false的算loss
                        probs = F.log_softmax(logits[~mask_expert], dim=1)
                        uniform_target = torch.full_like(probs, 1.0 / logits.size(1))
                        
                        loss_kl = F.kl_div(probs, uniform_target, reduction='batchmean')
                        batch_loss_observer += loss_kl

                # normalization
                num_classifiers = len(self.global_classifiers)
                if num_classifiers > 0:
                    batch_loss_expert /= num_classifiers
                    batch_loss_observer /= num_classifiers
                
                total_loss = batch_loss_expert + (self.args.gen_observer_weight * batch_loss_observer)

                total_loss.backward()
                self.optimizer.step()
                
                epoch_loss += total_loss.item()

            logger.log(f"  Epoch {epoch+1}/{self.args.global_epochs} | Loss: {epoch_loss:.4f}")

        logger.log("[Server] Generator training finished.")

    def _update_global_label_map(self, class_names):
        for name in class_names:
            if name not in self.label_to_global_id:
                new_id = len(self.label_to_global_id)
                self.label_to_global_id[name] = new_id
                self.global_id_to_label[new_id] = name

    def _get_label_space_id(self, class_names):
        """
        把相同類別名稱集合轉成相同unique ID
        """
        # 做sort確保順序不影響ID (e.g., ['a', 'b'] == ['b', 'a'])
        sorted_names = sorted(list(class_names))
        
        # MD5 Hash 
        # return hashlib.md5(name_str.encode('utf-8')).hexdigest()
        
        return str(sorted_names)

    def aggregate_clients(self, client_uploads, logger):
        """
        weights_sum = {
            # --- 第一個 Label Space (例如 MNIST) ---
            "['0', '1', ..., '9']": {  # key 是 ls_id (字串)
                # value 是一個字典，存這組所有 Client 的參數總和
                "weight": tensor([[...], ...]),  # 形狀 [10, 256], 是 Client 1 + Client 2 的權重
                "bias":   tensor([...])          # 形狀 [10],      是 Client 1 + Client 2 的偏差
            },
            # --- 第二個 Label Space (例如 CIFAR-10) ---
            "['airplane', ..., 'truck']": { # key 是 ls_id
                # 因為只有 Client 3, 所以就是 Client 3 的參數
                "weight": tensor([[...], ...]),  # 形狀 [10, 256]
                "bias":   tensor([...])          # 形狀 [10]
            }
        }
        """
        weights_sum = defaultdict(lambda: defaultdict(float))
        # 每個label space(類別名稱集合)在這一輪global round有幾個client上傳
        count = defaultdict(int)

        print(f"[Server] Aggregating from {len(client_uploads)} clients (grouped by label space)...") 
        for upload in client_uploads:
            # 把client的class_names類別名稱集合轉成label space ID(同label set -> 同ID)
            c_names = upload['class_names']
            self._update_global_label_map(c_names)
            ls_id = self._get_label_space_id(c_names)

            # print(ls_id)
            # e.g. ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]、
            # ["Ankle boot", "Bag", "Coat", "Dress", "Pullover", "Sandal", "Shirt", "Sneaker", "T-shirt/top", "Trouser"]

            cls_state_dict = upload['classifier_state_dict']
            input_feature_dim = self.args.global_feature_dim

            # 如果這個label space是第一次出現
            if ls_id not in self.global_classifiers:
                print(f"[Server] New Label Space Group: {ls_id}")
                num_classes = cls_state_dict['weight'].shape[0]

                self.global_classifiers[ls_id] = Classifier(
                    input_dim=input_feature_dim, 
                    num_classes=num_classes
                ).to(self.device)

            # FedAvg client classifier (Sum)
            count[ls_id] += 1
            for key, param in cls_state_dict.items():
                param = param.cpu()
                if isinstance(weights_sum[ls_id][key], float):
                    weights_sum[ls_id][key] = param
                else:
                    weights_sum[ls_id][key] += param

        # FedAvg client classifier (Average)
        for ls_id, model in self.global_classifiers.items():
            if count[ls_id] > 0:
                new_state_dict = {}
                for key in weights_sum[ls_id]:
                    new_state_dict[key] = weights_sum[ls_id][key] / count[ls_id]
                
                final_state_dict = {}
                for k, v in new_state_dict.items():
                    final_state_dict[k] = v.to(self.device)
                model.load_state_dict(final_state_dict)

    def get_global_classifier(self, client_class_names):
        ls_id = self._get_label_space_id(client_class_names)
        clf_weight = None

        if ls_id in self.global_classifiers:
            clf_weight = self.global_classifiers[ls_id].state_dict()
        
        return clf_weight
