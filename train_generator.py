import argparse
import os
import torch
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import numpy as np

from data.datasets import get_raw_dataset_transform, get_readable_class_names
from utils.logger import Logger
from utils.nets import (
    ConditionalImageGenerator, 
    MLP, CNN, ResNet, BasicBlock, 
    MobileNetV2, MobileNetV3, 
    LeNet, AlexNet, 
    ShuffleNetV2, SqueezeNet     
)
from utils.loss import Gen_DiversityLoss, total_variation_loss, BNSM_Hook, get_bn_loss, CAM_Hook


def register_mapping(local_id_to_global_id, d1, l1, d2, l2):
    """
    將 (Dataset1, Label1) 與 (Dataset2, Label2) 連結到同一個 Global ID。
    """
    gid = None
    
    # Check d1:l1
    if d1 in local_id_to_global_id and l1 in local_id_to_global_id[d1]:
            gid = local_id_to_global_id[d1][l1]
    
    # Check d2:l2
    if gid is None and d2 in local_id_to_global_id and l2 in local_id_to_global_id[d2]:
            gid = local_id_to_global_id[d2][l2]
            
    # 如果兩邊都沒有，分配一個新的 GID
    if gid is None:
            current_max = -1
            for d in local_id_to_global_id:
                if local_id_to_global_id[d]:
                    current_max = max(current_max, max(local_id_to_global_id[d].values()))
            gid = current_max + 1
            
    if d1 not in local_id_to_global_id: local_id_to_global_id[d1] = {}
    local_id_to_global_id[d1][l1] = gid
    
    if d2 not in local_id_to_global_id: local_id_to_global_id[d2] = {}
    local_id_to_global_id[d2][l2] = gid


def complete_missing_ids(local_id_to_global_id, label_space_meta):
    """
    補全剩下的 ID (沒有 Mapping 到的類別，給予新的 unique Global ID)
    """
    for d_id in label_space_meta.keys():
        if d_id not in local_id_to_global_id:
            local_id_to_global_id[d_id] = {}
        
        class_names = label_space_meta[d_id]
        
        for l_id in range(len(class_names)):
            if l_id not in local_id_to_global_id[d_id]:
                current_max = -1
                for d in local_id_to_global_id:
                    if local_id_to_global_id[d]:
                        current_max = max(current_max, max(local_id_to_global_id[d].values()))
                new_gid = current_max + 1
                
                local_id_to_global_id[d_id][l_id] = new_gid
    return local_id_to_global_id


def test_discover_mappings_with_real_images(local_id_to_global_id, clients_dict, test_loaders_dict, label_space_meta, device, logger):
    logger.log("\n[Testing] Testing Cross-Dataset Label Mappings with REAL images ...")

    dataset_ids = list(clients_dict.keys())
    entropy_threshold = 2.0

    # 建立一個輔助功能，用來從 Dataloader 中依照指定的 label 取出 N 張真實圖片
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
        return images_cat.to(device)

    for i, src_id in enumerate(dataset_ids):
        for j, tgt_id in enumerate(dataset_ids):
            if i == j: continue 

            tgt_clients = clients_dict.get(tgt_id, [])
            src_clients = clients_dict.get(src_id, [])

            src_names = label_space_meta[src_id]
            tgt_names = label_space_meta[tgt_id]
            
            src_loader = test_loaders_dict[src_id] 
            tgt_loader = test_loaders_dict[tgt_id]

            for label_idx, label_name in enumerate(src_names):
                # 1. 取得 Data A 真實圖片
                img_src_real = get_real_images(src_loader, label_idx, num_samples=128)
                if img_src_real is None:
                    continue

                # 2. 讓 Data B 的 Models 去預測這批 Data A 的圖片
                with torch.no_grad():
                    avg_probs_tgt = 0
                    for c_model in tgt_clients:
                        c_model.eval()
                        _, logits = c_model(img_src_real)
                        avg_probs_tgt += F.softmax(logits, dim=1).mean(dim=0)
                    avg_probs_tgt /= len(tgt_clients)

                    entropy_tgt = -torch.sum(avg_probs_tgt * torch.log(avg_probs_tgt + 1e-8)).item()
                    pred_tgt_idx = torch.argmax(avg_probs_tgt).item()
                    pred_tgt_name = tgt_names[pred_tgt_idx]
                    tgt_conf = avg_probs_tgt[pred_tgt_idx].item()

                    if entropy_tgt > entropy_threshold:
                        logger.log(f"{src_id}:'{label_name}' -> {tgt_id} | predict:'{pred_tgt_name}' | Entropy:{entropy_tgt:.2f} ⚠️ [Filtered: High Uncertainty]")
                        continue
                    else: 
                        logger.log(f"{src_id}:'{label_name}' -> {tgt_id} | predict:'{pred_tgt_name}' | Entropy:{entropy_tgt:.2f}")

                    # 3. 取得 Data B 對應預測類別的真實圖片
                    img_tgt_real = get_real_images(tgt_loader, pred_tgt_idx, num_samples=32)
                    if img_tgt_real is None:
                        continue

                    # 4. 把 Data B 的照片丟回去給 Data A 的模型預測
                    avg_probs_src = 0
                    for c_model in src_clients:
                        c_model.eval()
                        _, logits = c_model(img_tgt_real)
                        avg_probs_src += F.softmax(logits, dim=1).mean(dim=0)
                    avg_probs_src /= len(src_clients)

                    entropy_src = -torch.sum(avg_probs_src * torch.log(avg_probs_src + 1e-8)).item()
                    pred_src_cycle_idx = torch.argmax(avg_probs_src).item()
                    pred_src_cycle_name = src_names[pred_src_cycle_idx]
                    conf_src = avg_probs_src[pred_src_cycle_idx].item()

                    if entropy_src > entropy_threshold:
                        match_symbol = "❓" # 標記為存疑
                        logger.log(f"  -> Check -> {src_id} | Predict:'{pred_src_cycle_name}' | {match_symbol} | Entropy:{entropy_src:.2f} ⚠️ [Filtered: Weak Cycle-Back]")
                    else:
                        match_symbol = "✅" if pred_src_cycle_idx == label_idx else "❌"
                        if match_symbol == "✅":
                            logger.log(f"[Real Img Match] {src_id}:'{label_name}' <==> {tgt_id}:'{pred_tgt_name}' ✅ ")
                            register_mapping(local_id_to_global_id, src_id, label_idx, tgt_id, pred_tgt_idx)

def img_train_dataset_generators(clients_dict, label_space_meta, dataset_meta, device, logger):
    logger.log("[Testing] Training Per-Dataset Generators Offline ...")

    div_loss_fn = Gen_DiversityLoss(metric='l1').to(device)

    feat_gen_noise_dim = 128

    gen_epochs = 2000
    gen_lr = 2e-3

    fedted_beta = 0.0
    alpha_ce = 1.0
    alpha_ent = 0.0
    tv_weight = 0.005

    generators_dict = {}
    
    for ls_id, dataset_clients in clients_dict.items():
        if len(dataset_clients) == 0:
            continue

        class_names = label_space_meta[ls_id]
        num_local_classes = len(class_names)

        generator = ConditionalImageGenerator(
            num_classes=num_local_classes, 
            noise_dim=feat_gen_noise_dim,
            img_channels=3, 
            img_size=32   
        ).to(device)
        optimizer = torch.optim.Adam(generator.parameters(), lr=gen_lr)
        
        for client_model in dataset_clients:
            client_model.eval() 
            for param in client_model.parameters():      
                param.requires_grad = False

            for module in client_model.modules():
                if hasattr(module, 'inplace'):
                    module.inplace = False

        generator.train()
        batch_size = 64
        epoch_loss_tracker = []

        for epoch in tqdm(range(gen_epochs), colour='blue', ncols=100, desc=f"Gen:{ls_id}"):
            class_order = torch.randperm(num_local_classes).tolist()

            #for c_idx in class_order:
            for _ in range(10):
                batch_labels = torch.randint(0, num_local_classes, (batch_size,)).to(device)
                z_noise = torch.randn(batch_size, feat_gen_noise_dim).to(device)

                optimizer.zero_grad()
                gen_imgs = generator(z_noise, batch_labels)
                gen_imgs.retain_grad()
                
                div_loss = div_loss_fn(gen_imgs.view(batch_size, -1), z_noise)
                
                total_cls_loss = 0.0
                total_ent_loss = 0.0

                for client_model in dataset_clients:
                    
                    _, logits = client_model(gen_imgs)
                    
                    total_cls_loss += F.cross_entropy(logits, batch_labels)

                    probs = F.softmax(logits, dim=1)
                    log_probs = F.log_softmax(logits, dim=1)
                    total_ent_loss += -torch.sum(probs * log_probs, dim=1).mean()

                cls_loss = total_cls_loss / len(dataset_clients)
                ent_loss = total_ent_loss / len(dataset_clients)
                tv_loss = total_variation_loss(gen_imgs) 

                loss = (alpha_ce * cls_loss) + (alpha_ent * ent_loss) + (fedted_beta * div_loss) + (tv_weight * tv_loss)
                loss.backward()
                optimizer.step()
                epoch_loss_tracker.append(loss.item())

        avg_loss = sum(epoch_loss_tracker) / len(epoch_loss_tracker)
        logger.log(f"     Dataset [{ls_id}] Generator | Epochs: {gen_epochs} | Final Loss: {avg_loss:.4f}")

        generators_dict[ls_id] = generator

        # Draw
        generator.eval()
        with torch.no_grad():
            os.makedirs('test_img', exist_ok=True) 
            save_dir = logger.log_dir

            num_samples_per_class = 1 
            num_classes_to_plot = num_local_classes
            sample_labels = torch.arange(num_classes_to_plot).repeat(num_samples_per_class).to(device)
            sample_z = torch.randn(num_samples_per_class * num_classes_to_plot, feat_gen_noise_dim).to(device)
            
            sample_imgs = generator(sample_z, sample_labels)
            sample_imgs = (sample_imgs * 0.5 + 0.5).clamp(0, 1).cpu().numpy()

            cols = min(10, num_classes_to_plot)
            rows = math.ceil(num_classes_to_plot / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
            axes_flat = [axes] if rows * cols == 1 else axes.flatten()
            
            for idx in range(num_classes_to_plot):
                ax = axes_flat[idx]
                img = sample_imgs[idx] 
                img = np.transpose(img, (1, 2, 0))
                ax.imshow(img)
                ax.set_title(str(class_names[idx]), fontsize=12)
                ax.axis('off') 

            for idx in range(num_classes_to_plot, len(axes_flat)):
                axes_flat[idx].axis('off')

            plt.tight_layout()
            save_path = os.path.join(save_dir, f'{ls_id}_offline_generated_samples.png')
            plt.savefig(save_path)
            plt.close() 
            logger.log(f"     -> Offlline Sample images saved to {save_path}")

    logger.log("[Testing] Offline Per-Dataset Generators trained.")

    return generators_dict


def img_discover_label_mappings(local_id_to_global_id, generators_dict, clients_dict, label_space_meta, device, logger):
    logger.log("[Testing] Discovering Cross-Dataset Label Mappings ...")
    dataset_ids = list(generators_dict.keys())
    feat_gen_noise_dim = 128
    entropy_threshold = 2.0

    for i, src_id in enumerate(dataset_ids):
        for j, tgt_id in enumerate(dataset_ids):
            if i == j: continue 
            
            tgt_clients = clients_dict.get(tgt_id, [])
            src_clients = clients_dict.get(src_id, [])

            src_gen = generators_dict[src_id]
            tgt_gen = generators_dict[tgt_id]

            src_names = label_space_meta[src_id]
            tgt_names = label_space_meta[tgt_id]

            src_gen.eval()
            tgt_gen.eval()

            for label_idx, label_name in enumerate(src_names):
                num_samples = 128
                z = torch.randn(num_samples, feat_gen_noise_dim).to(device)
                label_tensor = torch.tensor([label_idx] * num_samples).to(device)

                with torch.no_grad():
                    img_src = src_gen(z, label_tensor)

                    avg_probs_tgt = 0
                    for c_model in tgt_clients:
                        c_model.eval()
                        _, logits = c_model(img_src)
                        avg_probs_tgt += F.softmax(logits, dim=1).mean(dim=0)
                    avg_probs_tgt /= len(tgt_clients)

                    entropy_tgt = -torch.sum(avg_probs_tgt * torch.log(avg_probs_tgt + 1e-8)).item()
                    pred_tgt_idx = torch.argmax(avg_probs_tgt).item()
                    pred_tgt_name = tgt_names[pred_tgt_idx]
                    tgt_conf = avg_probs_tgt[pred_tgt_idx].item()

                    if entropy_tgt > entropy_threshold:
                        logger.log(f"{src_id}:'{label_name}' -> {tgt_id} | predict:'{pred_tgt_name}' | Entropy:{entropy_tgt:.2f} ⚠️ [Filtered: High Uncertainty]")
                        continue
                    else: 
                        logger.log(f"{src_id}:'{label_name}' -> {tgt_id} | predict:'{pred_tgt_name}' | Entropy:{entropy_tgt:.2f}")


                    z_cycle = torch.randn(num_samples, feat_gen_noise_dim).to(device)
                    label_tensor_tgt = torch.tensor([pred_tgt_idx] * num_samples).to(device)
                    img_tgt = tgt_gen(z_cycle, label_tensor_tgt)

                    avg_probs_src = 0
                    for c_model in src_clients:
                        c_model.eval()
                        _, logits = c_model(img_tgt)                                                                                                                                                                
                        avg_probs_src += F.softmax(logits, dim=1).mean(dim=0)
                    avg_probs_src /= len(src_clients)

                    entropy_src = -torch.sum(avg_probs_src * torch.log(avg_probs_src + 1e-8)).item()
                    pred_src_cycle_idx = torch.argmax(avg_probs_src).item()
                    pred_src_cycle_name = src_names[pred_src_cycle_idx]
                    conf_src = avg_probs_src[pred_src_cycle_idx].item()

                    if entropy_src > entropy_threshold:
                        match_symbol = "❓" # 標記為存疑
                        logger.log(f"  -> Check -> {src_id} | Predict:'{pred_src_cycle_name}' | {match_symbol} | Entropy:{entropy_src:.2f} ⚠️ [Filtered: Weak Cycle-Back]")
                    else:
                        match_symbol = "✅" if pred_src_cycle_idx == label_idx else "❌"
                        if match_symbol == "✅":
                            logger.log(f"[Match] {src_id}:'{label_name}' <==> {tgt_id}:'{pred_tgt_name}' ✅ ")
                            register_mapping(local_id_to_global_id, src_id, label_idx, tgt_id, pred_tgt_idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the directory containing .pth files")
    parser.add_argument("--device", type=str, default="cuda:1")
    args = parser.parse_args()

    logger = Logger(args)
    logger.log(f"Loading checkpoints from {args.model_path}")

    target_datasets = ['MNIST', 'EMNIST', 'CIFAR10'] 
    
    DATASET_META = { 
        'EMNIST': {'in_ch': 3, 'classes': 47,  'size': 32},
        'MNIST':  {'in_ch': 3, 'classes': 10,  'size': 32}, 
        'CIFAR10':  {'in_ch': 3, 'classes': 10,  'size': 32}, 
    }

    DATA_ROOT = './data/raw'

    label_space_meta = {}
    for d_name in target_datasets:
        label_space_meta[d_name] = get_readable_class_names(d_name, root=DATA_ROOT)

    clients_dict = {}

    for d_name in target_datasets:
        clients_dict[d_name] = [] # 改成 List，用來裝這一個 dataset 旗下的所有異構模型
        meta = DATASET_META[d_name]

        if not os.path.exists(args.model_path):
            logger.log(f"Path {args.model_path} doesn't exist.")
            continue

        for filename in os.listdir(args.model_path):
            # 檔名格式為 "client_model_CIFAR10_c22_MLP.pth" 這種類型
            # 因此只要確保檔名以 client_model_{d_name} 開頭即可
            if filename.startswith(f"client_model_{d_name}_c") and filename.endswith(".pth"):
                ckpt_path = os.path.join(args.model_path, filename)
                
                # 從檔名解析出架構名稱 (例如從 "client_model_CIFAR10_c22_MLP.pth" 去掉 ".pth" 後，以 "_" 切割取最後一部分)
                arch_name = filename.replace(".pth", "").split("_")[-1]

                # 根據檔名分配不同的架構
                if arch_name == 'MLP':
                    # 注意：MLP 需要傳入 img_size
                    model = MLP(in_channels=meta['in_ch'], num_classes=meta['classes'], img_size=meta['size']).to(args.device)
                elif arch_name == 'CNN':
                    model = CNN(in_channels=meta['in_ch'], num_classes=meta['classes']).to(args.device)
                elif arch_name == 'ResNet8':
                    model = ResNet(BasicBlock, [1, 1, 1, 0], in_channels=meta['in_ch'], num_classes=meta['classes']).to(args.device)
                elif arch_name == 'ResNet18':
                    model = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=meta['in_ch'], num_classes=meta['classes']).to(args.device)
                elif arch_name == 'MobileNetV2':
                    model = MobileNetV2(in_channels=meta['in_ch'], num_classes=meta['classes']).to(args.device)
                elif arch_name == 'MobileNetV3':
                    model = MobileNetV3(in_channels=meta['in_ch'], num_classes=meta['classes']).to(args.device)
                elif arch_name == 'LeNet':
                    model = LeNet(in_channels=meta['in_ch'], num_classes=meta['classes']).to(args.device)
                elif arch_name == 'AlexNet':
                    model = AlexNet(in_channels=meta['in_ch'], num_classes=meta['classes']).to(args.device)
                elif arch_name == 'ShuffleNet':
                    model = ShuffleNetV2(in_channels=meta['in_ch'], num_classes=meta['classes']).to(args.device)
                elif arch_name == 'SqueezeNet':
                    model = SqueezeNet(in_channels=meta['in_ch'], num_classes=meta['classes']).to(args.device)
                else:
                    logger.log(f"Warning: Unknown model architecture '{arch_name}' in {filename}. Skipping.")
                    continue

                # 載入權重並加入 Dict
                model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
                model.eval()
                clients_dict[d_name].append(model)
                logger.log(f"Loaded {arch_name} for {d_name} from {filename}")

        if len(clients_dict[d_name]) == 0:
            logger.log(f"WARNING: No models loaded for dataset: {d_name}.")


    test_loaders_dict = {}
    for d_name in target_datasets:
        test_dataset = get_raw_dataset_transform(name=d_name, root=DATA_ROOT, train=False)
        test_loaders_dict[d_name] = DataLoader(test_dataset, batch_size=128, shuffle=False)

    valid_datasets = [d for d, models in clients_dict.items() if len(models) > 0]
    
    if len(valid_datasets) > 1:
        local_id_to_global_id = {}

        # test_discover_mappings_with_real_images(local_id_to_global_id, clients_dict, test_loaders_dict, label_space_meta, args.device, logger)
    
        generators_dict = img_train_dataset_generators(clients_dict, label_space_meta, DATASET_META, args.device, logger)
        
        img_discover_label_mappings(local_id_to_global_id, generators_dict, clients_dict, label_space_meta, args.device, logger)

        local_id_to_global_id = complete_missing_ids(local_id_to_global_id, label_space_meta)

        logger.log("\n[Result] Constructed Local-to-Global ID Map:")
        for d_name, map_dict in local_id_to_global_id.items():
            logger.log(f"Dataset: {d_name}")
            for lid, gid in sorted(map_dict.items()):
                cname = label_space_meta[d_name][lid]
                logger.log(f"  Local ID {lid} ('{cname}') -> Global ID {gid}")

    else:
        logger.log("Not enough models loaded to perform cross-dataset testing.")

if __name__ == "__main__":
    main()