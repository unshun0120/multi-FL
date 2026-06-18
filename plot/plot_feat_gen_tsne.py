import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from utils.nets import ConditionalGenerator

def get_tsne_data_from_checkpoint(checkpoint_path, samples_per_class=200):
    print(f"\n--- Checkpoint: {checkpoint_path} ---")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    global_registry = checkpoint['global_registry']
    exp_conf = checkpoint['exp_conf']
    noise_dim = exp_conf.get('feat_gen_noise_dim', 100)
    
    global_to_datasets = {}
    for d_name, mapping in global_registry.items():
        for l_id, g_id in mapping.items():
            if g_id not in global_to_datasets:
                global_to_datasets[g_id] = []
            if d_name not in global_to_datasets[g_id]:
                global_to_datasets[g_id].append(d_name)
            
    num_global_classes = max(global_to_datasets.keys()) + 1

    generator = ConditionalGenerator(
            num_global_classes=num_global_classes, 
            noise_dim=noise_dim,
            output_dim=exp_conf.get('global_feature_dim', 256)
        ).to('cpu')
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()

    features_list = []
    source_names_list = []
    
    with torch.no_grad():
        for g_id in range(num_global_classes):
            if g_id not in global_to_datasets:
                continue

            source_name = " & ".join(sorted(global_to_datasets[g_id]))
            
            z = torch.randn(samples_per_class, noise_dim)
            y = torch.full((samples_per_class,), g_id, dtype=torch.long)
            
            generated_features = generator(z, y) 
            if generated_features.dim() > 2: generated_features = generated_features.view(samples_per_class, -1)

            features_list.append(generated_features.cpu().numpy())
            source_names_list.extend([source_name] * samples_per_class)

    X = np.concatenate(features_list, axis=0)
    Y_dataset = np.array(source_names_list)

    print(f" t-SNE ... ({X.shape[0]} data points)...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    X_2d = tsne.fit_transform(X)
    
    return X_2d, Y_dataset, list(set(source_names_list))

def plot_dual_tsne(ckpt_dict, output_path="dual_generator_tsne.png"):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    dataset_colors = {
        'MNIST': '#1f77b4', 
        'EMNIST': '#ff7f0e', 
        'FashionMNIST': '#2ca02c',
        'CIFAR10': '#2ca02c', 
        'CIFAR100': '#d62728',
        'EMNIST & MNIST': '#9467bd'  
    }
    
    for idx, (method_name, ckpt_path) in enumerate(ckpt_dict.items()):
        ax = axes[idx]
        
        if not os.path.exists(ckpt_path):
            ax.set_title(f"{method_name} - Checkpoint Not Found", color='red')
            continue
            
        X_2d, Y_dataset, unique_sources = get_tsne_data_from_checkpoint(ckpt_path)

        unique_sources.sort(key=lambda x: "&" in x)
        
        for d_name in unique_sources:
            mask = (Y_dataset == d_name)
            if not np.any(mask): continue
            
            color = dataset_colors.get(d_name, 'gray')
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1], 
                c=color, label=d_name, 
                alpha=0.6, s=15, edgecolors='w', linewidth=0.5
            )
            
        ax.set_title(f'{method_name} Generator Features', fontsize=16, fontweight='bold')
        ax.set_xlabel('x', fontsize=12)
        if idx == 0:
            ax.set_ylabel('y', fontsize=12)
            
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(title='Dataset', loc='best', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

# ==========================================
# 3. 執行區塊
# ==========================================
if __name__ == "__main__":
    checkpoint_paths = {
        "FedTED": "logs/2026-05-12_07-04-29/FedTED/config_checkpoints.pth",
        "FedTED_dir": "logs/2026-05-12_17-34-01/FedTED_dir/config_checkpoints.pth"
        
        # "FedTED": "logs/2026-05-12_02-19-31/FedTED/config_checkpoints.pth",
        # "FedTED_dir": "logs/2026-05-12_12-45-34/FedTED_dir/config_checkpoints.pth"
    }
    
    plot_dual_tsne(checkpoint_paths, output_path="./plot/feat_gen_tsne/mnist+emnist+cifar10.png")