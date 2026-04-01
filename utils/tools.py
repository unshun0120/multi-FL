import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_generated_features_tsne(global_models, label_space_meta, device, logger=None, num_samples_per_class=100, noise_dim=128, save_path='tsne_generated_features.png'):
    """
    """
    def log_msg(msg):
        if logger:
            logger.log(msg)
        else:
            print(msg)

    log_msg("[Utils] Visualizing generated features with t-SNE...")
    
    all_features = []
    all_labels = []
    all_datasets = []
    
    datasets_to_plot = ['MNIST', 'EMNIST']
    
    # 1. 產生所有特徵
    for ds_name in datasets_to_plot:
        if ds_name not in global_models or 'generator' not in global_models[ds_name]:
            log_msg(f"  [Warning] Generator for {ds_name} not found. Skipping.")
            continue
        
        gen = global_models[ds_name]['generator']
        gen.eval()
        
        num_classes = len(label_space_meta[ds_name])
        
        with torch.no_grad():
            for c in range(num_classes):
                labels = torch.full((num_samples_per_class,), c, dtype=torch.long).to(device)
                z = torch.randn(num_samples_per_class, noise_dim).to(device)
                
                feats = gen(z, labels).cpu().numpy()
                
                all_features.append(feats)
                all_labels.extend([c] * num_samples_per_class)
                all_datasets.extend([ds_name] * num_samples_per_class)
                
    if len(all_features) == 0:
        log_msg("  [Error] No features generated. Cannot plot t-SNE.")
        return

    all_features = np.vstack(all_features)
    all_labels = np.array(all_labels)
    all_datasets = np.array(all_datasets)
    
    log_msg(f"  -> Running t-SNE on {len(all_features)} samples...")
    
    # 2. 執行 t-SNE 降維
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    feats_2d = tsne.fit_transform(all_features)
    
    # 3. 開始畫圖
    plt.figure(figsize=(14, 10))
    cmap = plt.get_cmap('tab10')
    markers = {'MNIST': 'o', 'EMNIST': '^'}
    
    for ds_name in datasets_to_plot:
        if ds_name not in markers: continue
            
        for c in range(10): 
            idx = (all_datasets == ds_name) & (all_labels == c)
            if np.any(idx):
                plt.scatter(
                    feats_2d[idx, 0], feats_2d[idx, 1], 
                    color=cmap(c), 
                    marker=markers[ds_name], 
                    label=f'{ds_name} - {c}', 
                    alpha=0.7, 
                    edgecolors='w',
                    s=80
                )
                
    plt.title('t-SNE of Generator Output Features (MNIST vs EMNIST)', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, title="Dataset - Class")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(save_path)
    log_msg(f"[Utils] t-SNE plot saved to {save_path}")
    plt.close() 