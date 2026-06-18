import os
import sys
import torch
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.nets import ContextUnet, DDPM

def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    base_weight_dir = os.path.join(parent_dir, "logs/2026-06-09_10-34-22/GeFL_DDPM_baseline_total/global_gans")
    
    output_dir = "./plot/Gen_img"
    os.makedirs(output_dir, exist_ok=True)

    samples_per_class = 2
    
    max_classes_to_draw = 62

    dataset_specs = {
        'MNIST':   {'channels': 3, 'size': 32, 'classes': 10},
        'EMNIST':  {'channels': 3, 'size': 32, 'classes': 62},
        'CIFAR10': {'channels': 3, 'size': 32, 'classes': 10}
    }

    for d_name, specs in dataset_specs.items():
        weight_path = os.path.join(base_weight_dir, f"{d_name}_DDPM.pth")
        
        if not os.path.exists(weight_path):
            print(f"No File:  {d_name} : {weight_path}")
            continue
            
        unet = ContextUnet(
            in_channels=specs['channels'], 
            n_feat=64, 
            n_classes=specs['classes']
        ).to(device)
        
        ddpm = DDPM(
            nn_model=unet, 
            betas=(1e-4, 0.02), 
            n_T=1000, 
            device=device, 
            drop_prob=0.1
        ).to(device)

        checkpoint = torch.load(weight_path, map_location=device)
        ddpm.load_state_dict(checkpoint['generator'])
        ddpm.eval()

        all_generated_images = []
        
        num_classes = min(specs['classes'], max_classes_to_draw)

        with torch.no_grad():
            for class_idx in tqdm(range(num_classes), colour="cyan"):
                x_gen, _ = ddpm.sample(
                    n_sample=samples_per_class, 
                    size=(specs['channels'], specs['size'], specs['size']), 
                    device=device, 
                    guide_w=1.5,    
                    label=class_idx  
                )
                
                x_gen = (x_gen + 1.0) / 2.0
                x_gen = torch.clamp(x_gen, 0.0, 1.0)
                
                all_generated_images.append(x_gen.cpu())

        all_generated_images = torch.cat(all_generated_images, dim=0)

        grid = make_grid(all_generated_images, nrow=samples_per_class, padding=2, pad_value=1.0) 

        save_path = os.path.join(output_dir, f"{d_name}_generated_samples.png")
        save_image(grid, save_path)

if __name__ == "__main__":
    main()