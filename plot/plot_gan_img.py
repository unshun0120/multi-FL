import os
import sys
import torch
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.nets import DCGANGenerator

def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    base_weight_dir = os.path.join(parent_dir, "logs/2026-05-14_11-13-16/GeFL_GAN_DDPM/global_gans")
    
    output_dir = "./plot/Gen_img/GAN"
    os.makedirs(output_dir, exist_ok=True)

    samples_per_class = 10
    
    max_classes_to_draw = 10
    noise_dim = 128

    dataset_specs = {
        'MNIST':   {'channels': 3, 'size': 32, 'classes': 10},
        'EMNIST':  {'channels': 3, 'size': 32, 'classes': 62},
        'CIFAR10': {'channels': 3, 'size': 32, 'classes': 10}
    }

    for d_name, specs in dataset_specs.items():
        weight_path = os.path.join(base_weight_dir, f"{d_name}_GAN.pth")
        
        if not os.path.exists(weight_path):
            print(f"No File:  {d_name} : {weight_path}")
            continue
            
        gen = DCGANGenerator(
            num_classes=specs['classes'],
            noise_dim=noise_dim,
            img_size=specs['size'],
            channels=specs['channels']
        ).to(device)

        checkpoint = torch.load(weight_path, map_location=device)
        gen.load_state_dict(checkpoint['generator'])
        gen.eval()

        all_generated_images = []
        
        num_classes = min(specs['classes'], max_classes_to_draw)

        with torch.no_grad():
            for class_idx in tqdm(range(num_classes), colour="cyan"):
                z = torch.randn(samples_per_class, noise_dim).to(device)
                y_local = torch.full((samples_per_class,), class_idx, dtype=torch.long).to(device)
                
                x_gen = gen(z, y_local)
                all_generated_images.append(x_gen.cpu())

        all_generated_images = torch.cat(all_generated_images, dim=0)

        grid = make_grid(
            all_generated_images, 
            nrow=samples_per_class, 
            padding=2, 
            pad_value=1.0, 
            normalize=True, 
            value_range=(-1, 1)
        )

        save_path = os.path.join(output_dir, f"{d_name}_generated_samples.png")
        save_image(grid, save_path)

if __name__ == "__main__":
    main()