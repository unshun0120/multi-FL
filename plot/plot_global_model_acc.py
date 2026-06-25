import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def plot_global_accuracy():
    methods_dirs = {
        # mnist + emnist + fashionmnist
        # "Ours_GeFL": "logs/2026-05-11_18-56-10/Ours_GeFL",
        # "UDON": "logs/2026-05-12_00-51-46/UDON", 
        # "FedTED": "logs/2026-05-12_02-19-31/FedTED",
        # "FedTED_dir": "logs/2026-05-12_12-45-34/FedTED_dir",

        # mnist + emnist + cifar-10
        # "GAN": "logs/2026-05-30_03-23-43/Ours_GeFL",
        # "UDON": "logs/2026-05-29_14-11-19/UDON", 
        # "FedTED": "logs/2026-05-29_18-17-24/FedTED",
        # "FedFTG": "logs/2026-06-02_11-45-44/Ours",
        # "DDPM": "logs/2026-05-30_19-36-14/GeFL_DDPM",

        'Independent':'logs/2026-06-24_17-15-29/GeFL_DDPM_baseline_total',
        # 'Identical':'logs/2026-06-16_00-35-08/BaseFL_public',
        #'Class Name':'logs/2026-06-17_13-34-19/GeFL_DDPM_baseline_total_public',

        # "bi-direct": "logs/2026-06-19_12-23-21/GeFL_DDPM_baseline_total_gan",
        # "single-direct": "logs/2026-06-19_17-55-44/GeFL_DDPM_baseline_total_gan", 
        # "feature-bi": "logs/2026-06-19_12-23-24/GeFL_DDPM_baseline_total_gan",
        # "image-cs": "logs/2026-06-19_23-40-42/GeFL_DDPM_baseline_total_gan",

        # "bi-direct (MCC=0.9126)": "logs/2026-06-23_15-41-16/GeFL_DDPM_baseline_total_gan",
        # "single-direct (MCC=0.6350)": "logs/2026-06-23_15-42-03/GeFL_DDPM_baseline_total_gan", 
        # "feature-bi (MCC=0.0)": "logs/2026-06-23_15-43-33/GeFL_DDPM_baseline_total_gan",
        # "image-cs (MCC=0.3982)": "logs/2026-06-23_15-42-48/GeFL_DDPM_baseline_total_gan",

        # "FedTED (MCC=1.0)": "logs/2026-06-12_12-25-32/FedTED",
        #"FedTED_DDPM": "logs/2026-06-14_12-29-20/FedTED_DDPM_2",
    }
    
    #datasets = ["MNIST", "EMNIST", "FashionMNIST", "mix"]
    datasets = ["MNIST", "EMNIST", "CIFAR10", "mix"]
    
    output_dir = "./plot/global_accuracy_plots/ddpm_test"
    os.makedirs(output_dir, exist_ok=True)
    
    for dataset in datasets:
        plt.figure(figsize=(10, 6))
        csv_filename = f"global_model_acc_{dataset}.csv"
        
        has_data = False
        
        for method_name, method_path in methods_dirs.items():
            filepath = os.path.join(method_path, csv_filename)
            
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                
                x_col = 'Round' if 'Round' in df.columns else 'global_round'
                y_col = 'Accuracy' if 'Accuracy' in df.columns else 'accuracy'
                
                plt.plot(
                    df[x_col], 
                    df[y_col], 
                    marker='o', 
                    label=method_name,
                    linewidth=2
                )
                has_data = True
            else:
                print(f"File not found: {filepath}")
                
        if has_data:
            plt.title(f"Global Model Accuracy on {dataset}", fontsize=14)
            plt.xlabel("Global Round", fontsize=12)
            plt.ylabel("Accuracy (%)", fontsize=12)
            plt.ylim(0, 105)
            plt.xlim(24, 51)
            plt.gca().xaxis.set_major_locator(MultipleLocator(1))
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='lower right', fontsize=10)
            # plt.legend(loc='upper right', fontsize=10)
            
            save_path = os.path.join(output_dir, f"Global_Acc_{dataset}.pdf")
            plt.savefig(save_path, bbox_inches='tight')
            
        plt.close()

if __name__ == "__main__":
    plot_global_accuracy()