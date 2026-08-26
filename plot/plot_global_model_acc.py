import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def plot_global_accuracy_by_epoch():
    methods_dirs = {
        # "class_name": "logs/2026-07-03_21-26-24/GeFL_DeepInversion_gen",
        # "Ours": "logs/2026-07-03_17-52-24/GeFL_DeepInversion_gen",
        # "identical": "logs/2026-07-03_21-27-09/GeFL_DeepInversion_gen",

        "class_name": "logs/2026-07-04_00-13-48/GeFL_DeepInversion_gen",
        "Ours": "logs/2026-07-04_00-12-12/GeFL_DeepInversion_gen",
        "identical": "logs/2026-07-04_00-14-42/GeFL_DeepInversion_gen",
        #"feature-based": "logs/2026-07-04_00-15-19/GeFL_DeepInversion_gen",
       #"single-direct": "logs/2026-07-04_12-29-20/GeFL_DeepInversion_gen",

    }

    datasets = ["MNIST", "EMNIST", "CIFAR10", "mix"]
    output_dir = "./plot/global_temp/gan_iid"
    os.makedirs(output_dir, exist_ok=True)

    for dataset in datasets:
        plt.figure(figsize=(10, 6))
        #csv_filename = f"global_model_acc_{dataset}.csv"
        csv_filename = f"mix_new.csv"

        has_data = False
        round_id = None

        for method_name, method_path in methods_dirs.items():
            filepath = os.path.join(method_path, csv_filename)

            if not os.path.exists(filepath):
                print(f"File not found: {filepath}")
                continue

            df = pd.read_csv(filepath)

            if 'Epoch' not in df.columns:
                print(f"Skip {method_name}: no Epoch column in {filepath}")
                continue

            round_col = 'Round' if 'Round' in df.columns else 'global_round'
            acc_col = 'Accuracy' if 'Accuracy' in df.columns else 'accuracy'

            df = df.sort_values('Epoch')
            round_id = df[round_col].iloc[0]

            plt.plot(
                df['Epoch'],
                df[acc_col],
                marker='o',
                label=method_name,
                linewidth=2
            )

            has_data = True

        if has_data:
            plt.title(
                f"Global Model Accuracy on {dataset}",
                fontsize=14
            )
            plt.xlabel("Training Epoch", fontsize=12)
            plt.ylabel("Accuracy (%)", fontsize=12)
            plt.ylim(0, 20)
            plt.xlim(0, 50)
            plt.gca().xaxis.set_major_locator(MultipleLocator(1))
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='lower right', fontsize=10)

            plt.gca().xaxis.set_major_locator(MultipleLocator(10))

            save_path = os.path.join(
                output_dir,
                f"Global_Acc_{dataset}_Round_{round_id}_Epoch_mix.pdf"
            )
            plt.savefig(save_path, bbox_inches='tight')

        plt.close()


def plot_global_accuracy():
    round = 15
    methods_dirs = {
        # mnist + emnist + fashionmnist
        # "Ours_GeFL": "logs/2026-05-11_18-56-10/Ours_GeFL",
        # "UDON": "logs/2026-05-12_00-51-46/UDON", 
        # "FedTED": "logs/2026-05-12_02-19-31/FedTED",
        # "FedTED_dir": "logs/2026-05-12_12-45-34/FedTED_dir",

        # mnist + emnist + cifar-10
        # "GAN": "logs_temp/2026-05-30_03-23-43/Ours_GeFL",
        # #"UDON": "logs/2026-05-29_14-11-19/UDON", 
        # "FedTED": "logs_temp/2026-05-29_18-17-24/FedTED",
        # "FedFTG": "logs_temp/2026-06-02_11-45-44/Ours",
        # # "DDPM": "logs/2026-05-30_19-36-14/GeFL_DDPM",
        # "DeepInversion": "logs_temp/2026-06-26_16-32-46/GeFL_DDPM_baseline_total_DI",

        # 'Class Name':'logs_temp/2026-06-15_23-17-39/BaseFL_public',
        # 'Independent':'logs_temp/2026-06-16_00-10-09/BaseFL_public',
        # 'Identical':'logs/2026-06-16_00-35-08/BaseFL_public',

        # "bi-direct": "logs/2026-06-19_12-23-21/GeFL_DDPM_baseline_total_gan",
        # "single-direct": "logs/2026-06-19_17-55-44/GeFL_DDPM_baseline_total_gan", 
        # "feature-bi": "logs/2026-06-19_12-23-24/GeFL_DDPM_baseline_total_gan",
        # "image-cs": "logs/2026-06-19_23-40-42/GeFL_DDPM_baseline_total_gan",

        # "Ours": "logs_temp/2026-06-23_15-41-16/GeFL_DDPM_baseline_total_gan",
        # 'Missing_link':'logs/start25_gan_slamdunk/GeFL_slamdunk', 

        # "feature": "logs_temp/2026-06-23_15-43-33/GeFL_DDPM_baseline_total_gan",
        # "cosine-similarity": "logs_temp/2026-06-23_15-42-48/GeFL_DDPM_baseline_total_gan",

        # "class_name (MCC=1.0)": "logs/2026-07-04_07-47-20/GeFL_DDPM_baseline_total_gan",

        # "Public (MCC=1.0)": "logs_temp/2026-06-12_12-25-32/FedTED",

        # "single-direct (MCC=0.6350)": "logs/2026-06-23_15-42-03/GeFL_DDPM_baseline_total_gan", 
        # "FedTED_DDPM": "logs/2026-06-14_12-29-20/FedTED_DDPM_2",

        # 'Independent':'logs/2026-06-25_20-20-46/GeFL_DDPM_baseline_total',
        # 'Class_Name':'logs/2026-06-25_20-21-19/GeFL_DDPM_baseline_total',

        # 'ours':'logs/2026-07-10_13-36-34/GeFL_DDPM_baseline_total_gan',
        # 'feature':'logs/2026-07-10_13-36-38/GeFL_DDPM_baseline_total_gan',
        # 'SlamDunk':'logs/2026-07-11_17-18-48/GeFL_slamdunk', 

        # global round 20
        # 'ours':'logs/start20_gan_our/GeFL_DDPM_baseline_total_gan',
        # 'Missing_link':'logs/start25_gan_slamdunk/GeFL_slamdunk', 
        # 'feature':'logs/start20_gan_feature/GeFL_DDPM_baseline_total_gan',
        # "cosine-similarity": "logs/start20_gan_cs/GeFL_DDPM_baseline_total_gan",

        'Ours':f'logs/start{round}_noniid_gan_ours/GeFL_gan_pacfl_iid',
        'Missing Link':f'logs/start{round}_noniid_gan_missinglink/GeFL_gan_pacfl_iid', 
        'feature':f'logs/start{round}_noniid_gan_feature/GeFL_gan_pacfl_iid',
        "cosine-similarity": f'logs/start{round}_noniid_gan_cs/GeFL_gan_pacfl_iid',
    }
    
    #datasets = ["MNIST", "EMNIST", "FashionMNIST", "mix"]
    datasets = ["MNIST", "EMNIST", "CIFAR10", "mix"]
    
    output_dir = f"./plot/global_accuracy_plots/global_{round}_noniid"
    os.makedirs(output_dir, exist_ok=True)
    
    for dataset in datasets:
        plt.figure(figsize=(10, 6))
        if dataset == "mix":
            csv_filename = f"mix_new.csv"
        else:
            csv_filename = f"global_model_acc_{dataset}.csv"
        #csv_filename = f"mix_new.csv"
        
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
            plt.xlim(round, round+25)
            plt.gca().xaxis.set_major_locator(MultipleLocator(1))
            plt.grid(True, linestyle='--', alpha=0.7)
            #plt.legend(loc='lower right', fontsize=10)
            plt.legend(loc='upper right', fontsize=10)

            save_path = os.path.join(output_dir, f"Global_Acc_{dataset}.pdf")
            #save_path = os.path.join(output_dir, f"Global_Acc_{dataset}_mix.pdf")
            plt.savefig(save_path, bbox_inches='tight')
            
        plt.close()

if __name__ == "__main__":
    plot_global_accuracy()
    # plot_global_accuracy_by_epoch()