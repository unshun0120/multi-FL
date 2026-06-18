import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import datetime
import numpy as np

def plot_metric_comparison(csv_files, metric, output_filename="comparison_plot.png"):
    plt.figure(figsize=(10, 6))

    n_lines = len(csv_files)
    
    #colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    #markers = ['o', 's', '^', 'D']

    colors = plt.cm.tab20(np.linspace(0, 1, n_lines))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', '<', '>', 'h', 'd']

    metric_info = {
        "recall": "Recall (TPR)", 
        "specificity": "Specificity (TNR)",
        "precision": "Precision", 
        "f1_score": "F1-Score",
        "mcc": "MCC",
        "average_accuracy": "Balanced Accuracy ((TPR+TNR)/2)"
    }
    y_label = metric_info.get(metric, metric)

    for idx, file_path in enumerate(csv_files):
        if not os.path.exists(file_path):
            print(f"Can't find file: {file_path}")
            continue
            
        df = pd.read_csv(file_path)
        
        df.columns = df.columns.str.strip()
        
        x_col = df.columns[0]
        x_data = df[x_col]

        if metric not in df.columns:
            continue
        
        y_data = df[metric]
        
        label_name = os.path.basename(file_path).replace('.csv', '')
        
        plt.plot(
            x_data, 
            y_data, 
            marker=markers[idx % len(markers)], 
            color=colors[idx % len(colors)], 
            linewidth=2.5, 
            markersize=6,
            label=label_name
        )

    plt.title(f'Label Mapping Performance ({y_label})', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Entropy Threshold Ratio', fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    
    if metric == "mcc":
        plt.ylim(-1.05, 1.05)
    else:
        plt.ylim(-0.05, 1.05)
    # plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(100))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()

    final_save_path = os.path.join(save_dir, f"{metric}_{output_filename}")
    plt.savefig(final_save_path, dpi=300)
    print(f"\n Save {metric} plot to {final_save_path}")
    
    plt.close()

if __name__ == "__main__":
    print('1')
    real_dir = "logs/2026-05-12_17-40-17/gan_ddpm_oldEnt"
    di_synthetic_dir = "logs/2026-05-12_17-40-17/new_client_2026-05-12_18-47-52"
    fast_synthetic_dir = "logs/Local_iid_het_gb40/fast_oldGen"
    nayer_synthetic_dir = "logs/Local_iid_het_gb40/nayer_oldGen"
    fed_synthetic_dir = "logs/2026-05-12_17-40-17/new_client_2026-05-13_14-32-41"

    gan_dir = "logs/2026-05-12_17-40-17/gan_ddpm_oldEnt"
    old_ddpm_synthetic_dir = "logs/2026-05-12_17-40-17/gan_ddpm_oldEnt"
    new_ddpm_synthetic_dir = "logs/2026-05-12_17-40-17/gan_ddpm_oldEnt"

    files_to_compare = [
        os.path.join(real_dir, "GeFL_GAN_DDPM_Real_mapping_acc_per_round.csv"),
        # os.path.join(real_dir, "Real_real_new_entropy.csv"),

        # os.path.join(synthetic_dir, "syn_oldGen_oldEnt.csv"),
        # os.path.join(synthetic_dir, "syn_oldGen_newEnt.csv"),
        # os.path.join(synthetic_dir, "syn_newGen_oldEnt.csv"),
        # os.path.join(synthetic_dir, "syn_newGen_newEnt.csv")

        os.path.join(di_synthetic_dir, "DI_syn_oldGen_oldEnt.csv"),
        # os.path.join(fast_synthetic_dir, "FAST_syn_oldGen_oldEnt.csv"),
        # os.path.join(nayer_synthetic_dir, "NAYER_syn_oldGen_oldEnt.csv"),
        os.path.join(fed_synthetic_dir, "Fed_syn_oldGen_oldEnt.csv"),

        os.path.join(gan_dir, "GeFL_GAN_DDPM_GAN_mapping_acc_per_round.csv"),
        #os.path.join(gan_dir, "GeFL_syn_GeFL_newEnt.csv"),

        #os.path.join(old_ddpm_synthetic_dir, "GeFL_GAN_DDPM_DDPM_mapping_acc_per_round.csv"),
        os.path.join(new_ddpm_synthetic_dir, "new_GeFL_DDPM_DDPM_mapping_acc_per_round.csv"),
    ]

    # folder
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # save_dir = os.path.join("./plot/plot_label_mapping", f"label_mapping_{current_time}")
    save_dir = os.path.join("./plot/plot_label_mapping", f"all_oldEnt_2")
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n Output directory: {save_dir}")

    # log file
    log_file_path = os.path.join(save_dir, "source_csv_paths.log")
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("=== Plotting Source Information ===\n")
        f.write(f"Generated at: {current_time}\n\n")
        f.write("Input CSV Files:\n")
        for fp in files_to_compare:
            status = "[Found]" if os.path.exists(fp) else "[Missing]"
            f.write(f"{status} {fp}\n")
    print(f"log file: {log_file_path}\n")

    metrics_to_plot = ["recall", "specificity", "precision", "average_accuracy", "f1_score", "mcc"]
    
    for m in metrics_to_plot:
        plot_metric_comparison(files_to_compare, metric=m, output_filename="plot.png")