import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import datetime
import numpy as np

def plot_metric_comparison(csv_files, metric, output_filename="comparison_plot.png", save_dir="."):
    loaded_data = {}
    all_rounds = set()
    x_col_global = 'entropy_ratio' 

    for file_path, custom_label in csv_files.items():
        if not os.path.exists(file_path):
            print(f"Can't find file: {file_path}")
            continue
            
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()

        if metric not in df.columns:
            continue

        if 'entropy_ratio' in df.columns:
            x_col = 'entropy_ratio'
        elif 'threshold' in df.columns:
            x_col = 'threshold'
        else:
            x_col = df.columns[0]
            
        x_col_global = x_col

        if 'global_round' in df.columns:
            all_rounds.update(df['global_round'].unique())
            
        loaded_data[custom_label] = df

    if not loaded_data:
        return

    all_rounds = sorted(list(all_rounds))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    metric_info = {
        "recall": "Recall (TPR)", 
        "specificity": "Specificity (TNR)",
        "precision": "Precision", 
        "f1_score": "F1-Score",
        "mcc": "MCC",
        "average_accuracy": "Balanced Accuracy ((TPR+TNR)/2)"
    }
    y_label = metric_info.get(metric, metric)

    for r in all_rounds:
        plt.figure(figsize=(10, 6)) 
        line_idx = 0
        has_data_for_round = False

        for custom_label, df in loaded_data.items():
            if 'global_round' in df.columns:
                df_sub = df[df['global_round'] == r]
            else:
                df_sub = df

            if df_sub.empty:
                continue

            df_sub = df_sub.sort_values(by=x_col_global)
            
            x_data = df_sub[x_col_global]
            y_data = df_sub[metric]

            plt.plot(
                x_data, y_data, 
                marker=markers[line_idx % len(markers)], 
                color=colors[line_idx % len(colors)], 
                linewidth=2.5, markersize=8, label=custom_label
            )
            line_idx += 1
            has_data_for_round = True

        if not has_data_for_round:
            plt.close()
            continue

        plt.title(f'Label Mapping Performance (Round {int(r)})', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Entropy Threshold Ratio', fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        
        if metric == "mcc":
            plt.ylim(-1.05, 1.05)
        else:
            plt.ylim(-0.05, 1.05)
            
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.legend(loc='lower right', fontsize=12)
        plt.tight_layout()

        final_save_path = os.path.join(save_dir, f"{metric}_Round_{int(r)}_{output_filename}")
        plt.savefig(final_save_path, dpi=300)
        print(f"Saved {metric} plot for Round {int(r)} to {final_save_path}")
        
        plt.close()

if __name__ == "__main__":
    print('1')
    real_dir = "logs/2026-05-12_17-40-17/gan_ddpm_newEnt"
    di_synthetic_dir = "logs/2026-05-12_17-40-17/new_client_2026-05-12_18-47-52"
    fed_synthetic_dir = "logs/2026-05-12_17-40-17/new_client_2026-05-13_14-32-41"
    gan_dir = "logs/2026-06-19_21-08-41/GeFL_GAN_baseline"

    oldEnt_dir = "logs/2026-06-24_10-35-49/GeFL_GAN_baseline/mapping_results"
    newEnt_dir = "logs/2026-06-19_21-08-41/GeFL_GAN_baseline/mapping_results"

    files_to_compare = {
        # os.path.join(real_dir, "GeFL_GAN_DDPM_Real_mapping_acc_per_round.csv"): "Raw Data",
        # os.path.join(di_synthetic_dir, "DI_syn_oldGen_newEnt.csv"): "DeepInversion",
        # os.path.join(fed_synthetic_dir, "Fed_syn_oldGen_newEnt.csv"): "FedFTG",
        # os.path.join(gan_dir, "GeFL_GAN_DDPM_GAN_mapping_acc_per_round.csv"): "GAN",

        os.path.join(oldEnt_dir, "GeFL_GAN_baseline_Ours_mapping_acc.csv"): "oldEnt",
        os.path.join(newEnt_dir, "GeFL_GAN_baseline_Ours_mapping_acc.csv"): "newEnt",
    }

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join("./plot/plot_label_mapping", f"gan_oldEnt_newEnt")
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n Output directory: {save_dir}")

    log_file_path = os.path.join(save_dir, "source_csv_paths.log")
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("=== Plotting Source Information ===\n")
        f.write(f"Generated at: {current_time}\n\n")
        f.write("Input CSV Files:\n")
        for fp, label in files_to_compare.items():
            status = "[Found]" if os.path.exists(fp) else "[Missing]"
            f.write(f"{status} [{label}] {fp}\n")
    print(f"log file: {log_file_path}\n")

    metrics_to_plot = ["recall", "specificity", "precision", "average_accuracy", "f1_score", "mcc"]
    
    for m in metrics_to_plot:
        plot_metric_comparison(files_to_compare, metric=m, output_filename="plot.png", save_dir=save_dir)