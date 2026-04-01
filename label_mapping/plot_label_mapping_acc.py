import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import datetime

def plot_metric_comparison(csv_files, metric, output_filename="comparison_plot.png"):
    plt.figure(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    metric_info = {
        "recall": "Recall (TPR)",
        "specificity": "Specificity (TNR)",
        "average_accuracy": "Balanced Accuracy ((TPR+TNR)/2)"
    }
    y_label = metric_info.get(metric, metric) + " (%)"

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
        
        y_data = df[metric] * 100
        
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
    
    plt.ylim(0, 105)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(100))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()

    final_save_path = os.path.join(save_dir, f"{metric}_{output_filename}")
    plt.savefig(final_save_path, dpi=300)
    print(f"\n Save {metric} plot to {final_save_path}")
    
    plt.close()

if __name__ == "__main__":
    real_dir = "logs/2026-03-19_18-17-49/new_client_2026-03-24_19-43-22"
    synthetic_dir = "logs/2026-03-19_18-17-49/new_client_2026-03-24_19-43-22"
    
    files_to_compare = [
        os.path.join(real_dir, "real_old_entropy.csv"),
        #os.path.join(real_dir, "real_new_entropy.csv"),
        #os.path.join(synthetic_dir, "syn_oldGen_oldEnt.csv"),
        #os.path.join(synthetic_dir, "syn_oldGen_newEnt.csv"),
        #os.path.join(synthetic_dir, "syn_newGen_oldEnt.csv"),
        os.path.join(synthetic_dir, "syn_newGen_newEnt.csv")
    ]

    # folder
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join("label_mapping/res_img/non-iid", f"label_mapping_{current_time}")
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

    metrics_to_plot = ["recall", "specificity", "average_accuracy"]
    
    for m in metrics_to_plot:
        plot_metric_comparison(files_to_compare, metric=m, output_filename="plot.png")