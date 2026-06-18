import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_mapping_results():
    results_dir = "logs/2026-05-23_12-08-23/GeFL_DDPM_baseline/mapping_results" 
    
    output_base_dir = "./plot/label_mapping_baseline/new_ent_cs"
    
    methods = {
        # "Identical": "GeFL_DDPM_baseline_Identical_mapping_acc.csv",
        # "Independent": "GeFL_DDPM_baseline_Independent_mapping_acc.csv",
        # "Feature_biDirection": "GeFL_DDPM_baseline_Feature_Bi_Direction_mapping_acc.csv",
        # "Image_singleDirection": "GeFL_DDPM_baseline_Single_Direction_mapping_acc.csv",
        "Image_biDirection (Ours)": "GeFL_DDPM_baseline_Ours_mapping_acc.csv",
        "Cosine_Similarity": "GeFL_DDPM_baseline_Cosine_Similarity_mapping_acc.csv" 
    }

    # results_dir = "" 
    
    # output_base_dir = "./plot/label_mapping_baseline/new_old_entropy"
    
    # methods = {
    #     "New_Image_biDirection": "logs/2026-05-23_12-08-23/GeFL_DDPM_baseline/mapping_results/GeFL_DDPM_baseline_Ours_mapping_acc.csv",
    #     "Old_Image_biDirection": "logs/2026-05-23_12-08-38/GeFL_DDPM_baseline/mapping_results/GeFL_DDPM_baseline_Ours_mapping_acc.csv",
    # }

    metrics = {
        "Recall": "recall",
        "Specificity": "specificity",
        "Precision": "precision",
        "Average_Accuracy": "average_accuracy",
        "F1_Score": "f1_score",
        "MCC": "mcc"
    }


    data = {}
    for method, filename in methods.items():
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            data[method] = pd.read_csv(filepath)
        else:
            print(f"Not exit : {filepath} ")

    if not data:
        print("No csv file")
        return

    rounds = sorted(list(set.intersection(*[set(df['global_round'].unique()) for df in data.values()])))

    for metric_name, metric_col in metrics.items():
        metric_dir = os.path.join(output_base_dir, metric_name)
        os.makedirs(metric_dir, exist_ok=True)

        for rnd in rounds:
            # plt.figure(figsize=(10, 6))

            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax2 = None

            for method in data.keys():
                df = data[method]
                df_round = df[df['global_round'] == rnd].sort_values(by='entropy_ratio')
                
                if df_round.empty:
                    continue

                if "Cosine" in method:
                    if ax2 is None:
                        ax2 = ax1.twiny() 
                    
                    ax2.plot(
                        df_round['entropy_ratio'],
                        df_round[metric_col], 
                        marker='^',      
                        linestyle='--',  
                        label=f"{method}",
                        linewidth=2.5,
                        color='purple'   
                    )
                else:
                    ax1.plot(
                        df_round['entropy_ratio'], 
                        df_round[metric_col], 
                        marker='o', 
                        label=f"{method}",
                        linewidth=2
                    )

            plt.title(f"{metric_name} (Global Round: {rnd})", fontsize=14)
            plt.xlabel("Entropy Ratio", fontsize=12)
            plt.ylabel(metric_name, fontsize=12)
            plt.xticks([x/100.0 for x in range(10, 105, 5)])
            plt.grid(True, linestyle='--', alpha=0.7)

            if ax2 is not None:
                ax2.set_xlabel("Cosine Similarity Threshold (-1.0 ~ 1.0)", fontsize=12, color='purple', fontweight='bold')
                ax2.set_xlim(-1.05, 1.05)
                ax2.set_xticks([x/10.0 for x in range(-10, 11, 2)])
                ax2.tick_params(axis='x', colors='purple')

            if metric_name == "MCC":
                plt.ylim(-1.05, 1.05)
            else:
                plt.ylim(-0.05, 1.05)

            lines_1, labels_1 = ax1.get_legend_handles_labels()
            if ax2 is not None:
                lines_2, labels_2 = ax2.get_legend_handles_labels()
                ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=11, loc='best')
            else:
                ax1.legend(fontsize=11, loc='best')

            save_path = os.path.join(metric_dir, f"{metric_name}_Round_{rnd}.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            

if __name__ == "__main__":
    plot_mapping_results()