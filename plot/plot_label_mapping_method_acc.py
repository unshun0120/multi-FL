import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_mapping_results():
    results_dir = ""

    #output_base_dir = "./plot/label_mapping_baseline/improve_single"

    plot_slamdunk = False
    
    # methods = {
    #     # "Identical": "GeFL_DDPM_baseline_Identical_mapping_acc.csv",
    #     # "Independent": "GeFL_DDPM_baseline_Independent_mapping_acc.csv",
    #     "Image_biDirection (Ours)": "GeFL_GAN_baseline_Ours_mapping_acc.csv",
    #     "Cosine_Similarity": "GeFL_GAN_baseline_Cosine_Similarity_mapping_acc.csv",
    #     "Feature_biDirection": "GeFL_GAN_baseline_Feature_Bi_Direction_mapping_acc.csv",
    #     #"Image_singleDirection": "GeFL_GAN_baseline_Single_Direction_mapping_acc.csv",

    #     "Missing Link": "./offline_missing_link_results/offline_missing_link_mapping_acc.csv",
    # }

    # methods = {
    #     "Ours": "./logs_temp/2026-06-19_21-08-41/GeFL_GAN_baseline/mapping_results/GeFL_GAN_baseline_Ours_mapping_acc.csv",
    #     # "Cosine_Similarity": "./logs_temp/2026-06-19_21-08-41/GeFL_GAN_baseline/mapping_results/GeFL_GAN_baseline_Cosine_Similarity_mapping_acc.csv",
    #     # "Feature_biDirection": "./logs_temp/2026-06-19_21-08-41/GeFL_GAN_baseline/mapping_results/GeFL_GAN_baseline_Feature_Bi_Direction_mapping_acc.csv",
    #     "Missing Link": "./offline_missing_link_results/offline_missing_link_mapping_acc.csv",
    #     # "Non-IID (Ours)": "./logs/total100_gan_ours_noniid/GeFL_DDPM_baseline_total_gan_noniid/image-bi_noniid_mapping_acc.csv"
        
    #     "single-direct": "./label_mapping/offline_ours_results/single-direct/offline_improve_mapping_acc.csv",
    #     "Improve": "./label_mapping/offline_ours_results/2026_07_18_10_51_05/offline_improve_mapping_acc.csv",
    # }

    # noniid
    output_base_dir = "./label_mapping/offline_noniid_results(noniid_label)/improve_single_label_noniid(2)"

    methods = {
        "bi-direct": "./label_mapping/offline_noniid_results(noniid_label)/image-bi/label_mapping/offline_image-bi_noniid_mapping_acc.csv",
        "Missing Link": "./label_mapping/offline_noniid_results(noniid_label)/missing_link/label_mapping/offline_missing_link_noniid_mapping_acc.csv",
        #"Improve": os.path.join(output_base_dir, "label_mapping/offline_improve_single_noniid_mapping_acc.csv"),
        "Improve": "./label_mapping/offline_noniid_results(noniid_label)/improve_single/label_mapping/offline_improve_single_noniid_mapping_acc.csv",
        #"single-direct": "./label_mapping/offline_noniid_results/2026_07_20_10_40_34/label_mapping/offline_image-single_noniid_mapping_acc.csv",
        "Improve_noniid": os.path.join(output_base_dir, "label_mapping/offline_improve_single_label_noniid_noniid_mapping_acc.csv"),
    }

    color_map = {
        "Ours": "tab:blue",   
        "Feature_biDirection": "tab:orange",       
        "Cosine_Similarity": "tab:green",
        "SlamDunk": "tab:red",  
        "Missing Link": "gold",        

        "bi-direct (ours)": "tab:blue",   
        "Improve": "tab:orange",  
        "single-direct": "tab:green", 
        "Improve_noniid": "tab:purple",
    }

    x_col_map = {
        "Ours": "entropy_ratio",
        "Feature_biDirection": "entropy_ratio",
        "Image_singleDirection": "entropy_ratio",
        "Cosine_Similarity": "entropy_ratio",
        #"Missing Link": "missing_threshold",
        "Missing Link": "entropy_ratio",
        "Improve_noniid": "missing_threshold",
    }

    metrics = {
        "Recall": "recall",
        "Specificity": "specificity",
        "Precision": "precision",
        "Average_Accuracy": "average_accuracy",
        "F1_Score": "f1_score",
        "MCC": "mcc"
    }

    slam_dunk_values = {
        5: {
            "Recall": 0.7000000000,
            "Specificity": 0.9939849624,
            "Precision": 0.4666666667,
            "Average_Accuracy": 0.8469924812,
            "F1_Score": 0.5600000000,
            "MCC": 0.5677044675,
        },
        10: {
            "Recall": 0.9000000000,
            "Specificity": 0.9962406015,
            "Precision": 0.6428571429,
            "Average_Accuracy": 0.9481203008,
            "F1_Score": 0.7500000000,
            "MCC": 0.7586031733,
        },
        15: {
            "Recall": 0.8000000000,
            "Specificity": 0.9947368421,
            "Precision": 0.5333333333,
            "Average_Accuracy": 0.8973684211,
            "F1_Score": 0.6400000000,
            "MCC": 0.6501231009,
        },
        20: {
            "Recall": 0.6000000000,
            "Specificity": 0.9954887218,
            "Precision": 0.5000000000,
            "Average_Accuracy": 0.7977443609,
            "F1_Score": 0.5454545455,
            "MCC": 0.5440135294,
        },
        25: {
            "Recall": 0.8000000000,
            "Specificity": 0.9939849624,
            "Precision": 0.5000000000,
            "Average_Accuracy": 0.8969924812,
            "F1_Score": 0.6153846154,
            "MCC": 0.6291209011,
        },
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
                x_col = x_col_map.get(method, "entropy_ratio")

                if x_col not in df.columns:
                    print(f"{method} missing x column: {x_col}")
                    continue

                df_round = df[df['global_round'] == rnd].sort_values(by=x_col)
                
                if df_round.empty:
                    continue

                line_color = color_map.get(method, 'black')

                if "Cosine" in method:
                    if ax2 is None:
                        ax2 = ax1.twiny() 
                    
                    ax2.plot(
                        df_round[x_col],
                        df_round[metric_col], 
                        marker='^',      
                        linestyle='--',  
                        label=f"{method}",
                        linewidth=2.5,
                        color=line_color   
                    )
                else:
                    ax1.plot(
                        df_round[x_col], 
                        df_round[metric_col], 
                        marker='o', 
                        label=f"{method}",
                        linewidth=2,
                        color=line_color
                    )

            if plot_slamdunk and rnd in slam_dunk_values and metric_name in slam_dunk_values[rnd]:
                ax1.axhline(
                    y=slam_dunk_values[rnd][metric_name],
                    color=color_map["SlamDunk"],
                    linestyle="-",
                    linewidth=2.5,
                    label="SlamDunk"
                )

            ax1.set_title(f"{metric_name} (Global Round: {rnd})", fontsize=14)

            ax1.set_xlabel("Entropy Ratio / Missing Link Threshold (0.0 ~ 1.0)", fontsize=12)
            ax1.set_ylabel(metric_name, fontsize=12)

            ax1.set_xlim(0.05, 1.05)
            ax1.set_xticks([x / 10.0 for x in range(1, 11)])

            ax1.grid(True, linestyle='--', alpha=0.7)

            if ax2 is not None:
                ax2.set_xlabel(
                    "Cosine Similarity Threshold (-1.0 ~ 1.0)",
                    fontsize=12,
                    color='purple',
                    fontweight='bold'
                )
                ax2.set_xlim(-1.05, 1.05)
                ax2.set_xticks([x / 10.0 for x in range(-10, 11, 2)])
                ax2.tick_params(axis='x', colors='tab:green')
                ax2.grid(False)

            if metric_name == "MCC":
                ax1.set_ylim(-1.05, 1.05)
            else:
                ax1.set_ylim(-0.05, 1.05)

            lines_1, labels_1 = ax1.get_legend_handles_labels()
            if ax2 is not None:
                lines_2, labels_2 = ax2.get_legend_handles_labels()
                ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=11, loc='best')
            else:
                ax1.legend(fontsize=11, loc='best')

            save_path = os.path.join(metric_dir, f"{metric_name}_Round_{rnd}.pdf")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            

if __name__ == "__main__":
    plot_mapping_results()