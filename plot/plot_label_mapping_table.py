import os
import pandas as pd
import matplotlib.pyplot as plt


def make_best_table():
    # results_dir = "logs_temp/2026-06-19_21-08-41/GeFL_GAN_baseline/mapping_results"
    results_dir = ""
    # output_dir = "./label_mapping/offline_pacfl_noniid_3cluster(dirirchlet)/improve_single/table"
    # os.makedirs(output_dir, exist_ok=True)

    # methods = {
    #     "Ours": "GeFL_GAN_baseline_Ours_mapping_acc.csv",
    #     "Single-direct": "GeFL_GAN_baseline_Single_Direction_mapping_acc.csv",
    #     "Cosine-Similarity": "GeFL_GAN_baseline_Cosine_Similarity_mapping_acc.csv",
    #     "Feature-Bi": "GeFL_GAN_baseline_Feature_Bi_Direction_mapping_acc.csv",
    # }

    # methods = {
    #     "Ours": "./logs_temp/2026-06-19_21-08-41/GeFL_GAN_baseline/mapping_results/GeFL_GAN_baseline_Ours_mapping_acc.csv",
    #     "Single-direct": "./logs_temp/2026-06-19_21-08-41/GeFL_GAN_baseline/mapping_results/GeFL_GAN_baseline_Single_Direction_mapping_acc.csv",
    #     "Cosine-Similarity": "./logs_temp/2026-06-19_21-08-41/GeFL_GAN_baseline/mapping_results/GeFL_GAN_baseline_Cosine_Similarity_mapping_acc.csv",
    #     "Feature-Bi": "./logs_temp/2026-06-19_21-08-41/GeFL_GAN_baseline/mapping_results/GeFL_GAN_baseline_Feature_Bi_Direction_mapping_acc.csv",
    #     "Missing Link": "./offline_missing_link_results/offline_missing_link_mapping_acc.csv",
    #     "Improve-single": "./label_mapping/offline_ours_results/improve_single/offline_improve_mapping_acc.csv",
    #     "Improve-single\n(no Result Voting)": "./label_mapping/offline_ours_results/improve_single(noResultVoting)/offline_improve_mapping_acc.csv",
    # }

    # 1, 42, 758, 1248, 15698
    seed = 15698
    distribution = "noniid"
    output_dir = f"./label_mapping/pacfl_3cluster/{distribution}/improve_single_noniid_seed{seed}/table"
    #output_dir = f"./label_mapping/pacfl_3cluster/{distribution}/ablation/seed{seed}_table"
    os.makedirs(output_dir, exist_ok=True)
    methods = {
        # "bi-direct": "./label_mapping/offline_pacfl_noniid_results(label)/image-bi/label_mapping/offline_image-bi_noniid_mapping_acc.csv",
        # "Improve-single": f"./label_mapping/pacfl_3cluster/{distribution}/improve_single_seed{seed}/label_mapping/offline_improve_single_noniid_mapping_acc.csv",
        "Feature": f"./label_mapping/pacfl_3cluster/{distribution}/feature_seed{seed}/label_mapping/offline_feature_noniid_mapping_acc.csv",
        "Cosine-Similarity": f"./label_mapping/pacfl_3cluster/{distribution}/image-cs_seed{seed}/label_mapping/offline_image-cs_noniid_mapping_acc.csv",
        "Missing Link": f"./label_mapping/pacfl_3cluster/{distribution}/missing_link_seed{seed}/label_mapping/offline_missing_link_noniid_mapping_acc.csv",
        "Improve_noniid": f"./label_mapping/pacfl_3cluster/{distribution}/improve_single_noniid_seed{seed}/label_mapping/offline_improve_single_noniid_noniid_mapping_acc.csv",

        # "Improve_noniid\n(noSame)": f"./label_mapping/pacfl_3cluster/{distribution}/improve_single_noniid_seed{seed}_noSame/label_mapping/offline_improve_single_noniid_noniid_mapping_acc.csv",
        # "Improve_noniid\n(noCheck)": f"./label_mapping/pacfl_3cluster/{distribution}/improve_single_noniid_seed{seed}_noCheck/label_mapping/offline_improve_single_noniid_noniid_mapping_acc.csv",
        # "Improve_noniid\n(noSame)\n(noCheck)": f"./label_mapping/pacfl_3cluster/{distribution}/improve_single_noniid_seed{seed}_noSame_noCheck/label_mapping/offline_improve_single_noniid_noniid_mapping_acc.csv",
    }
    no_entropy_methods={}
    slam_dunk_values={}

    # no_entropy_methods = {
    #     "Ours": "Ours\n(no Entropy)",
    #     "Improve-single": "Improve-single\n(no Entropy)",
    # }

    x_col_map = {
        "Ours": "entropy_ratio",
        "Single-direct": "entropy_ratio",
        "Feature": "entropy_ratio",
        "Cosine-Similarity": "cs_threshold",
        "Missing Link": "missing_threshold",
        "Improve-single": "entropy_ratio",
        "Improve_noniid": "entropy_ratio",
        "Improve_noniid\n(noSame)": "entropy_ratio",
        "Improve_noniid\n(noCheck)": "entropy_ratio",
        "Improve_noniid\n(noSame)\n(noCheck)": "entropy_ratio",
        "Improve-single\n(no Result Voting)": "entropy_ratio",

        "bi-direct": "entropy_ratio",
        "Improve-single\n + \nMissing Link": "missing_threshold",
        "Improve-single\n(Single-direct)": "entropy_ratio",
        #"Missing Link": "entropy_ratio",
    }

    # slam_dunk_values = {
    #     5: {
    #         "recall": 0.7000000000,
    #         "specificity": 0.9939849624,
    #         "precision": 0.4666666667,
    #         "average_accuracy": 0.8469924812,
    #         "f1_score": 0.5600000000,
    #         "mcc": 0.5677044675,
    #     },
    #     10: {
    #         "recall": 0.9000000000,
    #         "specificity": 0.9962406015,
    #         "precision": 0.6428571429,
    #         "average_accuracy": 0.9481203008,
    #         "f1_score": 0.7500000000,
    #         "mcc": 0.7586031733,
    #     },
    #     15: {
    #         "recall": 0.8000000000,
    #         "specificity": 0.9947368421,
    #         "precision": 0.5333333333,
    #         "average_accuracy": 0.8973684211,
    #         "f1_score": 0.6400000000,
    #         "mcc": 0.6501231009,
    #     },
    #     20: {
    #         "recall": 0.6000000000,
    #         "specificity": 0.9954887218,
    #         "precision": 0.5000000000,
    #         "average_accuracy": 0.7977443609,
    #         "f1_score": 0.5454545455,
    #         "mcc": 0.5440135294,
    #     },
    #     25: {
    #         "recall": 0.8000000000,
    #         "specificity": 0.9939849624,
    #         "precision": 0.5000000000,
    #         "average_accuracy": 0.8969924812,
    #         "f1_score": 0.6153846154,
    #         "mcc": 0.6291209011,
    #     },
    # }

    # method_order = ["SlamDunk", "Cosine-Similarity", "Ours", "Ours\n(no Entropy)", "Single-direct", "Feature-Bi", "Missing Link", "Improve-single",  "Improve-single\n(no Entropy)", "Improve-single\n(no Result Voting)"]
    #method_order = ["bi-direct", "Missing Link", "Improve-single", "Improve_noniid", "Improve-single\n + \nMissing Link", "Improve-single\n(Single-direct)"]
    method_order = ["Missing Link", "Feature", "Cosine-Similarity", "Improve-single", "Improve_noniid", "Improve_noniid\n(noSame)", "Improve_noniid\n(noCheck)", "Improve_noniid\n(noSame)\n(noCheck)"]
    data = {}
    all_rounds = set()

    for method, filename in methods.items():
        path = os.path.join(results_dir, filename)
        if not os.path.exists(path):
            print(f"Not exist: {path}")
            continue

        df = pd.read_csv(path)
        data[method] = df
        all_rounds.update(df["global_round"].unique())

    all_rounds.update(slam_dunk_values.keys())
    all_rounds = sorted(all_rounds)

    all_rows = []

    for rnd in all_rounds:
        rows = []

        if rnd in slam_dunk_values:
            v = slam_dunk_values[rnd]
            rows.append({
                "Round": int(rnd),
                "Method": "SlamDunk",
                "Best Threshold": "-",
                "F1": v["f1_score"],
                "MCC": v["mcc"],
                "Recall": v["recall"],
                "Precision": v["precision"],
                "Specificity": v["specificity"],
            })


        for method, no_entropy_name in no_entropy_methods.items():
            if method not in data:
                continue

            df_no_entropy = data[method]

            if "entropy_ratio" not in df_no_entropy.columns:
                continue

            df_no_entropy = df_no_entropy[
                (df_no_entropy["global_round"] == rnd) &
                (df_no_entropy["entropy_ratio"].round(6) == 1.0)
            ]

            if df_no_entropy.empty:
                continue

            no_entropy_row = df_no_entropy.loc[df_no_entropy["mcc"].idxmax()]

            rows.append({
                "Round": int(rnd),
                "Method": no_entropy_name,
                "Best Threshold": "-",
                "F1": no_entropy_row["f1_score"],
                "MCC": no_entropy_row["mcc"],
                "Recall": no_entropy_row["recall"],
                "Precision": no_entropy_row["precision"],
                "Specificity": no_entropy_row["specificity"],
            })

        # for method in ["Cosine-Similarity", "Ours", "Single-direct", "Feature-Bi", "Missing Link", "Improve-single", "Improve-single\n(no Result Voting)"]:
        # for method in ["bi-direct", "Missing Link", "Improve-single", "Improve_noniid", "Improve-single\n + \nMissing Link", "Improve-single\n(Single-direct)"]:
        for method in ["Missing Link", "Feature", "Cosine-Similarity", "Improve-single", "Improve_noniid", "Improve_noniid\n(noSame)", "Improve_noniid\n(noCheck)", "Improve_noniid\n(noSame)\n(noCheck)"]:
            if method not in data:
                continue

            df = data[method]
            df_round = df[df["global_round"] == rnd].copy()

            if df_round.empty:
                continue

            x_col = x_col_map.get(method, "entropy_ratio")

            if x_col not in df_round.columns:
                print(f"{method} missing x column: {x_col}")
                continue

            exclude_no_entropy_methods = [
                "bi-direct",
                "Ours",
                "Single-direct",
                "Feature-Bi",
                "Missing Link",
                "Improve-single"
            ]

            if method in exclude_no_entropy_methods:
                df_round = df_round[
                    (df_round[x_col] >= 0.1) &
                    (df_round[x_col] <= 0.95)
                ]

            if df_round.empty:
                continue

            # best_row = df_round.loc[df_round["mcc"].idxmax()]
            best_row = df_round.loc[df_round["f1_score"].idxmax()]

            rows.append({
                "Round": int(rnd),
                "Method": method,
                "Best Threshold": best_row[x_col],
                "F1": best_row["f1_score"],
                "MCC": best_row["mcc"],
                "Recall": best_row["recall"],
                "Precision": best_row["precision"],
                "Specificity": best_row["specificity"],
            })

        if len(rows) == 0:
            continue

        round_df = pd.DataFrame(rows)

        round_df["Method"] = pd.Categorical(round_df["Method"], categories=method_order, ordered=True)
        round_df = round_df.sort_values("Method").reset_index(drop=True)

        all_rows.append(round_df.copy())

        display_df = round_df.copy()
        display_df["Best Threshold"] = display_df["Best Threshold"].apply(
            lambda x: "-" if x == "-" else f"{float(x):.2f}"
        )

        for col in ["F1", "MCC", "Recall", "Precision", "Specificity"]:
            display_df[col] = display_df[col].map(lambda x: f"{x:.4f}")

        display_df = display_df.rename(columns={
            "Best Threshold": "Best\nThreshold",
        })

        # fig_h = 1.4 + 0.55 * len(display_df)
        fig_h = 1.6 + 0.72 * len(display_df)
        fig, ax = plt.subplots(figsize=(11.5, fig_h))
        ax.axis("off")

        table = ax.table(
            cellText=display_df.values,
            colLabels=display_df.columns,
            cellLoc="center",
            colLoc="center",
            #loc="center",
            colWidths=[0.10, 0.15, 0.14, 0.11, 0.11, 0.11, 0.10, 0.10],
            bbox=[0.01, 0.01, 0.98, 0.95],
        )

        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.15, 1.5)

        ours_color = "#fff2cc"  # 淡黃色
        improve_color = "#d9eaf7"  # 淡藍色
        method_col_idx = list(display_df.columns).index("Method")

        for row_idx in range(len(display_df)):
            if str(display_df.loc[row_idx, "Method"]) == "Ours":
                table_row_idx = row_idx + 1   
                table[table_row_idx, method_col_idx].set_facecolor(ours_color)

            elif str(display_df.loc[row_idx, "Method"]) == "Improve_noniid":
                table_row_idx = row_idx + 1   
                table[table_row_idx, method_col_idx].set_facecolor(improve_color)

        for c in range(len(display_df.columns)):
            table[0, c].set_text_props(weight="bold", color="black")
            table[0, c].set_facecolor("white")

        mcc_color = "#f4cccc"  # 淡紅色
        # mcc_col_idx = list(display_df.columns).index("MCC")
        mcc_col_idx = list(display_df.columns).index("F1")

        for row_idx in range(len(display_df) + 1):
            table[row_idx, mcc_col_idx].set_facecolor(mcc_color)

        for c in range(len(display_df.columns)):
            table[0, c].set_height(table[0, c].get_height() * 1.6)

        metric_cols = ["F1", "MCC", "Recall", "Precision", "Specificity"]

        for row_idx in range(len(display_df)):
            for col_idx in range(len(display_df.columns)):
                table[row_idx + 1, col_idx].set_text_props(weight="normal")

        for metric in metric_cols:
            col_idx = list(display_df.columns).index(metric)
            max_value = round_df[metric].max()

            for row_idx in range(len(round_df)):
                if round_df.loc[row_idx, metric] == max_value:
                    table[row_idx + 1, col_idx].set_text_props(weight="bold")

        ax.set_title(
            f"Best Label Mapping Performance Table (Round {rnd})",
            fontsize=15,
            fontweight="bold",
            pad=2
        )

        pdf_path = os.path.join(output_dir, f"round_{rnd}_table.pdf")
        plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    make_best_table()