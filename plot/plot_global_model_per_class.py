import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# df_independent = pd.read_csv('logs_temp/2026-06-16_00-10-09/BaseFL_public/global_model_class_acc.csv')
# df_independent['Mapping'] = 'Independent'

# # df_identical = pd.read_csv('logs/2026-06-16_00-35-08/BaseFL_public/global_model_class_acc.csv')
# # df_identical['Mapping'] = 'Identical'

# df_class_name = pd.read_csv('logs_temp/2026-06-15_23-17-39/BaseFL_public/global_model_class_acc.csv')
# df_class_name['Mapping'] = 'Class_Name'

# df_bi = pd.read_csv('logs/2026-06-19_12-23-21/GeFL_DDPM_baseline_total_gan/global_model_class_acc.csv')
# df_bi['Mapping'] = 'Bi-Direct (Ours)'

# df_single = pd.read_csv('logs/2026-06-19_17-55-44/GeFL_DDPM_baseline_total_gan/global_model_class_acc.csv')
# df_single['Mapping'] = 'Single-Direct'

# df_cs = pd.read_csv('logs/2026-06-19_23-40-42/GeFL_DDPM_baseline_total_gan/global_model_class_acc.csv')
# df_cs['Mapping'] = 'Image-cs'

# df_feat = pd.read_csv('logs/2026-06-19_12-23-24/GeFL_DDPM_baseline_total_gan/global_model_class_acc.csv')
# df_feat['Mapping'] = 'Feature-Bi'

# df_bi = pd.read_csv('logs/start15_gan_our/GeFL_DDPM_baseline_total_gan/global_model_class_acc.csv')
# df_bi['Mapping'] = 'Ours'

df_slamdunk = pd.read_csv('logs/start25_gan_slamdunk/GeFL_slamdunk/global_model_class_acc.csv')
df_slamdunk['Mapping'] = 'Missing link'

# df_cs = pd.read_csv('logs/start15_gan_cs/GeFL_DDPM_baseline_total_gan/global_model_class_acc.csv')
# df_cs['Mapping'] = 'cosine-similarity'

# df_feat = pd.read_csv('logs/start15_gan_feature/GeFL_DDPM_baseline_total_gan/global_model_class_acc.csv')
# df_feat['Mapping'] = 'feature'

df_bi = pd.read_csv("logs_temp/2026-06-23_15-41-16/GeFL_DDPM_baseline_total_gan/global_model_class_acc.csv")
df_bi['Mapping'] = 'Ours'

df_cs = pd.read_csv('logs_temp/2026-06-23_15-43-33/GeFL_DDPM_baseline_total_gan/global_model_class_acc.csv')
df_cs['Mapping'] = 'cosine-similarity'

df_feat = pd.read_csv('logs_temp/2026-06-23_15-42-48/GeFL_DDPM_baseline_total_gan/global_model_class_acc.csv')
df_feat['Mapping'] = 'feature'



# df_all = pd.concat([df_class_name, df_independent], ignore_index=True)
# df_all = pd.concat([df_independent, df_identical, df_class_name], ignore_index=True)
# df_all = pd.concat([df_bi, df_cs, df_feat], ignore_index=True)

df_all = pd.concat([df_bi, df_slamdunk, df_feat, df_cs], ignore_index=True)

target_datasets = ["MNIST", "EMNIST"]
target_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

df_filtered = df_all[
    (df_all["Dataset"].isin(target_datasets)) &
    (df_all["Local_Class"].isin(target_classes))
].copy()

df_filtered["Class_Pos"] = df_filtered["Local_Class"] % 5

def make_panel_row(row):
    if row["Dataset"] == "MNIST" and row["Local_Class"] <= 4:
        return "MNIST_0_4"
    elif row["Dataset"] == "MNIST" and row["Local_Class"] >= 5:
        return "MNIST_5_9"
    elif row["Dataset"] == "EMNIST" and row["Local_Class"] <= 4:
        return "EMNIST_0_4"
    else:
        return "EMNIST_5_9"

df_filtered["Panel_Row"] = df_filtered.apply(make_panel_row, axis=1)

row_order = ["MNIST_0_4", "MNIST_5_9", "EMNIST_0_4", "EMNIST_5_9"]

sns.set_theme(style="whitegrid")

g = sns.relplot(
    data=df_filtered,
    x="Round",
    y="Accuracy",
    hue="Mapping",
    row="Panel_Row",
    col="Class_Pos",
    row_order=row_order,
    col_order=[0, 1, 2, 3, 4],
    kind="line",
    height=2.3,
    aspect=1.25,
    linewidth=2.2,
    palette="tab10"
)

g.set_axis_labels("Global Round", "Accuracy (%)")

for r, row_name in enumerate(row_order):
    for c in range(5):
        ax = g.axes[r, c]

        if row_name == "MNIST_0_4":
            dataset_name = "MNIST"
            class_id = c
        elif row_name == "MNIST_5_9":
            dataset_name = "MNIST"
            class_id = c + 5
        elif row_name == "EMNIST_0_4":
            dataset_name = "EMNIST"
            class_id = c
        else:
            dataset_name = "EMNIST"
            class_id = c + 5

        ax.set_title(f"{dataset_name} | Class {class_id}", fontsize=9)
        ax.set_ylim(0, 105)
        ax.set_xlim(df_filtered["Round"].min(), df_filtered["Round"].max())

g.fig.subplots_adjust(left=0.08, right=0.88)

row0_pos = g.axes[0, 0].get_position()
row1_pos = g.axes[1, 0].get_position()
row2_pos = g.axes[2, 0].get_position()
row3_pos = g.axes[3, 0].get_position()

mnist_y = (row0_pos.y1 + row1_pos.y0) / 2
emnist_y = (row2_pos.y1 + row3_pos.y0) / 2

x_pos = row0_pos.x0 - 0.06

g.fig.text(x_pos, mnist_y, "MNIST", fontsize=20, fontweight="bold",
           va="center", ha="center", rotation=90)

g.fig.text(x_pos, emnist_y, "EMNIST", fontsize=20, fontweight="bold",
           va="center", ha="center", rotation=90)

save_dir = "./plot/Per_class"
os.makedirs(save_dir, exist_ok=True)

plt.savefig(f"{save_dir}/global_round_25.pdf", dpi=300, bbox_inches="tight")
plt.show()