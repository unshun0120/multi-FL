import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df_independent = pd.read_csv('logs/2026-06-17_13-55-17/GeFL_DDPM_baseline_total_public/global_model_class_acc.csv')
df_independent['Mapping'] = 'Independent'

# df_identical = pd.read_csv('logs/2026-06-16_00-35-08/BaseFL_public/global_model_class_acc.csv')
# df_identical['Mapping'] = 'Identical'

df_class_name = pd.read_csv('logs/2026-06-17_13-34-19/GeFL_DDPM_baseline_total_public/global_model_class_acc.csv')
df_class_name['Mapping'] = 'Class_Name'

# df_bi = pd.read_csv('logs/2026-06-16_01-09-42/GeFL_DDPM_baseline_total/global_model_class_acc.csv')
# df_bi['Mapping'] = 'Bi-Direct (Ours)'

# df_feat = pd.read_csv('logs/2026-06-16_01-09-51/GeFL_DDPM_baseline_total/global_model_class_acc.csv')
# df_feat['Mapping'] = 'Feature-Bi'

#df_all = pd.concat([df_independent, df_identical, df_class_name], ignore_index=True)
#df_all = pd.concat([df_bi, df_feat], ignore_index=True)
df_all = pd.concat([df_independent, df_class_name], ignore_index=True)

target_dataset = 'MNIST'
target_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

df_filtered = df_all[(df_all['Dataset'] == target_dataset) & 
                     (df_all['Local_Class'].isin(target_classes))]

sns.set_theme(style="whitegrid")

g = sns.relplot(
    data=df_filtered,
    x="Round", 
    y="Accuracy", 
    hue="Mapping",          
    col="Local_Class",      
    col_wrap=5,            
    kind="line",            
    height=3,               
    aspect=1.2,             
    linewidth=2.5,
    palette=['#1f77b4', '#ff7f0e', '#2ca02c']
)

g.set_axis_labels("Global Round", "Accuracy (%)")
g.set_titles("Class {col_name}")

for ax in g.axes.flat:
    ax.set_ylim(0, 105)

save_dir = f'./plot/Per_class/baseline/{target_dataset}'
os.makedirs(save_dir, exist_ok=True)  

plt.savefig(f'{save_dir}/class_comparison.pdf', dpi=300, bbox_inches='tight')