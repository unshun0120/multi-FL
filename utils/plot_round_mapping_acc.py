import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    csv_path = 'logs/2026-04-30_15-54-33/GeFL/mapping_results/GeFL_mapping_acc_per_round.csv'
    
    if not os.path.exists(csv_path):
        print(f"No file: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    df.columns = df.columns.str.strip()

    output_dir = 'logs/2026-04-30_15-54-33/GeFL/mapping_results/plots'
    os.makedirs(output_dir, exist_ok=True)

    entropy_ratios = sorted(df['entropy_ratio'].unique())

    for ratio in entropy_ratios:
        subset = df[df['entropy_ratio'] == ratio].copy()
        
        subset = subset.sort_values(by='global_round')
        
        plt.figure(figsize=(8, 5))
        plt.plot(subset['global_round'], subset['recall'], marker='o', linestyle='-', color='b', linewidth=2)
        
        plt.title(f'Entropy Threshold = {ratio}')
        plt.xlabel('Global Round')
        plt.ylabel('Recall')
        
        plt.xticks(subset['global_round'])
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(0, 1.05) 
        
        filename = f'recall_entropy_{ratio:.2f}.png'
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, bbox_inches='tight')
        
        plt.close()

if __name__ == "__main__":
    main()