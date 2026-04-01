"""
讀取多個 CSV 檔案，畫出每個 (dataset, model) 的準確率比較圖。

用法:
    python utils/plot_comparison.py \
        --csvs helpTrain:path/to/helpTrain/accuracy_log.csv \
               helpTest:path/to/helpTest/accuracy_log.csv \
               baseline:path/to/baseline/accuracy_log.csv \
        --output_dir logs/2026-02-23_12-37-52/Ours/
"""

import os
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime


def load_and_merge_csvs(csv_specs):
    """
    csv_specs: list of "label:path" strings
    Returns merged DataFrame with an extra 'label' column (overrides 'method' if given).
    """
    dfs = []
    for spec in csv_specs:
        if ':' not in spec:
            print(f"[Error] Format must be 'label:path', got: {spec}")
            continue
        label, path = spec.split(':', 1)
        if not os.path.exists(path):
            print(f"[Warning] File not found: {path}")
            continue
        
        df = pd.read_csv(path)
        df['label'] = label  # Use the user-provided label
        dfs.append(df)
        print(f"  Loaded: {label} <- {path} ({len(df)} rows)")
    
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)


def plot_all(df, output_dir, acc_col='combined_acc', figsize=(10, 6)):
    """
    For each (dataset, model), plot accuracy curves from all methods.
    """
    os.makedirs(output_dir, exist_ok=True)

    datasets = sorted(df['dataset'].unique())
    models = sorted(df['model'].unique())
    labels = df['label'].unique()

    # Style config
    colors = plt.cm.tab10.colors
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']

    count = 0
    for dataset in datasets:
        for model in models:
            fig, ax = plt.subplots(figsize=figsize)
            has_data = False

            for idx, label in enumerate(labels):
                subset = df[(df['dataset'] == dataset) & 
                            (df['model'] == model) & 
                            (df['label'] == label)]
                
                if subset.empty:
                    continue
                
                # Sort by epoch
                subset = subset.sort_values('epoch')
                epochs = subset['epoch'].values
                accs = subset[acc_col].values

                color = colors[idx % len(colors)]
                ls = linestyles[idx % len(linestyles)]
                marker = markers[idx % len(markers)]

                best = accs.max()
                final = accs[-1]

                ax.plot(epochs, accs,
                        label=f'{label} (best={best:.2f}%, final={final:.2f}%)',
                        color=color, linestyle=ls, marker=marker,
                        markersize=4, markevery=max(1, len(epochs) // 15),
                        linewidth=1.5, alpha=0.85)
                has_data = True

            if has_data:
                ax.set_xlabel('Epoch', fontsize=12)
                ax.set_ylabel('Accuracy (%)', fontsize=12)
                ax.set_title(f'{dataset} - {model}', fontsize=14, fontweight='bold')
                ax.legend(fontsize=9, loc='lower right')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(left=0)

                filepath = os.path.join(output_dir, f'{dataset}_{model}.png')
                fig.tight_layout()
                fig.savefig(filepath, dpi=150)
                count += 1

            plt.close(fig)

    print(f"\n  Total plots: {count}")

    # Also generate summary table
    print_summary(df, output_dir, acc_col)


def print_summary(df, output_dir, acc_col='combined_acc'):
    """Print and save a summary table."""
    summary_path = os.path.join(output_dir, 'summary.txt')

    lines = []
    datasets = sorted(df['dataset'].unique())
    labels = df['label'].unique()

    for dataset in datasets:
        lines.append(f"\n{'='*80}")
        lines.append(f"  Dataset: {dataset}")
        lines.append(f"{'='*80}")

        header = f"  {'Model':<20}"
        for label in labels:
            header += f" {label+' (final)':>18} {label+' (best)':>18}"
        lines.append(header)
        lines.append(f"  {'-' * (20 + 36 * len(labels))}")

        models = sorted(df[df['dataset'] == dataset]['model'].unique())
        for model in models:
            row = f"  {model:<20}"
            for label in labels:
                subset = df[(df['dataset'] == dataset) & 
                            (df['model'] == model) & 
                            (df['label'] == label)]
                if subset.empty:
                    row += f" {'N/A':>18} {'N/A':>18}"
                else:
                    accs = subset.sort_values('epoch')[acc_col].values
                    row += f" {accs[-1]:>17.2f}% {accs.max():>17.2f}%"
            lines.append(row)

    text = '\n'.join(lines)
    print(text)

    with open(summary_path, 'w') as f:
        f.write(text)
    print(f"\n  Summary saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--csvs', nargs='+', required=True,
                        help='CSV files: "label:path" (e.g., "helpTrain:path/to/csv")')
    parser.add_argument('--output_dir', type=str, default='plots/comparison/')
    parser.add_argument('--acc_col', type=str, default='combined_acc',
                        choices=['client_acc', 'gen_acc', 'combined_acc'],
                        help='Which accuracy column to plot')
    parser.add_argument('--figsize', type=float, nargs=2, default=[10, 6])

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(args.output_dir, exist_ok=True)

    source_log_path = os.path.join(args.output_dir, 'csv_sources.txt')
    with open(source_log_path, 'w') as f:
        f.write(f"Plot generation time: {timestamp}\n")
        f.write(f"Target accuracy column: {args.acc_col}\n")
        f.write("="*50 + "\n")
        f.write("CSV Sources:\n")
        for spec in args.csvs:
            f.write(f"  - {spec}\n")
    print(f"\n[Info] Created output directory: {args.output_dir}")
    print(f"[Info] Recorded CSV sources to : {source_log_path}\n")

    print("Loading CSVs...")
    df = load_and_merge_csvs(args.csvs)

    if df.empty:
        print("[Error] No data loaded.")
        return

    # Convert acc column to float
    df[args.acc_col] = pd.to_numeric(df[args.acc_col], errors='coerce')
    df = df.dropna(subset=[args.acc_col])

    print(f"\nTotal rows: {len(df)}")
    print(f"Datasets: {sorted(df['dataset'].unique())}")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Methods: {list(df['label'].unique())}")

    print(f"\nPlotting ({args.acc_col}) to: {args.output_dir}")
    plot_all(df, args.output_dir, acc_col=args.acc_col, figsize=tuple(args.figsize))
    print("\nDone!")


if __name__ == '__main__':
    main()