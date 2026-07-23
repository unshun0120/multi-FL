import os
import pandas as pd


def read_accuracy_csv(csv_path, dataset_name):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    round_col = 'Round' if 'Round' in df.columns else 'global_round'
    acc_col = 'Accuracy' if 'Accuracy' in df.columns else 'accuracy'

    if round_col not in df.columns:
        raise ValueError(f"{csv_path} does not contain Round/global_round column.")

    if acc_col not in df.columns:
        raise ValueError(f"{csv_path} does not contain Accuracy/accuracy column.")

    df = df[[round_col, acc_col]].copy()
    df.columns = ['Round', dataset_name]

    df = df.groupby('Round', as_index=False)[dataset_name].mean()

    return df


def create_mix_new_csv(log_dir):
    mnist_path = os.path.join(log_dir, 'global_model_acc_MNIST.csv')
    emnist_path = os.path.join(log_dir, 'global_model_acc_EMNIST.csv')
    cifar10_path = os.path.join(log_dir, 'global_model_acc_CIFAR10.csv')

    mnist_df = read_accuracy_csv(mnist_path, 'MNIST')
    emnist_df = read_accuracy_csv(emnist_path, 'EMNIST')
    cifar10_df = read_accuracy_csv(cifar10_path, 'CIFAR10')

    mix_df = pd.merge(mnist_df, emnist_df, on='Round', how='inner')
    mix_df = pd.merge(mix_df, cifar10_df, on='Round', how='inner')

    mix_df['Accuracy'] = mix_df[['MNIST', 'EMNIST', 'CIFAR10']].mean(axis=1)

    mix_df = mix_df[['Round', 'Accuracy']]
    mix_df = mix_df.sort_values('Round')

    save_path = os.path.join(log_dir, 'mix_new.csv')
    mix_df.to_csv(save_path, index=False, float_format='%.2f')

    print(f"Saved: {save_path}")
    print(mix_df.to_string(index=False))


if __name__ == "__main__":
    log_dir = [
        # "logs/2026-07-07_12-39-17/GeFL_DDPM_baseline_total_gan", 
        # "logs/2026-07-07_12-39-45/GeFL_DDPM_baseline_total_gan",

        # "logs/2026-07-07_12-42-21/GeFL_DDPM_baseline_total_gan",
        # "logs/2026-07-07_12-42-43/GeFL_DDPM_baseline_total_gan",

        "logs/2026-07-23_12-41-26/GeFL_gan_pacfl_noniid",
        "logs/2026-07-23_12-41-28/GeFL_gan_pacfl_noniid",
    ]

    for log in log_dir : 
    # log_dir = "logs/2026-07-04_07-47-20/GeFL_DDPM_baseline_total_gan"

        create_mix_new_csv(log)