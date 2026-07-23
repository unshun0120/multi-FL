import os
import pandas as pd


def read_accuracy_csv(csv_path, dataset_name):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    round_col = 'Round' if 'Round' in df.columns else 'global_round'
    acc_col = 'Accuracy' if 'Accuracy' in df.columns else 'accuracy'
    has_epoch = 'Epoch' in df.columns

    if round_col not in df.columns:
        raise ValueError(f"{csv_path} does not contain Round/global_round column.")

    if acc_col not in df.columns:
        raise ValueError(f"{csv_path} does not contain Accuracy/accuracy column.")

    if has_epoch:
        df = df[[round_col, 'Epoch', acc_col]].copy()
        df.columns = ['Round', 'Epoch', dataset_name]

        df['Round'] = pd.to_numeric(df['Round'], errors='coerce')
        df['Epoch'] = pd.to_numeric(df['Epoch'], errors='coerce')
        df[dataset_name] = pd.to_numeric(df[dataset_name], errors='coerce')

        df = df.dropna(subset=['Round', 'Epoch', dataset_name])
        df = df.groupby(['Round', 'Epoch'], as_index=False)[dataset_name].mean()

    else:
        df = df[[round_col, acc_col]].copy()
        df.columns = ['Round', dataset_name]

        df['Round'] = pd.to_numeric(df['Round'], errors='coerce')
        df[dataset_name] = pd.to_numeric(df[dataset_name], errors='coerce')

        df = df.dropna(subset=['Round', dataset_name])
        df = df.groupby('Round', as_index=False)[dataset_name].mean()

    return df, has_epoch


def create_mix_new_csv(log_dir):
    mnist_path = os.path.join(log_dir, 'global_model_acc_MNIST.csv')
    emnist_path = os.path.join(log_dir, 'global_model_acc_EMNIST.csv')
    cifar10_path = os.path.join(log_dir, 'global_model_acc_CIFAR10.csv')

    mnist_df, mnist_has_epoch = read_accuracy_csv(mnist_path, 'MNIST')
    emnist_df, emnist_has_epoch = read_accuracy_csv(emnist_path, 'EMNIST')
    cifar10_df, cifar10_has_epoch = read_accuracy_csv(cifar10_path, 'CIFAR10')

    has_epoch_list = [mnist_has_epoch, emnist_has_epoch, cifar10_has_epoch]

    if len(set(has_epoch_list)) != 1:
        raise ValueError(
            "CSV Epoch format is inconsistent. "
            "MNIST, EMNIST and CIFAR10 must either all have Epoch columns "
            "or all have no Epoch columns."
        )

    if mnist_has_epoch:
        mix_df = pd.merge(mnist_df, emnist_df, on=['Round', 'Epoch'], how='inner')
        mix_df = pd.merge(mix_df, cifar10_df, on=['Round', 'Epoch'], how='inner')

        mix_df['Accuracy'] = mix_df[['MNIST', 'EMNIST', 'CIFAR10']].mean(axis=1)

        mix_df = mix_df[['Round', 'Epoch', 'Accuracy']]
        mix_df = mix_df.sort_values(['Round', 'Epoch'])

    else:
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
    log_dir = "logs/2026-07-04_12-29-20/GeFL_DeepInversion_gen"

    create_mix_new_csv(log_dir)