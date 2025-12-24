import os
import matplotlib.pyplot as plt

def plot_accuracy_curves(history, save_dir, args, mode=""):
    train_data = history.get('train_detail', {})
    if not train_data: return

    cmap = plt.get_cmap('tab10')
    all_datasets = sorted(list(train_data.keys()))

    plt.figure(figsize=(10, 6))

    for i, d_name in enumerate(all_datasets):
        acc_list = train_data[d_name]
        
        if acc_list and len(acc_list) > 0:
            rounds = range(len(acc_list))
            color = cmap(i)
            plt.plot(rounds, acc_list, marker='', linestyle='-', linewidth=2, color=color, label=d_name)

    plt.title(f"Training Clients Accuracy (Alpha={args.dirichlet_alpha})")
    plt.xlim(0, args.global_rounds)
    plt.xlabel("Server Rounds")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    
    filename = "accuracy_curve.png"
    if mode == "Ours":
        filename = "Ours" + filename
    elif mode == "local_only":
        filename = "local_only" + filename

    plt.savefig(filename)
    plt.close()

def plot_new_client_accuracy(curves, save_dir, args, dataset_name, model_list):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure()

    for arch_id, acc_list in curves.items():
        name = model_list[arch_id]
        plt.plot(range(len(acc_list)), acc_list, linewidth=1.0, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Baseline: New Clients Training from Scratch (No Generator, Single dataset) - {dataset_name}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)

    out_path = os.path.join(save_dir, f"{dataset_name}_scratch_single_dataset.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
