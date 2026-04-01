import os
import matplotlib.pyplot as plt

def plot_accuracy_curves(history, save_dir, args, global_rounds, dirichlet_alpha):
    if not history: return

    cmap = plt.get_cmap('tab10')
    all_datasets = sorted(list(history.keys()))

    plt.figure(figsize=(10, 6))

    for i, d_name in enumerate(all_datasets):
        acc_list = history[d_name]
        
        if acc_list and len(acc_list) > 0:
            rounds = range(1, len(acc_list) + 1)
            color = cmap(i)
            plt.plot(rounds, acc_list, marker='', linestyle='-', linewidth=2, color=color, label=d_name)

    plt.title(f"Training Clients Accuracy (Alpha={dirichlet_alpha})")
    #plt.xlim(0, global_rounds)
    plt.xlabel("Server Rounds")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    filename = os.path.join(save_dir, f"{args.algorithm}_accuracy_curve.png")

    plt.savefig(filename)
    plt.close()

def plot_new_client_accuracy(curves, args, dataset_name, model_list, save_dir):
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
