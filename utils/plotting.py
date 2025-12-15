import os
import matplotlib.pyplot as plt

def plot_accuracy_curves(history, save_dir, args, filename="accuracy_curve.png"):
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
    
    save_path_train = os.path.join(save_dir, filename.replace('.png', '_train.png'))
    plt.savefig(save_path_train)
    plt.close()
