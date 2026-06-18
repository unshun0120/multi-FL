import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_student_accuracy_curve(csv_files, dataset_name="MNIST", save_dir="./"):
    plt.figure(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, file_path in enumerate(csv_files):
        if not os.path.exists(file_path):
            print(f"Can't find file: {file_path}")
            continue
            
        df = pd.read_csv(file_path)
        
        x_data = df['Epoch']
        y_data = df['Accuracy']
        
        method_name = os.path.basename(file_path).split('_')[0]
        
        plt.plot(
            x_data, y_data, 
            color=colors[idx % len(colors)], 
            linewidth=2, 
            label=f"{method_name} (Max: {y_data.max():.2f}%)"
        )

    plt.title(f'Student Model Training Accuracy over Epochs ({dataset_name})', fontsize=15, fontweight='bold')
    plt.xlabel('Epochs', fontsize=13)
    plt.ylabel('Test Accuracy (%)', fontsize=13)
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{dataset_name}_oldGen_student_accuracy_curve.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved accuracy curve plot to {out_path}")
    plt.close()

if __name__ == "__main__":
    # base_dir = "logs/Local_iid_het_gb40/new_client_2026-04-08_21-48-09"
    base_dir = "logs/Local_noniid_hom_gb40"
    dataset = "EMNIST"

    csv = [
        # os.path.join(base_dir, "FAST_MNIST_student_acc.csv"),
        # os.path.join(base_dir, "NAYER_MNIST_student_acc.csv"),
        # os.path.join(base_dir, "DI_MNIST_student_acc.csv"),

        os.path.join(base_dir, f"student_cnn_di_oldGen/DI_{dataset}_student_acc.csv"),
        os.path.join(base_dir, f"student_cnn_fast_oldGen/FAST_{dataset}_student_acc.csv"),
        os.path.join(base_dir, f"student_cnn_nayer_oldGen/NAYER_{dataset}_student_acc.csv"),
    ]
    plot_student_accuracy_curve(csv, dataset_name=dataset, save_dir=f"label_mapping/student_cnn_res_img/noniid_hom_gb40/")