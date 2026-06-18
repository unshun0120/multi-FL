import csv
import matplotlib.pyplot as plt
import os

DATASET_NAME = "CIFAR10"
DATASET_MODE = 'single_1'
DATASET_MODE_1 = 'single'

# DATASET_NAME = "SuperDataset"
# DATASET_MODE = 'super'

CSV_FILES = [

    # f"./logs/2026-05-11_18-56-10/Ours_GeFL/{DATASET_MODE}/our_finetune_global_{DATASET_MODE_1}_dataset/{DATASET_NAME}_newclient_history.csv",
    # f"./logs/2026-05-12_02-19-31/FedTED/{DATASET_MODE}/our_finetune_global_{DATASET_MODE_1}_dataset/{DATASET_NAME}_newclient_history.csv",
    # f"./logs/2026-05-12_12-45-34/FedTED_dir/{DATASET_MODE}/our_finetune_global_{DATASET_MODE_1}_dataset/{DATASET_NAME}_newclient_history.csv",
    # f"./logs/2026-05-12_00-51-46/UDON/{DATASET_MODE}/our_finetune_global_{DATASET_MODE_1}_dataset/{DATASET_NAME}_newclient_history.csv",
    # f"./logs/2026-05-11_18-56-10/Ours_GeFL/scratch_{DATASET_MODE}/baseline_{DATASET_MODE_1}_dataset/{DATASET_NAME}_newclient_history.csv",

    f"./logs/2026-05-12_10-39-36/Ours_GeFL/{DATASET_MODE}/our_finetune_global_{DATASET_MODE_1}_dataset/{DATASET_NAME}_newclient_history.csv",
    f"./logs/2026-05-12_07-04-29/FedTED/{DATASET_MODE}/our_finetune_global_{DATASET_MODE_1}_dataset/{DATASET_NAME}_newclient_history.csv",
    f"./logs/2026-05-12_17-34-01/FedTED_dir/{DATASET_MODE}/our_finetune_global_{DATASET_MODE_1}_dataset/{DATASET_NAME}_newclient_history.csv",
    f"./logs/2026-05-12_05-37-28/UDON/{DATASET_MODE}/our_finetune_global_{DATASET_MODE_1}_dataset/{DATASET_NAME}_newclient_history.csv",
    f"./logs/2026-05-12_10-39-36/Ours_GeFL/scratch_{DATASET_MODE}/baseline_{DATASET_MODE_1}_dataset/{DATASET_NAME}_newclient_history.csv",

    # f"../logs/2026-05-01_18-03-18/Ours_GeFL_realimg/{DATASET_MODE}/our_finetune_global_{DATASET_MODE}_dataset/{DATASET_NAME}_newclient_history.csv",
    # f"../logs/2026-05-02_15-52-18/Ours_GeFL_gan/{DATASET_MODE}/our_finetune_global_{DATASET_MODE}_dataset/{DATASET_NAME}_newclient_history.csv",
    # f"../logs/2026-05-01_16-40-45/Ours_GeFL_classname/{DATASET_MODE}/our_finetune_global_{DATASET_MODE}_dataset/{DATASET_NAME}_newclient_history.csv",
]

LABELS = [
    "Ours_GeFL",
    "FedTED",
    "FedTED_dir",
    "UDON",
    "Random_init",    
    
#     "Ours_GeFL_realimg",
#     "Ours_GeFL_gan",
#     "Ours_GeFL_classname",
]

# PLOT_TITLE = f"{DATASET_NAME} Accuracy"
PLOT_TITLE = f"{DATASET_NAME} Accuracy"
# PLOT_TITLE = f"Mix-dataset Accuracy"
OUTPUT_DIR = "./plot/new_client/mnist+emnist+fashionmnist"
OUTPUT_NAME = os.path.join(OUTPUT_DIR, f"{DATASET_NAME}.png")

def main():
    plt.figure(figsize=(10, 6))

    for filepath, label in zip(CSV_FILES, LABELS):
        # if not os.path.exists(filepath):
        #     continue

        abs_path = os.path.abspath(filepath) 
        if not os.path.exists(filepath):
            print(f"file not exit: \n   {abs_path}\n")
            continue
        
        epochs = []
        accuracies = []
        
        with open(filepath, mode='r') as f:
            reader = csv.reader(f)
            header = next(reader) 
            for row in reader:
                if len(row) >= 2:
                    epochs.append(int(row[0]))
                    accuracies.append(float(row[1]))

        plt.plot(epochs, accuracies, marker='o', markersize=4, label=label)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.xlim(0, 10)
    plt.title(PLOT_TITLE, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_NAME, dpi=300)

if __name__ == "__main__":
    main()