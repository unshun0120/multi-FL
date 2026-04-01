import os
import csv

CSV_HEADER = ['method', 'dataset', 'model', 'epoch', 'client_acc', 'gen_acc', 'combined_acc']

def init_csv(save_dir, filename='accuracy_log.csv'):
    """Initialize CSV file with header. Returns filepath."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
    return filepath

def append_csv(filepath, method, dataset, model, epoch, client_acc=None, gen_acc=None, combined_acc=None):
    """Append one row to CSV."""
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            method, dataset, model, epoch,
            f'{client_acc:.2f}' if client_acc is not None else '',
            f'{gen_acc:.2f}' if gen_acc is not None else '',
            f'{combined_acc:.2f}' if combined_acc is not None else '',
        ])