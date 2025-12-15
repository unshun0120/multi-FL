import os
from torchvision import datasets

DATA_ROOT = './data/raw'

def prepare_datasets():
    print("=== Preparing Datasets (MNIST, FashionMNIST, EMNIST, CIFAR10, CIFAR100) ===")

    if not os.path.exists(DATA_ROOT):
        os.makedirs(DATA_ROOT)

    print("Preparing MNIST ...")
    datasets.MNIST(root=DATA_ROOT, train=True,  download=True)
    datasets.MNIST(root=DATA_ROOT, train=False, download=True)

    print("Preparing Fashion-MNIST ...")
    datasets.FashionMNIST(root=DATA_ROOT, train=True,  download=True)
    datasets.FashionMNIST(root=DATA_ROOT, train=False, download=True)

    print("Preparing EMNIST (balanced) ...")
    datasets.EMNIST(root=DATA_ROOT, split='balanced', train=True,  download=True)
    datasets.EMNIST(root=DATA_ROOT, split='balanced', train=False, download=True)

    print("Preparing CIFAR-10 ...")
    datasets.CIFAR10(root=DATA_ROOT, train=True,  download=True)
    datasets.CIFAR10(root=DATA_ROOT, train=False, download=True)

    print("Preparing CIFAR-100 ...")
    datasets.CIFAR100(root=DATA_ROOT, train=True,  download=True)
    datasets.CIFAR100(root=DATA_ROOT, train=False, download=True)

if __name__ == "__main__":
    prepare_datasets()
