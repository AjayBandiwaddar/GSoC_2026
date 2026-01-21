from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader


def load_mnist(batch_size=128, root="data"):
    """
    Load MNIST dataset for anomaly detection experiments.

    Returns:
        train_loader: DataLoader for training
        train_dataset: raw training dataset
    """
    transform = transforms.ToTensor()

    train_dataset = MNIST(
        root=root,
        train=True,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return train_loader, train_dataset
