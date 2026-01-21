import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import os
import torch
import torch.nn as nn

from src.datasets.mnist import load_mnist
from src.models.autoencoder import Autoencoder
from src.evaluation.metrics import compute_auc
from src.utils.corruptions import gaussian_noise
from src.visualization.heatmaps import (
    reconstruction_error_map,
    save_error_heatmap
)


EPOCHS = 5
LR = 1e-3
DEVICE = "cpu"
BOTTLENECK_DIM = 8

os.makedirs("results/plots", exist_ok=True)


def main():
    # Load data
    train_loader, train_dataset = load_mnist(batch_size=128)

    # Model
    model = Autoencoder(bottleneck_dim=BOTTLENECK_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # Training
    model.train()
    for epoch in range(EPOCHS):
        for img, _ in train_loader:
            img = img.to(DEVICE)
            optimizer.zero_grad()
            recon = model(img)
            loss = criterion(recon, img)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    normal_errors = []
    anomaly_errors = []

    with torch.no_grad():
        for i in range(200):
            img, _ = train_dataset[i]
            img = img.unsqueeze(0)

            recon = model(img)
            err = torch.mean((recon - img) ** 2).item()
            normal_errors.append(err)

            anom = gaussian_noise(img)
            recon_anom = model(anom)
            err_anom = torch.mean((recon_anom - anom) ** 2).item()
            anomaly_errors.append(err_anom)

    auc = compute_auc(normal_errors, anomaly_errors)
    print(f"AUC: {auc:.4f}")

    # Save one example heatmap
    img, _ = train_dataset[0]
    img = img.unsqueeze(0)

    with torch.no_grad():
        recon = model(img)

    error_map = reconstruction_error_map(img, recon)
    save_error_heatmap(
        error_map,
        "results/plots/example_error_heatmap.png"
    )

    print("Baseline experiment completed.")


if __name__ == "__main__":
    main()
