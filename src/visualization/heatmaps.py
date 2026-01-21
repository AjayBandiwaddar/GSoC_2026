import torch
import matplotlib.pyplot as plt


def reconstruction_error_map(original, reconstructed):
    """
    Compute pixel-wise reconstruction error map.
    """
    return (original - reconstructed).pow(2).squeeze().cpu().numpy()


def save_error_heatmap(error_map, save_path, cmap="hot"):
    """
    Save a visualization of the reconstruction error heatmap.
    """
    plt.figure(figsize=(4, 4))
    plt.imshow(error_map, cmap=cmap)
    plt.colorbar()
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
