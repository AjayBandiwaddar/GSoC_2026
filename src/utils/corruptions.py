import torch
import random


def pixel_dropout(img, drop_prob=0.1):
    """
    Randomly drops pixels with probability drop_prob.
    """
    mask = torch.rand_like(img)
    return img * (mask > drop_prob)


def gaussian_noise(img, sigma=0.3):
    """
    Adds Gaussian noise to the image.
    """
    noise = sigma * torch.randn_like(img)
    out = img + noise
    return torch.clamp(out, 0.0, 1.0)


def stripe(img):
    """
    Adds a random horizontal or vertical stripe.
    """
    img = img.clone()

    _, h, w = img.shape

    if random.random() < 0.5:
        # horizontal stripe
        row = random.randint(0, h - 1)
        img[:, row, :] = 1.0
    else:
        # vertical stripe
        col = random.randint(0, w - 1)
        img[:, :, col] = 1.0

    return img


def random_patch(img, size=5):
    """
    Adds a random square patch.
    """
    img = img.clone()

    _, h, w = img.shape
    size = min(size, h, w)

    x = random.randint(0, h - size)
    y = random.randint(0, w - size)

    img[:, x:x+size, y:y+size] = 1.0

    return img
