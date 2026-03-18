"""CIFAR-10 data loading with normalization to [-1, 1]."""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar10_dataloader(
    data_dir: str = "./data",
    batch_size: int = 128,
    image_size: int = 32,
    num_workers: int = 4,
    train: bool = True,
) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=True,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def unnormalize(x: torch.Tensor) -> torch.Tensor:
    """Map from [-1, 1] back to [0, 1] for visualization."""
    return (x + 1) / 2
