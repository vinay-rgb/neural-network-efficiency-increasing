"""Data loading utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from .config import BATCH_SIZE, ROOT


def get_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


def get_mnist_loaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader) for MNIST."""
    tf = get_transforms()
    data_dir = ROOT / "data"

    train_ds = datasets.MNIST(data_dir, train=True,  download=True, transform=tf)
    test_ds  = datasets.MNIST(data_dir, train=False, download=True, transform=tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader


def get_car_loaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader) for Car Evaluation."""
    car = fetch_openml(name='car', version=3, parser='auto')
    X = pd.get_dummies(car.data).values.astype(np.float32)
    y = LabelEncoder().fit_transform(car.target).astype(np.int64)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def get_sonar_loaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader) for Sonar."""
    sonar = fetch_openml(name='sonar', version=1, parser='auto')
    X = sonar.data.values.astype(np.float32)
    X = StandardScaler().fit_transform(X).astype(np.float32)
    y = LabelEncoder().fit_transform(sonar.target).astype(np.int64)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def get_loaders(
    dataset_name: str = "mnist",
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Return dataloaders based on dataset name."""
    if dataset_name == "mnist":
        return get_mnist_loaders(batch_size, num_workers)
    elif dataset_name == "car":
        return get_car_loaders(batch_size, num_workers)
    elif dataset_name == "sonar":
        return get_sonar_loaders(batch_size, num_workers)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")



