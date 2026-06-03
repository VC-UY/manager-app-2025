"""
Chargement et partition du dataset pour chaque volontaire.

Partitions :
  - 'iid'     : tirage aléatoire uniforme
  - 'non-iid' : chaque volontaire reçoit principalement 2 classes (hétérogénéité réelle)
"""
import os
import math
import logging
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T


def load_dataset(dataset: str,
                 data_dir: str,
                 volunteer_id: int,
                 n_volunteers: int,
                 partition: str = "iid",
                 batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """
    Charge le dataset et retourne (train_loader, test_loader) pour ce volontaire.
    Les données test sont complètes pour tous les volontaires (évaluation globale).
    """
    os.makedirs(data_dir, exist_ok=True)

    if dataset == "mnist":
        tf_train = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
        tf_test  = tf_train
        train_ds = torchvision.datasets.MNIST(data_dir, train=True,  download=False, transform=tf_train)
        test_ds  = torchvision.datasets.MNIST(data_dir, train=False, download=False, transform=tf_test)

    elif dataset == "cifar10":
        tf_train = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        tf_test = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_ds = torchvision.datasets.CIFAR10(data_dir, train=True,  download=False, transform=tf_train)
        test_ds  = torchvision.datasets.CIFAR10(data_dir, train=False, download=False, transform=tf_test)
    else:
        raise ValueError(f"Dataset inconnu : {dataset}")

    # Targets utilisés pour analyser la partition et construire des classes.
    if hasattr(train_ds, "targets"):
        targets = np.array(train_ds.targets)
    else:
        targets = np.array([t for _, t in train_ds])

    # Partition des données d'entraînement
    if partition == "iid":
        indices = _iid_partition(train_ds, volunteer_id, n_volunteers)
    elif partition == "non-iid":
        indices = _non_iid_partition(train_ds, volunteer_id, n_volunteers)
    else:
        raise ValueError(f"Partition inconnue : {partition}")

    train_subset = Subset(train_ds, indices)
    labels = np.unique(targets[indices]).tolist()
    logging.info(
        f"[Dataset] Volontaire {volunteer_id}/{n_volunteers} "
        f"— partition {partition} : {len(train_subset)} exemples "
        f"— classes {labels}"
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,      batch_size=256,
                              shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, test_loader


def _iid_partition(dataset, vol_id: int, n: int):
    total = len(dataset)
    idx   = np.random.permutation(total)
    chunk = total // n
    start = vol_id * chunk
    end   = start + chunk if vol_id < n - 1 else total
    return idx[start:end].tolist()


def _non_iid_partition(dataset, vol_id: int, n: int):
    """Bloc de classes par volontaire (distribution non uniforme)."""
    if hasattr(dataset, "targets"):
        targets = np.array(dataset.targets)
    else:
        targets = np.array([t for _, t in dataset])

    n_classes = len(np.unique(targets))
    classes_per_vol = max(2, math.ceil(n_classes / n))
    assigned = [(vol_id * classes_per_vol + i) % n_classes for i in range(classes_per_vol)]

    indices = []
    for c in assigned:
        indices.extend(np.where(targets == c)[0].tolist())
    np.random.shuffle(indices)
    return indices
