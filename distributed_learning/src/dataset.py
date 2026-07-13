"""
Chargement et partition du dataset pour chaque volontaire.

Datasets disponibles
--------------------
  - cifar10   : CIFAR-10  — 50 000 images train, 10 000 test, 10 classes
  - cifar100  : CIFAR-100 — 50 000 images train, 10 000 test, 100 classes
  - imagenet  : ImageNet  — ~1,28 M images train, 50 000 val, 1000 classes
                Chemin attendu : <data_dir>/imagenet/
                Structure : imagenet/train/<class_id>/... et imagenet/val/<class_id>/...

Toutes les images sont redimensionnées à 224×224 pour être compatibles avec
les architectures ResNet et VGG utilisées.

Partitions
----------
  - 'iid'     : tirage aléatoire uniforme
  - 'non-iid' : chaque volontaire reçoit principalement un sous-ensemble de classes
"""
import logging
import math
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T


# ─── Taille d'entrée standard pour tous les modèles ─────────────────────────
_INPUT_SIZE = 224  # ResNet-50 attend 224×224

# Normalisation ImageNet (utilisée pour les 3 datasets)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

# Nombre de classes par dataset
DATASET_NUM_CLASSES = {
    "cifar10":  10,
    "cifar100": 100,
    "imagenet": 1000,
}

# Taille de l'ensemble d'entraînement par dataset
DATASET_TRAIN_SIZE = {
    "cifar10":  50_000,
    "cifar100": 50_000,
    "imagenet": 1_281_167,  # valeur de référence ILSVRC
}


def resolve_data_dir(data_dir: str) -> str:
    """Résout un chemin de données de façon robuste depuis n'importe quel cwd."""
    path = Path(data_dir).expanduser()
    if path.is_absolute():
        return str(path)

    project_root = Path(__file__).resolve().parents[1]
    candidate = (project_root / path).resolve()
    if candidate.exists():
        return str(candidate)

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return str(cwd_candidate)

    return str(candidate)


def _dataset_files_present(data_dir: str, dataset_name: str) -> bool:
    ds = dataset_name.lower().strip()
    if ds == "cifar10":
        return os.path.isdir(os.path.join(data_dir, "cifar-10-batches-py"))
    if ds == "cifar100":
        return os.path.isdir(os.path.join(data_dir, "cifar-100-python"))
    if ds == "imagenet":
        train_path = os.path.join(data_dir, "imagenet", "train")
        val_path = os.path.join(data_dir, "imagenet", "val")
        return os.path.isdir(train_path) and os.path.isdir(val_path)
    return False


def _try_load_dataset_or_raise(data_dir: str, dataset_name: str):
    allow_download = os.getenv("ALLOW_DATASET_DOWNLOAD", "0").lower() in {"1", "true", "yes", "on"}
    if _dataset_files_present(data_dir, dataset_name):
        return False

    if allow_download:
        logging.info("[Dataset] Aucun jeu de données local détecté ; téléchargement autorisé.")
        return True

    raise FileNotFoundError(
        f"Données locales introuvables pour '{dataset_name}' dans '{data_dir}'. "
        "Placez les fichiers du dataset dans ce dossier ou définissez ALLOW_DATASET_DOWNLOAD=1."
    )


def load_dataset(dataset: str,
                 data_dir: str,
                 volunteer_id: int,
                 n_volunteers: int,
                 partition: str = "iid",
                 batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """
    Charge le dataset et retourne (train_loader, test_loader) pour ce volontaire.
    Les données test sont complètes pour tous les volontaires (évaluation globale).

    Args:
        dataset      : 'cifar10', 'cifar100' ou 'imagenet'.
        data_dir     : répertoire racine pour les données téléchargées.
        volunteer_id : indice du volontaire (0-indexé).
        n_volunteers : nombre total de volontaires.
        partition    : 'iid' ou 'non-iid'.
        batch_size   : taille des batchs d'entraînement.

    Returns:
        (train_loader, test_loader)
    """
    resolved_data_dir = resolve_data_dir(data_dir)
    os.makedirs(resolved_data_dir, exist_ok=True)
    ds = dataset.lower().strip()

    if ds == "cifar10":
        allow_download = _try_load_dataset_or_raise(resolved_data_dir, ds)
        train_ds, test_ds = _load_cifar10(resolved_data_dir, download=allow_download)

    elif ds == "cifar100":
        allow_download = _try_load_dataset_or_raise(resolved_data_dir, ds)
        train_ds, test_ds = _load_cifar100(resolved_data_dir, download=allow_download)

    elif ds == "imagenet":
        train_ds, test_ds = _load_imagenet(resolved_data_dir)

    else:
        raise ValueError(
            f"Dataset inconnu : '{dataset}'. "
            f"Datasets supportés : {list(DATASET_NUM_CLASSES.keys())}"
        )

    # ── Targets pour la partition ──────────────────────────────────────────
    if hasattr(train_ds, "targets"):
        targets = np.array(train_ds.targets)
    else:
        # ImageFolder utilise .targets; sinon on scanne (lent, fallback seulement)
        logging.warning("[Dataset] Chargement des targets par scan — peut être lent.")
        targets = np.array([t for _, t in train_ds])

    # ── Partition des données d'entraînement ───────────────────────────────
    if partition == "iid":
        indices = _iid_partition(train_ds, volunteer_id, n_volunteers)
    elif partition == "non-iid":
        indices = _non_iid_partition(targets, volunteer_id, n_volunteers)
    else:
        raise ValueError(f"Partition inconnue : '{partition}'. Valeurs : 'iid', 'non-iid'.")

    train_subset = Subset(train_ds, indices)
    labels = np.unique(targets[indices]).tolist()
    logging.info(
        f"[Dataset] Volontaire {volunteer_id}/{n_volunteers} "
        f"— dataset={ds} partition={partition} : "
        f"{len(train_subset)} exemples — classes {labels}"
    )

    train_loader = DataLoader(
        train_subset, batch_size=batch_size,
        shuffle=True, num_workers=0, pin_memory=False,
        persistent_workers=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=32,
        shuffle=False, num_workers=0, pin_memory=False,
        persistent_workers=False,
    )
    return train_loader, test_loader


# ─── Chargement par dataset ──────────────────────────────────────────────────

def _cifar_train_transforms() -> T.Compose:
    """Transformations d'augmentation entraînement pour CIFAR (32x32)."""
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def _cifar_test_transforms() -> T.Compose:
    """Transformations test pour CIFAR (32x32, sans augmentation)."""
    return T.Compose([
        T.ToTensor(),
        T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def _load_cifar10(data_dir: str, download: bool = False):
    train_ds = torchvision.datasets.CIFAR10(
        data_dir, train=True, download=download,
        transform=_cifar_train_transforms(),
    )
    test_ds = torchvision.datasets.CIFAR10(
        data_dir, train=False, download=download,
        transform=_cifar_test_transforms(),
    )
    return train_ds, test_ds


def _load_cifar100(data_dir: str, download: bool = False):
    train_ds = torchvision.datasets.CIFAR100(
        data_dir, train=True, download=download,
        transform=_cifar_train_transforms(),
    )
    test_ds = torchvision.datasets.CIFAR100(
        data_dir, train=False, download=download,
        transform=_cifar_test_transforms(),
    )
    return train_ds, test_ds


def _load_imagenet(data_dir: str):
    """
    Charge ImageNet depuis <data_dir>/imagenet/.
    Structure attendue :
        <data_dir>/imagenet/train/<class_dir>/image.JPEG ...
        <data_dir>/imagenet/val/<class_dir>/image.JPEG ...

    Si le répertoire n'existe pas, lève une erreur claire.
    """
    imagenet_root = os.path.join(data_dir, "imagenet")
    train_path = os.path.join(imagenet_root, "train")
    val_path   = os.path.join(imagenet_root, "val")

    if not os.path.isdir(train_path):
        raise FileNotFoundError(
            f"ImageNet introuvable : '{train_path}' n'existe pas.\n"
            "Préparez le dataset avec la structure :\n"
            "  <data_dir>/imagenet/train/<class_id>/\n"
            "  <data_dir>/imagenet/val/<class_id>/\n"
            "Voir : https://image-net.org/challenges/LSVRC/2012/"
        )

    tf_train = T.Compose([
        T.RandomResizedCrop(_INPUT_SIZE),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
    tf_val = T.Compose([
        T.Resize(256),
        T.CenterCrop(_INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])

    train_ds = torchvision.datasets.ImageFolder(train_path, transform=tf_train)
    test_ds  = torchvision.datasets.ImageFolder(val_path,   transform=tf_val)
    return train_ds, test_ds


# ─── Fonctions de partition ─────────────────────────────────────────────────

def _iid_partition(dataset, vol_id: int, n: int):
    """Partition IID : chaque volontaire reçoit 1/n des données, tirage aléatoire."""
    total = len(dataset)
    idx   = np.random.permutation(total)
    chunk = total // n
    start = vol_id * chunk
    end   = start + chunk if vol_id < n - 1 else total
    return idx[start:end].tolist()


def _non_iid_partition(targets: np.ndarray, vol_id: int, n: int):
    """
    Partition Non-IID : chaque volontaire se voit attribuer un sous-ensemble
    de classes (≈ 2 classes pour 10 classes et 5 volontaires).
    """
    n_classes = len(np.unique(targets))
    classes_per_vol = max(2, math.ceil(n_classes / n))
    assigned = [(vol_id * classes_per_vol + i) % n_classes for i in range(classes_per_vol)]

    indices = []
    for c in assigned:
        indices.extend(np.where(targets == c)[0].tolist())
    np.random.shuffle(indices)
    return indices


def get_input_size(dataset: str) -> tuple:
    """Retourne (C, H, W) pour le dataset donné."""
    ds = dataset.lower().strip()
    if ds in ("cifar10", "cifar100"):
        return (3, 32, 32)
    return (3, 224, 224)


def get_num_classes(dataset: str) -> int:
    """Retourne le nombre de classes pour le dataset donné."""
    ds = dataset.lower().strip()
    if ds not in DATASET_NUM_CLASSES:
        raise ValueError(f"Dataset inconnu : '{dataset}'")
    return DATASET_NUM_CLASSES[ds]