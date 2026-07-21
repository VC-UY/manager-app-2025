"""
Définitions des modèles utilisés par les volontaires pour les expérimentations.

Modèles disponibles
-------------------
  - resnet18   : ResNet-18   (~11 M paramètres)  — léger, idéal pour débuter
  - resnet50   : ResNet-50   (~25 M paramètres)
  - resnet101  : ResNet-101  (~44 M paramètres)
  - resnet152  : ResNet-152  (~60 M paramètres)
  - vgg19      : VGG-19      (~143 M paramètres)

Tous ces modèles acceptent des images 3×224×224 pour ImageNet.
Pour CIFAR-10 et CIFAR-100 (3×32×32), conv1 et maxpool sont adaptés
pour travailler directement en 32×32 (pas de redimensionnement).

Utilisation :
    from src.model import create_model, model_parameter_bytes
    model = create_model("resnet18", num_classes=10)
    model = create_model("resnet50", num_classes=10)
    model = create_model("vgg19",    num_classes=100)
"""

import logging
import torch
import torch.nn as nn
import torchvision.models as tv_models

# ─── Registre des modèles supportés ──────────────────────────────────────────
_SUPPORTED_MODELS = ("resnet18", "resnet50", "resnet101", "resnet152", "vgg19")


def create_model(model_name: str = "resnet18",
                 num_classes: int = 10) -> nn.Module:
    """
    Instancie un modèle pré-architecturé (sans poids pré-entraînés)
    et adapte la couche de sortie au nombre de classes demandé.

    Args:
        model_name  : identifiant du modèle (voir _SUPPORTED_MODELS).
        num_classes : nombre de classes de sortie (10 pour CIFAR-10,
                      100 pour CIFAR-100, 1000 pour ImageNet).

    Returns:
        nn.Module prêt à l'emploi, avec weights=None (initialisation aléatoire).

    Raises:
        ValueError : si model_name n'est pas supporté.
    """
    name = model_name.lower().strip()
    from src.config import DATASET
    is_cifar = DATASET.lower().strip() in ("cifar10", "cifar100")

    if name == "resnet18":
        model = tv_models.resnet18(weights=None)
        if is_cifar:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "resnet50":
        model = tv_models.resnet50(weights=None)
        if is_cifar:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "resnet101":
        model = tv_models.resnet101(weights=None)
        if is_cifar:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "resnet152":
        model = tv_models.resnet152(weights=None)
        if is_cifar:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "vgg19":
        model = tv_models.vgg19(weights=None)
        # Remplacer la dernière couche linéaire du classifier VGG-19
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"Modèle inconnu : '{model_name}'. "
            f"Modèles supportés : {_SUPPORTED_MODELS}"
        )

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(
        f"[Model] '{name}' créé : {n_params:.1f} M paramètres, "
        f"{num_classes} classes de sortie."
    )
    return model


def model_parameter_bytes(model: nn.Module) -> int:
    """Taille totale des paramètres en float32 (non compressés), en octets."""
    return sum(p.numel() for p in model.parameters()) * 4


def list_models() -> tuple:
    """Retourne la liste des noms de modèles disponibles."""
    return _SUPPORTED_MODELS
