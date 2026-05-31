"""
Définitions des modèles légers utilisés par les volontaires.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightCNN_MNIST(nn.Module):
    """CNN léger pour MNIST (1×28×28 → 10 classes)."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(32 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, num_classes)
        self.drop  = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 16×14×14
        x = self.pool(F.relu(self.conv2(x)))   # 32×7×7
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


class LightCNN_CIFAR(nn.Module):
    """CNN léger pour CIFAR-10 (3×32×32 → 10 classes)."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 4 * 4, 256)
        self.fc2   = nn.Linear(256, num_classes)
        self.drop  = nn.Dropout(0.4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 32×16×16
        x = self.pool(F.relu(self.conv2(x)))   # 64×8×8
        x = self.pool(F.relu(self.conv3(x)))   # 64×4×4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


def create_model(dataset: str = "mnist", num_classes: int = 10) -> nn.Module:
    if dataset == "mnist":
        return LightCNN_MNIST(num_classes)
    elif dataset == "cifar10":
        return LightCNN_CIFAR(num_classes)
    raise ValueError(f"Dataset inconnu : {dataset}")


def model_parameter_bytes(model: nn.Module) -> int:
    """Taille totale des paramètres en float32 (non compressés)."""
    return sum(p.numel() for p in model.parameters()) * 4
