#!/usr/bin/env python3
"""Entrainement reel d'un modele sur un shard CIFAR (PyTorch CPU)."""

import json
import os
import pickle
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class SimpleNet(nn.Module):
    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.fc(x)


def find_data_file() -> Path:
    candidates = [
        Path("/input/data.pkl"),
        Path("/input/shard_0/data.pkl"),
        Path("input/data.pkl"),
        Path("data.pkl"),
    ]
    if Path("/input").exists():
        candidates.extend(Path("/input").rglob("data.pkl"))
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Aucun data.pkl trouve dans /input")


def main():
    data_file = find_data_file()
    output_dir = Path(os.environ.get("OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(data_file, "rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, (list, tuple)) and len(payload) == 2:
        data, labels = payload
    else:
        data = payload["data"]
        labels = payload["labels"]

    data = torch.tensor(data, dtype=torch.float32)
    if data.ndim == 4 and data.shape[-1] == 3:
        data = data.permute(0, 3, 1, 2)
    data = data / 255.0
    labels = torch.tensor(labels, dtype=torch.long)

    num_classes = int(labels.max().item()) + 1
    loader = DataLoader(TensorDataset(data, labels), batch_size=32, shuffle=True)

    model = SimpleNet(num_classes=num_classes)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = int(os.environ.get("TRAIN_EPOCHS", "2"))
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            opt.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        preds = model(data).argmax(1)
        accuracy = (preds == labels).float().mean().item()

    # Format compatible manager (sans torch/numpy dans le worker): listes Python
    weights = {key: value.detach().cpu().tolist() for key, value in model.state_dict().items()}
    model_path = output_dir / "model.pt"
    metrics_path = output_dir / "metrics.json"
    with open(model_path, "wb") as handle:
        pickle.dump({"format": "numpy_state_dict", "weights": weights}, handle)
    metrics_path.write_text(json.dumps({
        "accuracy": accuracy,
        "samples": int(data.shape[0]),
        "classes": num_classes,
        "epochs": epochs,
    }))
    print(f"OK train samples={data.shape[0]} acc={accuracy:.4f} -> {model_path}")


if __name__ == "__main__":
    main()
