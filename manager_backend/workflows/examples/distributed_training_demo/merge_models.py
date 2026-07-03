"""Fusion de checkpoints sans dependance a PyTorch (numpy/pickle)."""

import os
import pickle
from glob import glob

import numpy as np


def _load_weights(path):
    with open(path, "rb") as handle:
        payload = pickle.load(handle)

    # Format numpy/pickle produit par train_on_shard
    if isinstance(payload, dict) and payload.get("format") == "numpy_state_dict":
        return payload["weights"]

    # Ancien format torch state_dict (si torch est disponible)
    if isinstance(payload, dict):
        weights = {}
        for key, value in payload.items():
            if hasattr(value, "detach"):
                weights[key] = value.detach().cpu().numpy()
            else:
                weights[key] = np.asarray(value)
        return weights

    raise ValueError(f"Format de modele non reconnu: {path}")


def merge_models(input_path, output_path):
    """
    Fusionne plusieurs checkpoints (moyenne des poids).
    Cherche recursivement tous les fichiers model.pt sous input_path.
    """
    patterns = [
        os.path.join(input_path, "**/model.pt"),
        os.path.join(input_path, "*/model.pt"),
        os.path.join(input_path, "model.pt"),
    ]
    model_files = []
    for pattern in patterns:
        model_files.extend(glob(pattern, recursive=True))
    model_files = sorted(set(model_files))

    if not model_files:
        raise FileNotFoundError(f"Aucun model.pt trouve sous {input_path}")

    models = [_load_weights(path) for path in model_files]
    keys = models[0].keys()
    avg_model = {}
    for key in keys:
        stacked = np.stack([np.asarray(model[key], dtype=np.float64) for model in models], axis=0)
        avg_model[key] = (stacked.mean(axis=0)).astype(np.float32)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as handle:
        pickle.dump({"format": "numpy_state_dict", "weights": avg_model}, handle)

    return {
        "models_merged": len(model_files),
        "output_path": output_path,
        "sources": model_files,
    }
