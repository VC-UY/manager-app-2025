"""
Compression frugale des paramètres de modèle pour réduire l'empreinte bande passante.

Méthodes disponibles :
  - 'quantization'   : quantification dynamique int8 (float32 → int8 + scale/zero-point)
  - 'sparsification' : top-k par magnitude (seulement les k% valeurs les plus grandes)
  - 'none'           : sérialisation npz compressée sans perte
"""
import io
import logging
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


# ─── Quantification ───────────────────────────────────────────────────────────

def _quantize(data: np.ndarray, bits: int = 8) -> Tuple[np.ndarray, float, float]:
    """Mappe data dans [-2^(b-1), 2^(b-1)-1], retourne (quantized, scale, zp)."""
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    dmin, dmax = float(data.min()), float(data.max())
    if dmin == dmax:
        return np.zeros_like(data, dtype=np.int8), 1.0, 0.0
    scale = (dmax - dmin) / (qmax - qmin)
    zero_point = qmin - dmin / scale
    q = np.clip(np.round(data / scale + zero_point), qmin, qmax).astype(np.int8)
    return q, scale, float(zero_point)


def _dequantize(q: np.ndarray, scale: float, zero_point: float) -> np.ndarray:
    return (q.astype(np.float32) - zero_point) * scale


def quantize_model(model: nn.Module,
                   bits: int = 8) -> Tuple[bytes, Dict]:
    """Quantifie tous les tenseurs du modèle en int8."""
    state = model.state_dict()
    arrays, meta = {}, {"method": "quantization", "bits": bits, "params": {}}

    for name, tensor in state.items():
        data = tensor.cpu().float().numpy()
        q, scale, zp = _quantize(data, bits)
        arrays[name] = q
        meta["params"][name] = {
            "shape": list(data.shape),
            "scale": scale,
            "zero_point": zp,
        }

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue(), meta


def dequantize_model(model: nn.Module,
                     data: bytes,
                     meta: Dict) -> None:
    """Charge les tenseurs dequantifiés dans le modèle."""
    buf = io.BytesIO(data)
    arrays = np.load(buf, allow_pickle=False)
    state = {}
    for name, p in meta["params"].items():
        q = arrays[name]
        reconstructed = _dequantize(q, p["scale"], p["zero_point"])
        state[name] = torch.tensor(reconstructed.reshape(p["shape"]))
    model.load_state_dict(state)


# ─── Sparsification top-k ─────────────────────────────────────────────────────

def sparsify_model(model: nn.Module,
                   ratio: float = 0.05) -> Tuple[bytes, Dict]:
    """
    Garde uniquement les top-ratio% paramètres par valeur absolue.
    Les autres sont mis à zéro (envoi uniquement des indices + valeurs).
    """
    state = model.state_dict()
    arrays, meta = {}, {"method": "sparsification", "ratio": ratio, "params": {}}

    for name, tensor in state.items():
        data = tensor.cpu().float().numpy().ravel()
        n_keep = max(1, int(len(data) * ratio))
        # top-k par magnitude
        idx = np.argpartition(np.abs(data), -n_keep)[-n_keep:]
        vals = data[idx]
        arrays[f"{name}__idx"] = idx.astype(np.int32)
        arrays[f"{name}__val"] = vals.astype(np.float32)
        meta["params"][name] = {
            "shape":   list(tensor.shape),
            "n_total": int(len(data)),
            "n_keep":  n_keep,
        }

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue(), meta


def desparsify_model(model: nn.Module,
                     data: bytes,
                     meta: Dict) -> None:
    """Reconstruit le modèle depuis sa version sparse."""
    buf = io.BytesIO(data)
    arrays = np.load(buf, allow_pickle=False)
    state = {}
    for name, p in meta["params"].items():
        dense = np.zeros(p["n_total"], dtype=np.float32)
        idx  = arrays[f"{name}__idx"]
        vals = arrays[f"{name}__val"]
        dense[idx] = vals
        state[name] = torch.tensor(dense.reshape(p["shape"]))
    model.load_state_dict(state)


# ─── Sans compression ─────────────────────────────────────────────────────────

def pack_model(model: nn.Module) -> Tuple[bytes, Dict]:
    """Sérialise le modèle en npz compressé sans modification."""
    state = model.state_dict()
    arrays = {k: v.cpu().numpy() for k, v in state.items()}
    meta = {
        "method": "none",
        "params": {k: {"shape": list(v.shape)} for k, v in state.items()},
    }
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue(), meta


def unpack_model(model: nn.Module, data: bytes, meta: Dict) -> None:
    buf = io.BytesIO(data)
    arrays = np.load(buf, allow_pickle=False)
    state = {k: torch.tensor(arrays[k]) for k in arrays.files}
    model.load_state_dict(state)


# ─── API publique ─────────────────────────────────────────────────────────────

def compress_model(model: nn.Module,
                   method: str = "quantization",
                   bits: int = 8,
                   ratio: float = 0.05) -> Tuple[bytes, Dict]:
    """Compresse le modèle selon la méthode choisie."""
    if method == "quantization":
        return quantize_model(model, bits)
    elif method == "sparsification":
        return sparsify_model(model, ratio)
    else:
        return pack_model(model)


def decompress_model(model: nn.Module,
                     data: bytes,
                     meta: Dict) -> None:
    """Décompresse et charge les paramètres dans le modèle."""
    method = meta.get("method", "none")
    if method == "quantization":
        dequantize_model(model, data, meta)
    elif method == "sparsification":
        desparsify_model(model, data, meta)
    else:
        unpack_model(model, data, meta)


def average_models(model: nn.Module,
                   received_states: list) -> None:
    """
    FedAvg : moyenne du modèle local avec les modèles reçus par gossip.
    Pondération uniforme (1 / (1 + n_received)).
    """
    if not received_states:
        return
    local = model.state_dict()
    all_states = [local] + received_states
    n = len(all_states)
    averaged = {}
    for key in local:
        if local[key].is_floating_point():
            stacked = torch.stack([s[key].float() for s in all_states])
            averaged[key] = stacked.mean(0).to(local[key].dtype)
        else:
            averaged[key] = local[key]
    model.load_state_dict(averaged)
    logging.debug(f"Agrégation de {n} modèles effectuée")


def compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
    """Ratio de compression (original / compressé). > 1 = gain."""
    if compressed_bytes == 0:
        return 1.0
    return original_bytes / compressed_bytes
