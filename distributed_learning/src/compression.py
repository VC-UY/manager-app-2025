"""
Compression frugale des paramètres de modèle pour réduire l'empreinte bande passante.

Méthodes disponibles :
- 'quantization'   : quantification dynamique int8 (float32 -> int8 + scale/zero-point)
- 'sparsification' : top-k par magnitude (seulement les k% valeurs les plus grandes)
- 'none'           : sérialisation npz compressée sans perte

>>> CORRECTIONS APPORTEES (anti-NaN / anti-divergence) <<<
1. _quantize         : nettoyage NaN/Inf AVANT calcul min/max (sinon scale=NaN -> tout le modèle empoisonné).
2. _quantize         : si dmin==dmax, on conserve la VRAIE valeur constante au lieu de la mettre à 0
                       (préserve les biais convergés).
3. dequantize_model  : validation post-reconstruction (remplacement des NaN/Inf résiduels par 0).
4. desparsify_model  : idem.
5. average_models    : ignore les modèles reçus contenant NaN/Inf (filtrage anti-contamination)
                       et pondère davantage le modèle local (alpha=0.5) pour limiter la dilution
                       par le bruit de quantification.
6. _safe_finite      : helper pour valider rapidement tout state_dict.
"""

import io
import logging
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn


# ─── Helpers de robustesse ────────────────────────────────────────────────────
def _sanitize(data: np.ndarray) -> np.ndarray:
    """Remplace les NaN et +/-Inf par 0 pour éviter de polluer min/max/scale."""
    if not np.isfinite(data).all():
        logging.warning("[compression] Tenseur contenant des NaN/Inf détecté, nettoyage automatique.")
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data


def _safe_finite(state: Dict[str, torch.Tensor]) -> bool:
    """Retourne True si TOUS les tenseurs du state_dict sont finis."""
    for k, v in state.items():
        if v.is_floating_point() and not torch.isfinite(v).all():
            return False
    return True


# ─── Quantification ───────────────────────────────────────────────────────────
def _quantize(data: np.ndarray, bits: int = 8) -> Tuple[np.ndarray, float, float]:
    """Mappe data dans [-2^(b-1), 2^(b-1)-1], retourne (quantized, scale, zp).

    Robuste aux NaN/Inf et au cas constant.
    """
    data = _sanitize(data)

    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    dmin, dmax = float(data.min()), float(data.max())
    # Assurer que 0.0 est inclus dans l'intervalle pour éviter un zero-point géant
    # qui causerait une perte de précision dramatique sur les float32.
    dmin = min(0.0, dmin)
    dmax = max(0.0, dmax)

    # Cas constant : on stocke la constante dans scale, zp=0 -> reconstruction exacte.
    if dmin == dmax:
        return np.zeros_like(data, dtype=np.int8), 0.0, float(dmin)

    scale = (dmax - dmin) / (qmax - qmin)
    zero_point = qmin - dmin / scale
    q = np.clip(np.round(data / scale + zero_point), qmin, qmax).astype(np.int8)
    return q, scale, float(zero_point)


def _dequantize(q: np.ndarray, scale: float, zero_point: float) -> np.ndarray:
    """Reconstruit float32 depuis int8 + (scale, zero_point).

    Gestion du cas constant : si scale==0, la constante est stockée dans zero_point.
    """
    if scale == 0.0:
        # Cas constant -> on restitue directement la valeur stockée dans zero_point.
        return np.full(q.shape, zero_point, dtype=np.float32)
    return (q.astype(np.float32) - zero_point) * scale


def quantize_model(model: nn.Module, bits: int = 8) -> Tuple[bytes, Dict]:
    """Quantifie tous les tenseurs (floats) du modèle en int8."""
    state = model.state_dict()
    arrays: Dict[str, np.ndarray] = {}
    meta: Dict = {"method": "quantization", "bits": bits, "params": {}}

    for name, tensor in state.items():
        np_arr = tensor.detach().cpu().numpy()
        if not np.issubdtype(np_arr.dtype, np.floating):
            # On stocke tel quel (ex. buffers int, BN num_batches_tracked).
            arrays[name] = np_arr
            meta["params"][name] = {
                "shape": list(np_arr.shape),
                "dtype": str(np_arr.dtype),
                "raw": True,
            }
            continue

        q, scale, zp = _quantize(np_arr, bits)
        arrays[name] = q
        meta["params"][name] = {
            "shape": list(np_arr.shape),
            "dtype": str(np_arr.dtype),
            "scale": scale,
            "zero_point": zp,
            "raw": False,
        }

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue(), meta


def dequantize_model(model: nn.Module, data: bytes, meta: Dict) -> None:
    """Charge les tenseurs déquantifiés dans le modèle (avec validation finale)."""
    buf = io.BytesIO(data)
    arrays = np.load(buf, allow_pickle=False)
    state: Dict[str, torch.Tensor] = {}

    for name, p in meta["params"].items():
        arr = arrays[name]
        if p.get("raw", False):
            state[name] = torch.tensor(arr.reshape(p["shape"]))
            continue

        reconstructed = _dequantize(arr, p["scale"], p["zero_point"])
        # Filet de sécurité final.
        if not np.isfinite(reconstructed).all():
            logging.warning(f"[compression] NaN/Inf résiduels dans '{name}' après déquantification, remplacement par 0.")
            reconstructed = np.nan_to_num(reconstructed, nan=0.0, posinf=0.0, neginf=0.0)
        state[name] = torch.tensor(reconstructed.reshape(p["shape"]))

    model.load_state_dict(state)


# ─── Sparsification top-k ─────────────────────────────────────────────────────
def sparsify_model(model: nn.Module, ratio: float = 0.05) -> Tuple[bytes, Dict]:
    """Garde uniquement les top-`ratio` (%) paramètres par valeur absolue.

    Les autres sont mis à zéro (envoi uniquement des indices + valeurs).
    """
    state = model.state_dict()
    arrays: Dict[str, np.ndarray] = {}
    meta: Dict = {"method": "sparsification", "ratio": ratio, "params": {}}

    for name, tensor in state.items():
        np_arr = tensor.detach().cpu().numpy()
        if not np.issubdtype(np_arr.dtype, np.floating):
            arrays[name] = np_arr
            meta["params"][name] = {
                "shape": list(np_arr.shape),
                "n_total": int(np_arr.size),
                "raw": True,
            }
            continue

        np_arr = _sanitize(np_arr)
        flat = np_arr.flatten()
        n_total = flat.size
        k = max(1, int(n_total * ratio))

        idx = np.argpartition(np.abs(flat), -k)[-k:]
        vals = flat[idx].astype(np.float32)

        arrays[f"{name}__idx"] = idx.astype(np.int64)
        arrays[f"{name}__val"] = vals
        meta["params"][name] = {
            "shape": list(np_arr.shape),
            "n_total": int(n_total),
            "raw": False,
        }

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue(), meta


def desparsify_model(model: nn.Module, data: bytes, meta: Dict) -> None:
    """Reconstruit le modèle depuis sa version sparse (avec validation)."""
    buf = io.BytesIO(data)
    arrays = np.load(buf, allow_pickle=False)
    state: Dict[str, torch.Tensor] = {}

    for name, p in meta["params"].items():
        if p.get("raw", False):
            state[name] = torch.tensor(arrays[name].reshape(p["shape"]))
            continue

        dense = np.zeros(p["n_total"], dtype=np.float32)
        idx = arrays[f"{name}__idx"]
        vals = arrays[f"{name}__val"]
        dense[idx] = vals

        if not np.isfinite(dense).all():
            logging.warning(f"[compression] NaN/Inf résiduels dans '{name}' (sparse), remplacement par 0.")
            dense = np.nan_to_num(dense, nan=0.0, posinf=0.0, neginf=0.0)
        state[name] = torch.tensor(dense.reshape(p["shape"]))

    model.load_state_dict(state)


# ─── Sans compression ─────────────────────────────────────────────────────────
def pack_model(model: nn.Module) -> Tuple[bytes, Dict]:
    """Sérialise le modèle en npz compressé sans modification."""
    state = model.state_dict()
    arrays = {k: v.detach().cpu().numpy() for k, v in state.items()}
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
    # Validation finale
    for k, v in state.items():
        if v.is_floating_point() and not torch.isfinite(v).all():
            logging.warning(f"[compression] NaN/Inf dans '{k}' (pack), remplacement par 0.")
            state[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
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


def decompress_model(model: nn.Module, data: bytes, meta: Dict) -> None:
    """Décompresse et charge les paramètres dans le modèle."""
    method = meta.get("method", "none")
    if method == "quantization":
        dequantize_model(model, data, meta)
    elif method == "sparsification":
        desparsify_model(model, data, meta)
    else:
        unpack_model(model, data, meta)


def average_models(model: nn.Module,
                   received_states: List[Dict[str, torch.Tensor]],
                   local_weight: float = 0.5) -> None:
    """FedAvg robuste : moyenne du modèle local avec les modèles reçus par gossip.

    >>> Corrections :
    - Filtre les states contenant des NaN/Inf (anti-contamination virale).
    - Pondération paramétrable : `local_weight` pour le modèle local, le reste
      réparti uniformément entre les voisins (limite la dilution par bruit de
      quantification).
    """
    if not received_states:
        return

    # Filtre anti-NaN : on rejette tout modèle reçu corrompu.
    clean = []
    for i, s in enumerate(received_states):
        if _safe_finite(s):
            clean.append(s)
        else:
            logging.warning(f"[compression] Modèle reçu #{i} contient NaN/Inf -> ignoré dans l'agrégation.")

    if not clean:
        logging.warning("[compression] Tous les modèles reçus sont corrompus -> on garde le modèle local.")
        return

    local = model.state_dict()
    n_recv = len(clean)
    w_local = float(local_weight)
    w_recv = (1.0 - w_local) / n_recv

    averaged: Dict[str, torch.Tensor] = {}
    for key in local:
        if local[key].is_floating_point():
            acc = local[key].float() * w_local
            for s in clean:
                acc = acc + s[key].float() * w_recv
            averaged[key] = acc.to(local[key].dtype)
        else:
            averaged[key] = local[key]

    # Filet de sécurité ultime : si malgré tout on a un NaN, on garde le local.
    if not _safe_finite(averaged):
        logging.error("[compression] Agrégation produit du NaN -> rollback sur le modèle local.")
        return

    model.load_state_dict(averaged)
    logging.debug(f"[compression] Agrégation FedAvg : 1 local (w={w_local:.2f}) + {n_recv} voisins (w={w_recv:.3f} chacun).")


def compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
    """Ratio de compression (original / compressé). > 1 = gain."""
    if compressed_bytes == 0:
        return 1.0
    return original_bytes / compressed_bytes
