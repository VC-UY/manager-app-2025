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
import math
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


# ─── JointSQ ──────────────────────────────────────────────────────────────────
def mckp_greedy(tensor: torch.Tensor, max_bit: int) -> torch.Tensor:
    """
    Algorithme MCKP-Greedy pour allouer le budget de bits sur les composants du tenseur.
    Assigne 0, 2, 4 ou 8 bits à chaque élément.
    """
    device = tensor.device
    numel = tensor.numel()
    if numel == 0:
        return torch.zeros_like(tensor, dtype=torch.uint8)

    squared = tensor.pow(2)

    # Densités de profit incrémentales pour les transitions :
    # 0 -> 2 bits (poids +2) : coeff = 0.46875
    # 2 -> 4 bits (poids +2) : coeff = 0.029296875
    # 4 -> 8 bits (poids +4) : coeff = 0.0009715625
    d1 = 0.46875 * squared
    d2 = 0.029296875 * squared
    d3 = 0.0009715625 * squared

    all_densities = torch.cat([d1, d2, d3])
    all_weights = torch.cat([
        torch.full_like(d1, 2, dtype=torch.int32),
        torch.full_like(d2, 2, dtype=torch.int32),
        torch.full_like(d3, 4, dtype=torch.int32)
    ])
    all_types = torch.cat([
        torch.zeros_like(d1, dtype=torch.uint8),  # 0 -> 2
        torch.ones_like(d2, dtype=torch.uint8),   # 2 -> 4
        torch.full_like(d3, 2, dtype=torch.uint8)  # 4 -> 8
    ])
    orig_indices = torch.cat([
        torch.arange(numel, device=device),
        torch.arange(numel, device=device),
        torch.arange(numel, device=device)
    ])

    # Tri par densité décroissante
    sorted_densities, sorted_idx = torch.sort(all_densities, descending=True)
    sorted_weights = all_weights[sorted_idx]
    sorted_types = all_types[sorted_idx]
    sorted_orig_indices = orig_indices[sorted_idx]

    cum_weights = torch.cumsum(sorted_weights, dim=0)
    idx = torch.searchsorted(cum_weights, max_bit, right=True)
    idx = int(idx.item())

    selected_orig_indices = sorted_orig_indices[:idx]
    selected_types = sorted_types[:idx]

    mask = torch.zeros(numel, dtype=torch.uint8, device=device)
    
    # Assigner les bitwidths correspondants
    mask[selected_orig_indices[selected_types == 0]] = 2
    mask[selected_orig_indices[selected_types == 1]] = 4
    mask[selected_orig_indices[selected_types == 2]] = 8

    return mask


def stochastic_quantize(x: torch.Tensor, n: int) -> Tuple[torch.Tensor, float]:
    """
    Quantifie x stochastiquement sur 2n+1 niveaux dans [-1, 1], multiplié par la norme inf.
    Retourne (q, norm) où q est un tenseur int8 contenant les niveaux quantifiés dans [-n, n].
    """
    x = x.float()
    norm = float(x.abs().max().item())
    if norm == 0.0:
        return torch.zeros_like(x, dtype=torch.int8), 0.0

    sgn = torch.sign(x)
    sgn[sgn == 0] = 1.0
    
    p = x.abs() / norm
    renormalize_p = p * n
    floor_p = torch.floor(renormalize_p)
    compare = torch.rand_like(floor_p)
    final_p = renormalize_p - floor_p
    margin = (compare < final_p).float()
    
    q = sgn * (floor_p + margin)
    q = torch.clamp(q, -128, 127).to(torch.int8)
    return q, norm


def jointsq_compress_model(model: nn.Module, ratio: float = 0.05) -> Tuple[bytes, Dict]:
    """Compresse tous les tenseurs du modèle avec JointSQ (Sparsification/Quantification jointe)."""
    state = model.state_dict()
    arrays: Dict[str, np.ndarray] = {}
    meta: Dict = {"method": "jointsq", "ratio": ratio, "params": {}}

    for name, tensor in state.items():
        if not tensor.is_floating_point():
            np_arr = tensor.detach().cpu().numpy()
            arrays[name] = np_arr
            meta["params"][name] = {
                "shape": list(np_arr.shape),
                "dtype": str(np_arr.dtype),
                "raw": True,
            }
            continue

        clean_tensor = tensor.detach().clone()
        if not torch.isfinite(clean_tensor).all():
            clean_tensor = torch.nan_to_num(clean_tensor, nan=0.0, posinf=0.0, neginf=0.0)

        flat = clean_tensor.view(-1)
        numel = flat.numel()
        
        if numel == 0:
            arrays[name] = np.zeros(0, dtype=np.float32)
            meta["params"][name] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "raw": True,
            }
            continue

        max_bit = math.ceil(numel * 32 * ratio)
        mask = mckp_greedy(flat.abs(), max_bit)
        
        # 2-bit
        mask_2 = (mask == 2)
        idx_2 = torch.nonzero(mask_2).view(-1)
        if idx_2.numel() > 0:
            val_2, norm_2 = stochastic_quantize(flat[mask_2], n=2)
        else:
            val_2 = torch.zeros(0, dtype=torch.int8)
            norm_2 = 0.0
            
        # 4-bit
        mask_4 = (mask == 4)
        idx_4 = torch.nonzero(mask_4).view(-1)
        if idx_4.numel() > 0:
            val_4, norm_4 = stochastic_quantize(flat[mask_4], n=8)
        else:
            val_4 = torch.zeros(0, dtype=torch.int8)
            norm_4 = 0.0
            
        # 8-bit
        mask_8 = (mask == 8)
        idx_8 = torch.nonzero(mask_8).view(-1)
        if idx_8.numel() > 0:
            val_8, norm_8 = stochastic_quantize(flat[mask_8], n=128)
        else:
            val_8 = torch.zeros(0, dtype=torch.int8)
            norm_8 = 0.0

        arrays[f"{name}__idx2"] = idx_2.cpu().numpy().astype(np.int64)
        arrays[f"{name}__val2"] = val_2.cpu().numpy().astype(np.int8)
        arrays[f"{name}__idx4"] = idx_4.cpu().numpy().astype(np.int64)
        arrays[f"{name}__val4"] = val_4.cpu().numpy().astype(np.int8)
        arrays[f"{name}__idx8"] = idx_8.cpu().numpy().astype(np.int64)
        arrays[f"{name}__val8"] = val_8.cpu().numpy().astype(np.int8)
        
        meta["params"][name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "raw": False,
            "norm2": float(norm_2),
            "norm4": float(norm_4),
            "norm8": float(norm_8)
        }

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue(), meta


def jointsq_decompress_model(model: nn.Module, data: bytes, meta: Dict) -> None:
    """Décompresse et charge les paramètres du modèle compressés par JointSQ."""
    buf = io.BytesIO(data)
    arrays = np.load(buf, allow_pickle=False)
    state: Dict[str, torch.Tensor] = {}

    device = next(model.parameters()).device

    for name, p in meta["params"].items():
        if p.get("raw", False):
            state[name] = torch.tensor(arrays[name], device=device).view(p["shape"])
            continue

        shape = p["shape"]
        numel = int(np.prod(shape))
        flat = torch.zeros(numel, dtype=torch.float32, device=device)

        # 2-bit
        idx_2 = torch.tensor(arrays[f"{name}__idx2"], dtype=torch.long, device=device)
        val_2 = torch.tensor(arrays[f"{name}__val2"], dtype=torch.float32, device=device)
        norm_2 = p["norm2"]
        if idx_2.numel() > 0:
            flat[idx_2] = norm_2 * val_2 / 2.0

        # 4-bit
        idx_4 = torch.tensor(arrays[f"{name}__idx4"], dtype=torch.long, device=device)
        val_4 = torch.tensor(arrays[f"{name}__val4"], dtype=torch.float32, device=device)
        norm_4 = p["norm4"]
        if idx_4.numel() > 0:
            flat[idx_4] = norm_4 * val_4 / 8.0

        # 8-bit
        idx_8 = torch.tensor(arrays[f"{name}__idx8"], dtype=torch.long, device=device)
        val_8 = torch.tensor(arrays[f"{name}__val8"], dtype=torch.float32, device=device)
        norm_8 = p["norm8"]
        if idx_8.numel() > 0:
            flat[idx_8] = norm_8 * val_8 / 128.0

        if not torch.isfinite(flat).all():
            logging.warning(f"[compression] NaN/Inf résiduels dans '{name}' (jointsq), remplacement par 0.")
            flat = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)

        state[name] = flat.view(shape)

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
    elif method == "jointsq":
        return jointsq_compress_model(model, ratio)
    else:
        return pack_model(model)


def decompress_model(model: nn.Module, data: bytes, meta: Dict) -> None:
    """Décompresse et charge les paramètres dans le modèle."""
    method = meta.get("method", "none")
    if method == "quantization":
        dequantize_model(model, data, meta)
    elif method == "sparsification":
        desparsify_model(model, data, meta)
    elif method == "jointsq":
        jointsq_decompress_model(model, data, meta)
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
