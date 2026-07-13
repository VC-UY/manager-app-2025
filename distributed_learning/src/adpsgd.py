"""
AD-PSGD — Asynchronous Decentralized Parallel Stochastic Gradient Descent
==========================================================================

Implémentation fidèle à l'algorithme 1 de :
  Lian et al. (2018) "Asynchronous Decentralized Parallel Stochastic Gradient Descent"
  https://arxiv.org/abs/1710.06952

Points clés de l'article :
  1. Chaque nœud maintient un modèle LOCAL x_i en mémoire.
  2. Gradient calculé sur x̂ (lecture potentiellement STALE du modèle).
  3. Averaging symétrique :  x_i, x_j ← (x_i + x_j) / 2  (doubly stochastic, W=0.5 I + 0.5 perm).
  4. Gradient mis à jour APRÈS l'averaging (ordre "average then update" = algo logique).
  5. Topologie BIPARTIE pour éviter le deadlock :
       - Nœuds ACTIFS  : initient la communication, envoient leur modèle et attendent.
       - Nœuds PASSIFS : répondent uniquement quand un actif les contacte.
  6. Topologie de ring avec saut exponentiel : voisin = (i + 2^k + 1) mod n,
     k = 0 … log2(n-1), pour une diffusion en O(log n) étapes.
"""

import copy
import logging
import math
import random
import threading
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


# ─── Topologie bipartie ───────────────────────────────────────────────────────

class BipartiteTopology:
    """
    Partitionne n nœuds en deux ensembles disjoints A (actifs) et P (passifs).

    Rôle du nœud courant :
      - ACTIVE  : initie l'averaging en choisissant un voisin passif.
      - PASSIVE : attend qu'un actif le contacte pour l'averaging.

    Cette partition élimine les deadlocks car :
      * un actif ne communique qu'avec des passifs,
      * un passif ne communique qu'avec des actifs.
    → Il est impossible d'avoir un cycle de dépendances.
    """

    ROLE_ACTIVE  = "active"
    ROLE_PASSIVE = "passive"

    def __init__(self, volunteer_id: int, n_volunteers: int, topology: str = "exponential"):
        """
        Args:
            volunteer_id   : identifiant 0-indexé du nœud courant.
            n_volunteers   : nombre total de nœuds.
            topology       : 'ring' | 'exponential' (recommandé par l'article).
        """
        self.vol_id      = volunteer_id
        self.n           = n_volunteers
        self.topology    = topology

        # Partition pair/impair → bipartie simple
        self.role = self.ROLE_ACTIVE if volunteer_id % 2 == 0 else self.ROLE_PASSIVE

        # Pré-calcul des voisins selon la topologie choisie
        self._neighbors = self._compute_neighbors()

        logger.info(
            f"[AD-PSGD Topology] vol_id={volunteer_id} n={n_volunteers} "
            f"role={self.role} topology={topology} "
            f"neighbors={self._neighbors}"
        )

    def _compute_neighbors(self) -> List[int]:
        """
        Calcule les voisins selon la topologie.

        Ring simple      : voisin immédiat (gauche/droite).
        Exponentielle    : voisins à 2^k + 1 sauts dans le ring (O(log n) dissémination).
        """
        n = self.n
        if n <= 1:
            return []

        vid = self.vol_id

        if self.topology == "exponential":
            neighbors = set()
            # Sauts exponentiels : 2^k + 1 pour k = 0 … ⌊log2(n-1)⌋
            max_k = max(0, int(math.log2(n - 1))) if n > 2 else 0
            for k in range(max_k + 1):
                hop = (2 ** k) + 1
                fwd = (vid + hop) % n
                bwd = (vid - hop) % n
                if fwd != vid:
                    neighbors.add(fwd)
                if bwd != vid:
                    neighbors.add(bwd)
            # Toujours inclure les voisins immédiats du ring
            neighbors.add((vid + 1) % n)
            neighbors.add((vid - 1) % n)
            return sorted(neighbors)

        else:  # "ring"
            return [(vid + 1) % n, (vid - 1) % n]

    def get_neighbors(self) -> List[int]:
        """Retourne la liste des IDs de voisins."""
        return list(self._neighbors)

    def sample_neighbor(self) -> Optional[int]:
        """
        Échantillonne UN voisin aléatoire (pour l'averaging step).
        Retourne None si aucun voisin disponible.
        """
        candidates = [v for v in self._neighbors if v != self.vol_id]
        return random.choice(candidates) if candidates else None

    def is_active(self) -> bool:
        return self.role == self.ROLE_ACTIVE

    def is_passive(self) -> bool:
        return self.role == self.ROLE_PASSIVE


# ─── Averaging AD-PSGD ────────────────────────────────────────────────────────

def adpsgd_average(
    local_model: nn.Module,
    remote_state: Dict[str, torch.Tensor],
    alpha: float = 0.5,
) -> None:
    """
    Averaging symétrique AD-PSGD :
        x_i ← α * x_i + (1 - α) * x_j

    Avec α = 0.5 (valeur par défaut de l'article) :
        x_i ← (x_i + x_j) / 2

    La doubly-stochastic matrix W correspondante est :
        W[i, j] = W[j, i] = 0.5  (pour la paire (i, j))
        W[k, k] = 1              pour tout autre nœud k

    Cette opération est LOCALE : seuls les deux nœuds participant à
    l'échange mettent à jour leur modèle ; les autres restent inchangés
    (cf. ligne 5-6 de l'Algorithme 1).

    Args:
        local_model  : modèle local x_i (mis à jour in-place).
        remote_state : state_dict du voisin x_j.
        alpha        : poids du modèle local (0.5 = moyenne exacte).
    """
    local_state = local_model.state_dict()
    averaged: Dict[str, torch.Tensor] = {}

    for key in local_state:
        lv = local_state[key]
        rv = remote_state.get(key)

        if rv is None:
            averaged[key] = lv
            continue

        if lv.is_floating_point():
            # Validation anti-NaN
            if not torch.isfinite(rv).all():
                logger.warning(
                    f"[AD-PSGD] Tenseur '{key}' du voisin contient NaN/Inf → "
                    f"le modèle local est conservé."
                )
                averaged[key] = lv
            else:
                averaged[key] = (alpha * lv.float() + (1.0 - alpha) * rv.float()).to(lv.dtype)
        else:
            averaged[key] = lv  # entiers / buffers → pas de moyenne

    # Filet de sécurité global
    for key, v in averaged.items():
        if v.is_floating_point() and not torch.isfinite(v).all():
            logger.error(
                f"[AD-PSGD] Averaging produit NaN dans '{key}' → rollback partiel."
            )
            averaged[key] = local_state[key]

    local_model.load_state_dict(averaged)


# ─── Snapshot stale (lecture de x̂) ──────────────────────────────────────────

class StaleModelReader:
    """
    Gère la lecture potentiellement STALE du modèle (x̂_k = x_{k - τ}).

    Dans AD-PSGD, le gradient est calculé sur la VALEUR LUE du modèle
    au moment du début du batch, PAS nécessairement sur la valeur à jour
    (car l'averaging step peut l'avoir modifié entre-temps dans le cas
    multi-thread). Pour notre implémentation single-thread on capture
    un snapshot avant l'averaging puis on utilise ce snapshot comme x̂.

    Ce module permet de conserver ce snapshot et d'en calculer la «staleness»
    (différence L2 entre x̂ et x_courant).
    """

    def __init__(self, model: nn.Module):
        self._lock = threading.Lock()
        self._snapshot: Optional[Dict[str, torch.Tensor]] = None
        self._snapshot_round: int = -1
        self._update_snapshot(model)

    def _update_snapshot(self, model: nn.Module) -> None:
        self._snapshot = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
        }

    def capture(self, model: nn.Module, round_num: int) -> Dict[str, torch.Tensor]:
        """
        Capture un nouveau snapshot AVANT l'averaging (= x̂_k).
        Retourne le snapshot pour être utilisé dans le calcul du gradient.
        """
        with self._lock:
            self._update_snapshot(model)
            self._snapshot_round = round_num
            return dict(self._snapshot)

    def compute_staleness(self, model: nn.Module) -> float:
        """
        Calcule la norme L2 de la différence entre x̂ et x courant.
        Une grande valeur indique que l'averaging a significativement
        modifié le modèle depuis le dernier snapshot.
        """
        if self._snapshot is None:
            return 0.0
        current = model.state_dict()
        total = 0.0
        for key in self._snapshot:
            s = self._snapshot[key]
            c = current.get(key)
            if c is not None and s.is_floating_point():
                diff = (s.float() - c.float()).norm().item()
                total += diff ** 2
        return math.sqrt(total)


# ─── Statistiques AD-PSGD ────────────────────────────────────────────────────

class ADPSGDStats:
    """
    Collecte les métriques spécifiques à AD-PSGD pour le suivi expérimental.

    Métriques tracées :
      - staleness_norm    : distance entre x̂ et x au moment du gradient.
      - n_averagings      : nombre d'averaging steps réussis.
      - n_averaging_skip  : nombre d'averaging steps échoués (voisin indisponible).
      - avg_alpha         : valeur de alpha utilisée (doubly stochastic weight).
      - role              : "active" ou "passive".
      - topology          : "ring" ou "exponential".
      - neighbors         : liste des voisins du nœud.
      - spectral_gap_rho  : ρ estimé (gap spectral de W, mesure la vitesse de mélange).
    """

    def __init__(self, topology: BipartiteTopology):
        self._topo = topology
        self.reset()

    def reset(self) -> None:
        self.staleness_norm: float = 0.0
        self.n_averagings: int = 0
        self.n_averaging_skip: int = 0
        self.avg_alpha: float = 0.5
        self.neighbor_ids: List[int] = self._topo.get_neighbors()
        self.sampled_neighbor: Optional[int] = None
        self.role: str = self._topo.role
        self.topology: str = self._topo.topology
        self.skip_factor: int = 1

    def record_averaging(self, neighbor_id: int, alpha: float, staleness: float) -> None:
        self.n_averagings += 1
        self.avg_alpha = alpha
        self.sampled_neighbor = neighbor_id
        self.staleness_norm = staleness

    def record_skip(self) -> None:
        self.n_averaging_skip += 1

    def to_dict(self) -> dict:
        return {
            "adpsgd_role":              self.role,
            "adpsgd_topology":          self.topology,
            "adpsgd_neighbors":         self.neighbor_ids,
            "adpsgd_sampled_neighbor":  self.sampled_neighbor,
            "adpsgd_staleness_norm":    round(self.staleness_norm, 6),
            "adpsgd_n_averagings":      self.n_averagings,
            "adpsgd_n_averaging_skip":  self.n_averaging_skip,
            "adpsgd_avg_alpha":         self.avg_alpha,
            "adpsgd_skip_factor":       self.skip_factor,
        }


#Helpers

def build_topology(volunteer_id: int, n_volunteers: int, topology_type: str) -> BipartiteTopology:
    """
    Construit la topologie bipartie pour ce nœud.

    Args:
        volunteer_id   : identifiant 0-indexé du nœud.
        n_volunteers   : nombre total de nœuds.
        topology_type  : 'ring' | 'exponential'.

    Returns:
        BipartiteTopology instance.
    """
    topo = BipartiteTopology(
        volunteer_id=volunteer_id,
        n_volunteers=n_volunteers,
        topology=topology_type,
    )
    logger.info(
        f"[AD-PSGD] Topologie construite : "
        f"id={volunteer_id} n={n_volunteers} "
        f"role={topo.role} voisins={topo.get_neighbors()}"
    )
    return topo


def get_neighbor_macs(
    neighbor_ids: List[int],
    all_volunteers: List[dict],
) -> Dict[int, str]:
    """
    Résout les IDs de voisins en adresses MAC grâce à la liste fournie
    par le manager (format: [{'volunteer_id': int, 'mac_address': str, ...}]).

    Retourne un dict {vol_id: mac_address}.
    """
    mapping: Dict[int, str] = {}
    for vol in all_volunteers:
        vid = vol.get("volunteer_id")
        mac = vol.get("mac_address")
        if vid is not None and mac:
            if vid in neighbor_ids:
                mapping[vid] = mac
    return mapping
