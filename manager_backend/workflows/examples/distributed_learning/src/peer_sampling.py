"""
Peer Sampling Adaptatif avec Sliding-Window UCB (SW-UCB).

ARCHITECTURE DÉCENTRALISÉE : chaque volontaire instancie SON propre
SWUCBSelector. Les récompenses reflètent SA vision du réseau.

Formule SW-UCB (Garivier & Moulines, 2008) :
    UCB_i = r̄_i(t, τ) + sqrt( ξ * ln(min(t, τ)) / N_i(t, τ) )

Récompense composite (vitesse de bande passante) :
    reward = 0.6 * bw_normalisée + 0.3 * succès + 0.1 * (1 - latence_normalisée)

Cold start : tout voisin jamais tiré -> UCB = +∞.
"""
import logging
import math
import random
import threading
from collections import defaultdict, deque
from typing import Dict, List, Optional


class SWUCBSelector:
    """Sélecteur de voisins adaptatif (un par volontaire). Thread-safe."""

    BW_MAX_MBPS = 1000.0
    LATENCY_MAX_S = 30.0

    W_BW = 0.6
    W_SUCCESS = 0.3
    W_LATENCY = 0.1

    def __init__(self, window: int = 5, confidence: float = 0.95,
                 my_mac: Optional[str] = None):
        if not 0.5 < confidence < 1.0:
            raise ValueError(f"confidence doit être dans (0.5, 1.0), reçu {confidence}")

        self.window = max(1, int(window))
        self.confidence = float(confidence)
        self.my_mac = my_mac or "?"

        self.xi = 2.0 * (-math.log(1.0 - confidence))
        self.xi = max(1.5, min(self.xi, 4.0))

        self._history: deque = deque()
        self._total_pulls: Dict[str, int] = defaultdict(int)
        self._total_reward: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

        logging.info(f"[SW-UCB:{self.my_mac}] Init : window={self.window}, "
                     f"confidence={self.confidence}, xi={self.xi:.3f}")

    def select(self, candidates: List[str], k: int, current_round: int) -> List[str]:
        with self._lock:
            if not candidates:
                return []
            self._purge_old(current_round)
            ucb_scores = self._compute_ucb(candidates, current_round)

            shuffled = candidates.copy()
            random.shuffle(shuffled)
            ranked = sorted(shuffled, key=lambda c: ucb_scores[c], reverse=True)

            sample_size = min(k, len(ranked))
            chosen = ranked[:sample_size]

            logging.debug(f"[SW-UCB:{self.my_mac}] Round {current_round} : "
                          f"choisi {chosen} parmi {len(candidates)}. "
                          f"Top UCB : {[(c, round(ucb_scores[c], 3)) for c in chosen]}")
            return chosen

    def update(self, arm: str, reward: float, current_round: int) -> None:
        if not math.isfinite(reward):
            logging.warning(f"[SW-UCB:{self.my_mac}] Récompense non finie pour {arm}, ignorée.")
            return
        reward = max(0.0, min(1.0, reward))
        with self._lock:
            self._history.append((current_round, arm, reward))
            self._total_pulls[arm] += 1
            self._total_reward[arm] += reward
            self._purge_old(current_round)
        logging.debug(f"[SW-UCB:{self.my_mac}] Update arm={arm} reward={reward:.3f} round={current_round}")

    def update_from_transfer(self, arm: str, bytes_sent: int,
                             duration_s: float, success: bool,
                             current_round: int) -> None:
        if duration_s > 0 and bytes_sent > 0:
            bw_mbps = (bytes_sent * 8.0) / duration_s / 1e6
        else:
            bw_mbps = 0.0

        bw_norm = min(bw_mbps / self.BW_MAX_MBPS, 1.0)
        lat_norm = min(duration_s / self.LATENCY_MAX_S, 1.0)
        succ = 1.0 if success else 0.0

        reward = (self.W_BW * bw_norm
                  + self.W_SUCCESS * succ
                  + self.W_LATENCY * (1.0 - lat_norm))

        logging.debug(f"[SW-UCB:{self.my_mac}] Transfert {arm} : bw={bw_mbps:.1f} Mbps "
                      f"dur={duration_s:.2f}s succ={success} -> reward={reward:.3f}")
        self.update(arm, reward, current_round)

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "my_mac": self.my_mac,
                "window": self.window,
                "confidence": self.confidence,
                "xi": self.xi,
                "history_size": len(self._history),
                "n_arms_seen": len(self._total_pulls),
                "per_arm": {
                    arm: {
                        "total_pulls": self._total_pulls[arm],
                        "total_reward": round(self._total_reward[arm], 3),
                        "avg_reward": round(self._total_reward[arm] / self._total_pulls[arm], 3)
                        if self._total_pulls[arm] > 0 else 0.0,
                    }
                    for arm in self._total_pulls
                },
            }

    def _purge_old(self, current_round: int) -> None:
        cutoff = current_round - self.window
        while self._history and self._history[0][0] <= cutoff:
            self._history.popleft()

    def _compute_ucb(self, candidates: List[str], current_round: int) -> Dict[str, float]:
        sum_reward: Dict[str, float] = defaultdict(float)
        count: Dict[str, int] = defaultdict(int)
        for (_, arm, reward) in self._history:
            sum_reward[arm] += reward
            count[arm] += 1

        t_prime = max(1, min(current_round, self.window))
        log_t = math.log(t_prime) if t_prime > 1 else 1.0

        ucb: Dict[str, float] = {}
        for arm in candidates:
            n = count[arm]
            if n == 0:
                ucb[arm] = float("inf")
                continue
            mean_r = sum_reward[arm] / n
            explore = math.sqrt(self.xi * log_t / n)
            ucb[arm] = mean_r + explore
        return ucb


# ─── API legacy ──────────────────────────────────────────────────────────────
def get_peer_sample(my_mac: str, all_macs: List[str], k: int) -> List[str]:
    """[LEGACY] Sélection aléatoire pure (rétro-compatibilité)."""
    candidates = [mac for mac in all_macs if mac != my_mac]
    if not candidates:
        return []
    return random.sample(candidates, min(k, len(candidates)))


def build_peer_sampling_table(my_mac: str, all_macs: List[str], k: int) -> Dict:
    return {"my_mac": my_mac, "k": k, "sampled_peers": get_peer_sample(my_mac, all_macs, k)}
