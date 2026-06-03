"""
Suivi et rapport des statistiques d'apprentissage.

StatsTracker  : côté volontaire — round par round.
GlobalStats   : côté manager  — vue agrégée du système.
"""
import json
import os
import threading
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List


# ─── Statistiques par round (volontaire) ─────────────────────────────────────

@dataclass
class RoundStats:
    round_num:          int
    volunteer_ip:       str
    train_loss:         float
    train_acc:          float
    test_acc:           float
    train_duration_s:   float
    bytes_sent:         int
    bytes_received:     int
    n_models_received:  int
    compression_ratio:  float          # original_bytes / compressed_bytes
    timestamp:          float = field(default_factory=time.time)
    # New traceability fields
    neighbors_info:     List[dict] = field(default_factory=list)  # voisins renvoyés par le manager (ressources + score)
    sent_details:       List[dict] = field(default_factory=list)  # liste d'envois : dest, bytes, duration, ts_start, ts_end
    recv_details:       List[dict] = field(default_factory=list)  # liste de réceptions : sender, bytes, send_duration, send_ts_start, send_ts_end, recv_ts
    round_start_ts:     float = 0.0
    round_end_ts:       float = 0.0
    round_duration_s:   float = 0.0
    best_test_acc_so_far: float = 0.0
    best_test_acc_ts:   float = 0.0


class StatsTracker:
    """Suivi côté volontaire."""

    def __init__(self, volunteer_ip: str, results_dir: str = "./results"):
        self.volunteer_ip  = volunteer_ip
        self.results_dir   = results_dir
        self.rounds: List[RoundStats] = []
        self._total_sent = 0
        self._total_recv = 0
        self._lock = threading.Lock()
        os.makedirs(results_dir, exist_ok=True)

    def record(self, **kwargs) -> RoundStats:
        st = RoundStats(volunteer_ip=self.volunteer_ip, **kwargs)
        with self._lock:
            self.rounds.append(st)
            self._total_sent += st.bytes_sent
            self._total_recv += st.bytes_received
        logging.info(
            f"[Stats] Round {st.round_num:3d} | "
            f"loss={st.train_loss:.4f}  train_acc={st.train_acc:.3f}  "
            f"test_acc={st.test_acc:.3f}  "
            f"↑{st.bytes_sent/1024:.0f}KB ↓{st.bytes_received/1024:.0f}KB  "
            f"ratio_compr={st.compression_ratio:.1f}x  "
            f"durée={st.train_duration_s:.1f}s"
        )
        return st

    def summary(self) -> dict:
        with self._lock:
            if not self.rounds:
                return {}
            return {
                "volunteer_ip":          self.volunteer_ip,
                "total_rounds":          len(self.rounds),
                "best_test_acc":         max(r.test_acc for r in self.rounds),
                "final_test_acc":        self.rounds[-1].test_acc,
                "total_train_duration_s": sum(r.train_duration_s for r in self.rounds),
                "total_bytes_sent":      self._total_sent,
                "total_bytes_received":  self._total_recv,
                "avg_compression_ratio": (
                    sum(r.compression_ratio for r in self.rounds) / len(self.rounds)
                ),
                "rounds": [asdict(r) for r in self.rounds],
            }

    def save(self) -> str:
        safe_ip = self.volunteer_ip.replace(".", "_")
        path = os.path.join(self.results_dir, f"volunteer_{safe_ip}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2)
        logging.info(f"[Stats] Sauvegarde → {path}")
        return path


# ─── Statistiques globales (manager) ─────────────────────────────────────────

class GlobalStats:
    """Suivi côté manager : échanges de modèles + résumés par volontaire."""

    def __init__(self, results_dir: str = "./results"):
        self.results_dir = results_dir
        self._start   = time.time()
        self._lock    = threading.Lock()
        self._exchanges: List[dict] = []         # historique des échanges
        self._vol_stats: Dict[str, dict] = {}    # résumé reçu de chaque volontaire
        self._total_routed = 0                   # octets routés au total
        os.makedirs(results_dir, exist_ok=True)

    # ─── Enregistrement ───────────────────────────────────────────────────────

    def record_exchange(self, sender: str, receiver: str, payload_bytes: int, metadata: dict = None, delivered_ts: float = None) -> None:
        """Enregistre un échange de modèle. Peut inclure des métadonnées optionnelles
        fournies par l'expéditeur (ex: `send_ts_start`) et un timestamp de livraison
        `delivered_ts` lorsque la livraison a eu lieu.
        """
        with self._lock:
            entry = {
                "sender": sender,
                "receiver": receiver,
                "bytes": payload_bytes,
                "queued_ts": time.time(),
            }
            if metadata:
                # Copier les métadonnées utiles (payload_bytes, send_ts_start, ...)
                entry.update({k: v for k, v in metadata.items() if k is not None})
            if delivered_ts:
                entry["delivered_ts"] = delivered_ts
                if entry.get("send_ts_start"):
                    try:
                        entry["transfer_time_s"] = delivered_ts - float(entry.get("send_ts_start"))
                    except Exception:
                        entry["transfer_time_s"] = None
            self._exchanges.append(entry)
            self._total_routed += payload_bytes

    def update_volunteer_summary(self, vol_ip: str, summary: dict) -> None:
        with self._lock:
            self._vol_stats[vol_ip] = summary

    # ─── Résumé ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        with self._lock:
            runtime = time.time() - self._start
            return {
                "runtime_s":              runtime,
                "n_active_volunteers":    len(self._vol_stats),
                "total_model_exchanges":  len(self._exchanges),
                "total_bytes_routed":     self._total_routed,
                "throughput_KB_per_s":    self._total_routed / max(runtime, 1) / 1024,
                "volunteer_summaries":    dict(self._vol_stats),
                "exchanges":              list(self._exchanges),
            }

    def print_summary(self) -> None:
        s = self.summary()
        sep = "=" * 64
        logging.info(sep)
        logging.info("STATISTIQUES GLOBALES DU SYSTÈME")
        logging.info(sep)
        logging.info(f"  Durée d'exécution    : {s['runtime_s']:.0f} s")
        logging.info(f"  Volontaires actifs   : {s['n_active_volunteers']}")
        logging.info(f"  Échanges de modèles  : {s['total_model_exchanges']}")
        logging.info(f"  Octets routés        : {s['total_bytes_routed']/1024/1024:.2f} MB")
        logging.info(f"  Débit moyen          : {s['throughput_KB_per_s']:.1f} KB/s")
        for ip, vs in s["volunteer_summaries"].items():
            logging.info(
                f"  [{ip}] rounds={vs.get('total_rounds',0)}  "
                f"best_acc={vs.get('best_test_acc',0):.3f}  "
                f"BW_sent={vs.get('total_bytes_sent',0)/1024:.0f} KB"
            )
        logging.info(sep)

    def save(self) -> str:
        path = os.path.join(self.results_dir, "global_stats.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2)
        logging.info(f"[GlobalStats] Sauvegarde → {path}")
        return path
