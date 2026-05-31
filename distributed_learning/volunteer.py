#!/usr/bin/env python3
"""
Nœud Volontaire
───────────────
Rôle :
  1. Se connecte au Coordinateur (heartbeat TCP persistant).
  2. Entraîne un modèle local sur sa partition de données.
  3. Interroge le Manager pour obtenir ses voisins XOR.
  4. Envoie son modèle compressé à ses voisins via le Manager (gossip push).
  5. Poll le Manager pour récupérer les modèles envoyés par ses pairs.
  6. Agrège les modèles reçus (FedAvg).
  7. Sauvegarde les statistiques de chaque round.

Démarrage :
  python volunteer.py --id 0 --n-volunteers 5
                      --coordinator 192.168.1.10
                      --manager     192.168.1.11
                      [--my-ip 192.168.1.20]
"""
import argparse
import logging
import os
import random
import signal
import socket
import sys
import threading
import time
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else ".")

from src.config import (
    COORDINATOR_PORT, MANAGER_PORT,
    K_NEIGHBORS, GOSSIP_INTERVAL, GOSSIP_FANOUT,
    LOCAL_EPOCHS, BATCH_SIZE, LEARNING_RATE,
    DATASET, NUM_CLASSES, DATA_PARTITION,
    COMPRESSION, QUANTIZATION_BITS, SPARSIFICATION_RATIO,
    HEARTBEAT_INTERVAL, SOCKET_TIMEOUT,
    MAX_RETRIES, RETRY_DELAY, LOG_LEVEL, STATS_DIR,
)
from src.protocol import (
    send_message, receive_message,
    MSG_HEARTBEAT, MSG_ACK, MSG_DISCONNECT,
    MSG_SEND_MODEL, MSG_POLL_MODELS, MSG_MODEL_DELIVERY,
    MSG_REQUEST_NEIGHBORS, MSG_NEIGHBORS_RESPONSE,
    MSG_STATS_PUSH,
    MSG_ERROR,
)
from src.model import create_model, model_parameter_bytes
from src.dataset import load_dataset
from src.compression import compress_model, decompress_model, average_models, compression_ratio
from src.stats import StatsTracker
from src.volunteer_node import VolunteerNode, get_mac_address, get_resource_info

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  [VOLONTAIRE]    %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Volunteer:
    def __init__(self,
                 volunteer_id: int,
                 n_volunteers: int,
                 coordinator_host: str,
                 manager_host: str,
                 my_ip: str = "",
                 cpu_cores: int = None,
                 ram_gb: float = None,
                 network_bandwidth_mbps: float = None):
        self.vol_id          = volunteer_id
        self.n_volunteers    = n_volunteers
        self.coord_host      = coordinator_host
        self.manager_host    = manager_host
        self.my_ip           = my_ip or self._detect_ip(coordinator_host)
        
        # Obtenir MAC et ressources
        self.mac_address     = get_mac_address(coordinator_host)
        self.resources       = get_resource_info(
            cpu_cores=cpu_cores,
            ram_gb=ram_gb,
            network_bandwidth_mbps=network_bandwidth_mbps
        )

        # Modèle et optimiseur
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model     = create_model(DATASET, NUM_CLASSES).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()

        # Données
        self.train_loader, self.test_loader = load_dataset(
            DATASET, "./data", volunteer_id, n_volunteers, DATA_PARTITION, BATCH_SIZE
        )

        self._running      = True
        self._current_round = 0
        self._neighbors: List[dict] = []
        self._nb_lock = threading.Lock()

        self._model_bytes = model_parameter_bytes(self.model)
        self._stats = StatsTracker(self.my_ip, STATS_DIR)

        logging.info(
            f"Volontaire {self.my_ip} initialisé  "
            f"(id={volunteer_id}, MAC={self.mac_address}, device={self.device}, "
            f"modèle={self._model_bytes/1024:.0f} KB non compressé)  "
            f"Ressources: CPU={self.resources.cpu_cores} cores, "
            f"RAM={self.resources.ram_gb}GB, "
            f"Network={self.resources.network_bandwidth_mbps}Mbps"
        )

    # ─── Entrée principale ────────────────────────────────────────────────────

    def run(self):
        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        # Thread heartbeat vers le coordinateur
        threading.Thread(
            target=self._heartbeat_loop,
            daemon=True, name="heartbeat"
        ).start()

        # Pause courte pour que le coordinateur transmette la liste au manager
        time.sleep(8)

        self._gossip_loop()

    # ─── Heartbeat ───────────────────────────────────────────────────────────

    def _heartbeat_loop(self):
        """Maintient la connexion avec le coordinateur par heartbeats."""
        while self._running:
            conn = None
            try:
                conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                conn.settimeout(SOCKET_TIMEOUT)
                conn.connect((self.coord_host, COORDINATOR_PORT))
                logging.info(f"Connecté au coordinateur {self.coord_host}:{COORDINATOR_PORT}")

                while self._running:
                    # Créer le nœud volontaire avec infos actuelles
                    node = VolunteerNode(
                        mac_address=self.mac_address,
                        resources=self.resources,
                        current_ip=self.my_ip,
                    )
                    
                    send_message(conn, MSG_HEARTBEAT, {
                        "mac_address": self.mac_address,
                        "current_ip": self.my_ip,
                        "resources": self.resources.to_dict(),
                        "round": self._current_round,
                        "ts": time.time(),
                    })
                    msg_type, _, _ = receive_message(conn)
                    if msg_type != MSG_ACK:
                        logging.warning(f"Heartbeat : réponse inattendue {msg_type}")
                    time.sleep(HEARTBEAT_INTERVAL)

            except (ConnectionError, OSError, EOFError) as exc:
                logging.warning(f"Connexion coordinateur perdue : {exc}")
            except Exception as exc:
                logging.error(f"Heartbeat : erreur inattendue : {exc}")
            finally:
                if conn:
                    try:
                        conn.close()
                    except Exception:
                        pass

            if self._running:
                logging.info(f"Reconnexion coordinateur dans {RETRY_DELAY}s…")
                time.sleep(RETRY_DELAY)

    # ─── Boucle gossip principale ─────────────────────────────────────────────

    def _gossip_loop(self):
        """Boucle principale : entraînement local gossip push/pull agrégation."""
        while self._running:
            self._current_round += 1
            t_round = time.time()
            bytes_sent = 0
            bytes_recv = 0

            logging.info(f"{'─'*56}")
            logging.info(f"ROUND {self._current_round}")

            # 1. Mise à jour des voisins
            neighbors = self._fetch_neighbors()
            with self._nb_lock:
                self._neighbors = neighbors
            logging.info(f"Voisins XOR : {neighbors}")

            # 2. Entraînement local
            loss, tr_acc, duration = self._train()

            # 3. Push : envoi modèle aux voisins les plus prometteurs
            if neighbors:
                targets = neighbors[:min(GOSSIP_FANOUT, len(neighbors))]
                target_macs = [t.get("mac_address") for t in targets]
                logging.info(f"Envoi adaptatif vers : {target_macs}")
                for target in targets:
                    dest = target.get("mac_address") or target.get("current_ip")
                    ok, sent = self._push_model(dest)
                    if ok:
                        bytes_sent += sent

            # 4. Pull : récupération des modèles envoyés par les pairs
            received_states, recv = self._pull_models()
            bytes_recv = recv

            # 5. Agrégation FedAvg
            if received_states:
                average_models(self.model, received_states)
                logging.info(f"Agrégation de {len(received_states)} modèle(s) reçu(s)")

            # 6. Évaluation
            test_acc = self._evaluate()

            # 7. Ratio de compression
            ratio = compression_ratio(self._model_bytes, bytes_sent) if bytes_sent > 0 else 1.0

            # 8. Enregistrement stats local
            self._stats.record(
                round_num         = self._current_round,
                train_loss        = loss,
                train_acc         = tr_acc,
                test_acc          = test_acc,
                train_duration_s  = duration,
                bytes_sent        = bytes_sent,
                bytes_received    = bytes_recv,
                n_models_received = len(received_states),
                compression_ratio = ratio,
            )

            # 9. Envoi des stats au manager pour monitoring centralisé
            self._push_stats_to_manager()

            # 10. Attente avant prochain round
            elapsed = time.time() - t_round
            wait    = max(0.0, GOSSIP_INTERVAL - elapsed)
            if wait > 0:
                logging.debug(f"Attente {wait:.1f}s avant prochain round…")
                time.sleep(wait)

        self._stats.save()

    # ─── Entraînement local ───────────────────────────────────────────────────

    def _train(self):
        self.model.train()
        t0 = time.time()
        total_loss, correct, total = 0.0, 0, 0

        for _epoch in range(LOCAL_EPOCHS):
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                out  = self.model(X)
                loss = self.criterion(out, y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                correct    += out.argmax(1).eq(y).sum().item()
                total      += len(y)

        n_batches = len(self.train_loader) * LOCAL_EPOCHS
        avg_loss  = total_loss / max(n_batches, 1)
        train_acc = correct / max(total, 1)
        duration  = time.time() - t0
        logging.info(
            f"Entraînement : loss={avg_loss:.4f}  acc={train_acc:.3f}  "
            f"({LOCAL_EPOCHS} epochs, {duration:.1f}s)"
        )
        return avg_loss, train_acc, duration

    # ─── Évaluation ───────────────────────────────────────────────────────────

    def _evaluate(self) -> float:
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                out  = self.model(X)
                correct += out.argmax(1).eq(y).sum().item()
                total   += len(y)
        acc = correct / max(total, 1)
        logging.info(f"Précision test : {acc:.4f}  ({correct}/{total})")
        return acc

    # ─── Communication avec le Manager ────────────────────────────────────────

    def _fetch_neighbors(self) -> List[dict]:
        """Demande la liste des k voisins XOR au Manager.

        Le manager renvoie des objets de nœud contenant l'adresse MAC, l'IP
        courante et les ressources allouées pour la sélection adaptative.
        """
        for attempt in range(MAX_RETRIES):
            try:
                conn = self._connect_manager()
                send_message(conn, MSG_REQUEST_NEIGHBORS,
                             {"volunteer_ip": self.my_ip, "k": K_NEIGHBORS})
                msg_type, data, _ = receive_message(conn)
                conn.close()
                if msg_type == MSG_NEIGHBORS_RESPONSE:
                    return data.get("neighbors", [])
            except Exception as exc:
                logging.warning(f"Voisins (essai {attempt+1}) : {exc}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
        return []

    def _push_model(self, dest_ip: str):
        """Compresse et envoie le modèle au Manager à destination de dest_ip."""
        try:
            compressed, meta = compress_model(
                self.model, COMPRESSION,
                bits=QUANTIZATION_BITS, ratio=SPARSIFICATION_RATIO
            )
            conn = self._connect_manager()
            send_message(conn, MSG_SEND_MODEL,
                         {"sender_ip": self.my_ip, "dest_ip": dest_ip, "metadata": meta},
                         compressed)
            msg_type, rsp, _ = receive_message(conn)
            conn.close()

            if msg_type == MSG_ACK:
                ratio = compression_ratio(self._model_bytes, len(compressed))
                logging.info(
                    f"Modèle envoyé → {dest_ip}  "
                    f"({len(compressed)/1024:.1f} KB, ratio={ratio:.1f}x)"
                )
                return True, len(compressed)
            logging.warning(f"Push refusé par manager : {rsp}")
        except Exception as exc:
            logging.warning(f"Push vers {dest_ip} échoué : {exc}")
        return False, 0

    def _pull_models(self):
        """Récupère les modèles en attente pour ce volontaire."""
        received_states = []
        total_recv = 0

        # Plusieurs polls jusqu'à ce que la file soit vide
        for _ in range(10):
            try:
                conn = self._connect_manager()
                send_message(conn, MSG_POLL_MODELS,
                             {"volunteer_ip": self.my_ip, "max_models": 5})
                msg_type, data, payload = receive_message(conn)
                conn.close()

                if msg_type == MSG_MODEL_DELIVERY and payload:
                    meta   = data.get("metadata", {})
                    rcv_m  = create_model(DATASET, NUM_CLASSES).to(self.device)
                    decompress_model(rcv_m, payload, meta)
                    received_states.append(rcv_m.state_dict())
                    total_recv += len(payload)
                    logging.info(
                        f"Modèle reçu de {data.get('sender_ip')}  "
                        f"({len(payload)/1024:.1f} KB)"
                    )
                    # S'il reste des modèles en file, on reboucle
                    if data.get("n_pending", 0) == 0:
                        break
                else:
                    break   # File vide
            except Exception as exc:
                logging.warning(f"Pull échoué : {exc}")
                break

        return received_states, total_recv

    def _push_stats_to_manager(self):
        """Envoie un résumé compact des stats du volontaire au manager."""
        with self._stats._lock:
            rounds = self._stats.rounds
            if not rounds:
                return
            compact = {
                "current_round":          self._current_round,
                "total_rounds":           len(rounds),
                "best_test_acc":          max(r.test_acc for r in rounds),
                "final_test_acc":         rounds[-1].test_acc,
                "total_bytes_sent":       self._stats._total_sent,
                "total_bytes_received":   self._stats._total_recv,
                "total_train_duration_s": sum(r.train_duration_s for r in rounds),
                "avg_compression_ratio":  (
                    sum(r.compression_ratio for r in rounds) / len(rounds)
                ),
            }
        try:
            conn = self._connect_manager()
            send_message(conn, MSG_STATS_PUSH, {
                "volunteer_ip": self.my_ip,
                "summary": compact,
            })
            conn.close()
        except Exception as exc:
            logging.debug(f"Push stats manager échoué : {exc}")

    def _connect_manager(self) -> socket.socket:
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn.settimeout(SOCKET_TIMEOUT)
        conn.connect((self.manager_host, MANAGER_PORT))
        return conn

    # ─── Utilitaires ──────────────────────────────────────────────────────────

    @staticmethod
    def _detect_ip(remote_host: str) -> str:
        """Détecte l'IP locale utilisée pour atteindre remote_host."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((remote_host, 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def _shutdown(self, *_):
        logging.info("Arrêt du volontaire…")
        self._running = False
        self._stats.save()
        sys.exit(0)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Nœud Volontaire — Apprentissage distribué")
    p.add_argument("--id",           type=int, required=True,
                   help="Identifiant du volontaire (0-indexé)")
    p.add_argument("--n-volunteers", type=int, default=5,
                   help="Nombre total de volontaires attendus")
    p.add_argument("--coordinator",  type=str, required=True,
                   help="IP/hostname du coordinateur")
    p.add_argument("--manager",      type=str, required=True,
                   help="IP/hostname du manager")
    p.add_argument("--my-ip",        type=str, default="",
                   help="IP publique de cette machine (détection auto si omis)")
    p.add_argument("--cpu-cores",    type=int, default=None,
                   help="Nombre de cœurs CPU alloués (détection auto si omis)")
    p.add_argument("--ram-gb",       type=float, default=None,
                   help="RAM allouée en GB (détection auto si omis)")
    p.add_argument("--network-mbps", type=float, default=None,
                   help="Bande passante réseau allouée en Mbps (défaut 1000)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Volunteer(
        volunteer_id     = args.id,
        n_volunteers     = args.n_volunteers,
        coordinator_host = args.coordinator,
        manager_host     = args.manager,
        my_ip            = args.my_ip,
        cpu_cores        = args.cpu_cores,
        ram_gb           = args.ram_gb,
        network_bandwidth_mbps = args.network_mbps,
    ).run()
