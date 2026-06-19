#!/usr/bin/env python3
"""
Nœud Volontaire
───────────────
Rôle :
  1. Se connecte au Coordinateur (heartbeat TCP persistant).
  2. Entraîne un modèle local sur sa partition de données.
  3. Interroge le Manager pour obtenir les volontaires actifs.
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
import math
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
    LOCAL_EPOCHS, MAX_ROUNDS, BATCH_SIZE, LEARNING_RATE,
    DATASET, NUM_CLASSES, DATA_PARTITION,
    COMPRESSION, QUANTIZATION_BITS, SPARSIFICATION_RATIO,
    HEARTBEAT_INTERVAL, SOCKET_TIMEOUT,
    MAX_RETRIES, RETRY_DELAY, LOG_LEVEL, STATS_DIR,
    SW_UCB_CONFIDENCE,
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
from src.peer_sampling import get_peer_sample
from src.profiler import SystemProfiler, ModelProfiler
import psutil

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
        base_mac = get_mac_address(coordinator_host)
        parts = base_mac.split(":")
        if len(parts) == 6:
            parts[-1] = f"{self.vol_id:02X}"
            self.mac_address = ":".join(parts)
        else:
            self.mac_address = f"{base_mac}_{self.vol_id}"
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

        # Initialisation des profileurs et pré-check
        self.system_profiler = SystemProfiler()
        self.model_profiler = ModelProfiler(self.model)
        
        logging.info("--- Exécution du System Profiler & Model Profiler (Pré-check) ---")
        sys_avail = self.system_profiler.get_available_resources(self.resources.network_bandwidth_mbps)
        model_est = self.model_profiler.estimate_needs(
            dataset_name=DATASET,
            batch_size=BATCH_SIZE,
            optimizer_type="sgd",
            compression_type=COMPRESSION,
            quantization_bits=QUANTIZATION_BITS,
            sparsification_ratio=SPARSIFICATION_RATIO,
            gossip_interval=GOSSIP_INTERVAL,
            fanout=GOSSIP_FANOUT,
            network_bandwidth_mbps=self.resources.network_bandwidth_mbps
        )
        
        logging.info(f"System Profiler (Disponible) : RAM={sys_avail['ram_free']}GB, CPU_Load={sys_avail['cpu_load']}%, Battery={sys_avail['battery']}%")
        logging.info(f"Model Profiler (Requis)     : RAM_est={model_est['ram_needed']}GB, Gradients={model_est['gradient_size']}MB, Epoch_Time_est={model_est['epoch_time_estimate']}s")
        
        if sys_avail["ram_free"] < model_est["ram_needed"]:
            logging.error(f"[REFUS LOCAL] Ressources RAM insuffisantes : Disponible={sys_avail['ram_free']}GB < Requise={model_est['ram_needed']}GB")
            sys.exit(1)
        else:
            logging.info("[ACCEPTATION LOCALE] Ressources suffisantes détectées.")

        # Données
        self.train_loader, self.test_loader = load_dataset(
            DATASET, "./data", volunteer_id, n_volunteers, DATA_PARTITION, BATCH_SIZE
        )

        self._running      = True
        self._current_round = 0
        self.max_rounds     = MAX_ROUNDS
        self._neighbors: List[dict] = []
        self._neighbor_request_count = 0
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
        logging.info("Attente d'enregistrement du coordinateur (15s pour synchronisation)…")
        time.sleep(15)
        logging.info("Démarrage de la boucle gossip.")

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
            if self.max_rounds > 0 and self._current_round > self.max_rounds:
                logging.info(
                    f"Nombre maximum de rounds atteint ({self.max_rounds}), arrêt du volontaire."
                )
                break
            t_round = time.time()
            bytes_sent = 0
            bytes_recv = 0

            logging.info(f"{'─'*56}")
            logging.info(f"ROUND {self._current_round}")

            # 1. Mise à jour des voisins
            neighbors = self._fetch_neighbors()
            with self._nb_lock:
                self._neighbors = neighbors
            logging.info(f"Voisins échantillonnés (Peer Sampling) : {neighbors}")

            # 2. Entraînement local
            self.system_profiler.start_monitoring()
            loss, tr_acc, duration = self._train()
            sys_report = self.system_profiler.stop_monitoring()

            # 3. Push : envoi modèle aux voisins les plus prometteurs
            sent_details = []
            if neighbors:
                targets = neighbors[:min(GOSSIP_FANOUT, len(neighbors))]
                target_macs = [t.get("mac_address") for t in targets]
                logging.info(f"Envoi adaptatif vers : {target_macs}")
                for target in targets:
                    dest_mac = target.get("mac_address")
                    dest_ip  = target.get("current_ip")
                    # ✅ FIX BUG 6 : on passe dest_mac ET dest_ip séparément
                    # pour que le manager puisse résoudre le destinataire soit
                    # via _ip_to_mac (quand c'est une IP), soit directement
                    # dans self._volunteers (quand c'est un MAC).
                    # Avant : dest = dest_mac or dest_ip était envoyé comme
                    # champ dest_ip → si c'était un MAC, le manager ne le
                    # trouvait jamais via sa table IP → destinataire inconnu.
                    ok, sent, send_duration, send_ts_start, send_ts_end = self._push_model(
                        dest_mac=dest_mac, dest_ip=dest_ip
                    )
                    if ok:
                        bytes_sent += sent
                        sent_details.append({
                            "dest_mac": dest_mac,
                            "dest_ip": dest_ip,
                            "bytes": sent,
                            "send_duration_s": send_duration,
                            "send_ts_start": send_ts_start,
                            "send_ts_end": send_ts_end,
                        })

            # 4. Pull : récupération des modèles envoyés par les pairs
            received_states, recv, recv_details = self._pull_models()
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
            # Round timing and best-accuracy tracking
            round_end_ts = time.time()
            round_duration = round_end_ts - t_round

            # Best test acc so far (including this round)
            with self._stats._lock:
                prev_rounds = list(self._stats.rounds)
            best_acc = test_acc
            best_ts = round_end_ts
            for r in prev_rounds:
                if getattr(r, "test_acc", 0) > best_acc:
                    best_acc = r.test_acc
                    best_ts = getattr(r, "timestamp", best_ts)

            # Sauvegarder la dernière précision test pour le rapport final
            self._last_test_acc = test_acc

            # Métriques modèle temps réel du round
            gradient_size_mb = self.model_profiler.param_bytes / (1024**2)
            batch_time_avg_s = sum(self.model_profiler.batch_times[-len(self.train_loader):]) / max(len(self.train_loader), 1) if self.model_profiler.batch_times else 0.0

            # Obtenir le niveau de batterie
            try:
                bat = psutil.sensors_battery()
                battery_level = bat.percent if bat is not None else 100.0
            except:
                battery_level = 100.0

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
                neighbors_info    = neighbors,
                sent_details      = sent_details,
                recv_details      = recv_details,
                round_start_ts    = t_round,
                round_end_ts      = round_end_ts,
                round_duration_s  = round_duration,
                best_test_acc_so_far = best_acc,
                best_test_acc_ts  = best_ts,
                # New profiling fields
                cpu_percent_peak   = sys_report["cpu_peak"],
                cpu_percent_mean   = sys_report["cpu_avg"],
                ram_usage_gb_peak  = sys_report["ram_peak"],
                ram_usage_gb_mean  = sum(self.system_profiler.ram_history) / max(len(self.system_profiler.ram_history), 1) if self.system_profiler.ram_history else 0.0,
                battery_level      = battery_level,
                energy_used_joules = sys_report["energy_used"],
                gradient_size_mb   = gradient_size_mb,
                batch_time_avg_s   = batch_time_avg_s,
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
        
        self.model_profiler.start_training_tracking()

        for epoch in range(LOCAL_EPOCHS):
            t_epoch_start = time.time()
            correct_epoch, total_epoch = 0, 0
            for X, y in self.train_loader:
                t_batch_start = time.time()
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                out  = self.model(X)
                loss = self.criterion(out, y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                correct    += out.argmax(1).eq(y).sum().item()
                total      += len(y)
                
                correct_epoch += out.argmax(1).eq(y).sum().item()
                total_epoch   += len(y)
                
                batch_duration = time.time() - t_batch_start
                self.model_profiler.record_batch(batch_duration, loss.item())
                
            epoch_duration = time.time() - t_epoch_start
            epoch_acc = correct_epoch / max(total_epoch, 1)
            self.model_profiler.record_epoch(epoch_duration, epoch_acc)

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
        """Demande la liste complète des volontaires au Manager, effectue le Peer Sampling
        pour choisir k voisins de manière aléatoire et les ordonne avec SW-UCB.
        """
        for attempt in range(MAX_RETRIES):
            try:
                conn = self._connect_manager()
                # envoyer aussi volunteer_mac pour que le manager
                # puisse identifier et exclure ce volontaire de sa propre liste.
                send_message(conn, MSG_REQUEST_NEIGHBORS,
                             {"volunteer_ip": self.my_ip,
                              "volunteer_mac": self.mac_address,
                              "k": K_NEIGHBORS})
                msg_type, data, _ = receive_message(conn)
                conn.close()
                if msg_type == MSG_NEIGHBORS_RESPONSE:
                    volunteers_data = data.get("volunteers", [])
                    logging.debug(f"[Demande voisins] Manager a retourné {len(volunteers_data)} volontaires")
                    if not volunteers_data:
                        logging.warning("[Demande voisins] Manager n'a retourné aucun volontaire (pas encore enregistrés?)")
                        if attempt < MAX_RETRIES - 1:
                            time.sleep(RETRY_DELAY)
                        continue
                    return self._compute_local_neighbors(volunteers_data)
            except Exception as exc:
                logging.warning(f"[Demande voisins] Essai {attempt+1}/{MAX_RETRIES} échoué : {exc}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
        logging.error("[Demande voisins] Impossible d'obtenir la liste des volontaires après plusieurs tentatives")
        return []

    def _compute_local_neighbors(self, volunteers_data: List[dict]) -> List[dict]:
        """Effectue le Peer Sampling local et trie par SW-UCB."""
        my_mac = self.mac_address
        
        logging.debug(f"[Calcul voisins] Total volontaires reçus du manager : {len(volunteers_data)}")

        # Extraire tous les MACs des autres volontaires
        all_candidate_macs = [v["mac_address"] for v in volunteers_data if v["mac_address"] != my_mac]
        logging.info(f"[Calcul voisins] Candidats disponibles : {len(all_candidate_macs)} (après exclusion de soi-même)")

        if not all_candidate_macs:
            logging.warning(
                f"[Calcul voisins] Aucun autre volontaire disponible pour le Peer Sampling. "
                f"En attente d'autres volontaires…"
            )
            return []

        # Obtenir les k voisins échantillonnés aléatoirement (Peer Sampling)
        k_neighbors_requested = min(K_NEIGHBORS, len(all_candidate_macs))
        k_nearest = get_peer_sample(my_mac, all_candidate_macs, k_neighbors_requested)
        logging.info(
            f"[Calcul voisins] Peer Sampling : demandé {K_NEIGHBORS} voisins aléatoires, "
            f"obtenu {len(k_nearest)} (candidats : {len(all_candidate_macs)})"
        )

        if not k_nearest:
            logging.warning("[Calcul voisins] Le Peer Sampling a retourné une liste vide.")
            return []

        # Classer par SW-UCB
        self._neighbor_request_count += 1
        t = max(1, self._neighbor_request_count)

        vol_by_mac = {v["mac_address"]: v for v in volunteers_data}

        scored = []
        for mac in k_nearest:
            node_dict = vol_by_mac.get(mac)
            if not node_dict:
                logging.warning(f"[Calcul voisins] MAC {mac} introuvable dans les données reçues.")
                continue
            history = node_dict.get("bandwidth_history", [])
            if history:
                avg = sum(history) / len(history)
                count = len(history)
                bonus = SW_UCB_CONFIDENCE * math.sqrt(2 * math.log(t) / count)
                score = avg + bonus
                logging.debug(f"[Calcul voisins] {mac}: moyenne={avg:.2f}, historique={len(history)}, score={score:.2f}")
            else:
                base = node_dict.get("resources", {}).get("network_bandwidth_mbps", 1000.0)
                score = base + SW_UCB_CONFIDENCE * math.sqrt(2 * math.log(t))
                logging.debug(f"[Calcul voisins] {mac}: BW_base={base:.2f}, score={score:.2f} (pas historique)")
            scored.append((mac, score))

        scored.sort(key=lambda item: item[1], reverse=True)

        neighbors_info = []
        for idx, (mac, score) in enumerate(scored, start=1):
            node_dict = vol_by_mac[mac]
            info = dict(node_dict)
            info["sw_ucb_score"] = score
            info["sw_ucb_rank"] = idx
            neighbors_info.append(info)

        logging.info(
            f"[Calcul voisins] Topologie finalisée : {len(neighbors_info)} voisins "
            f"(scores SW-UCB: {[f'{s:.2f}' for _, s in scored]})"
        )
        return neighbors_info

    def _push_model(self, dest_mac: str = None, dest_ip: str = None):
        """Compresse et envoie le modèle au Manager à destination de dest_mac / dest_ip.

        FIX BUG 6 : l'ancienne signature n'acceptait qu'un seul paramètre `dest_ip`
        mais l'appelant lui passait parfois dest_mac (via `dest = dest_mac or dest_ip`).
        Le manager cherchait ce champ dans _ip_to_mac et current_ip → introuvable
        si c'était un MAC → "Destinataire inconnu" → aucun modèle livré.

        Correction : on envoie les deux champs. Le manager essaie d'abord dest_mac
        (lookup direct dans self._volunteers), puis dest_ip (via _ip_to_mac).
        """
        # Choisir le meilleur identifiant pour les logs
        dest_label = dest_mac or dest_ip or "?"
        try:
            compressed, meta = compress_model(
                self.model, COMPRESSION,
                bits=QUANTIZATION_BITS, ratio=SPARSIFICATION_RATIO
            )

            conn = self._connect_manager()
            send_ts_start = time.time()
            meta.update({
                "payload_bytes": len(compressed),
                "send_ts_start": send_ts_start,
            })
            #FIX : envoyer dest_mac ET dest_ip pour que le manager
            # puisse résoudre le destinataire quel que soit le format.
            # Le manager consulte d'abord dest_mac (clé directe dans _volunteers),
            # puis dest_ip (via _ip_to_mac), puis scan des current_ip.
            send_message(conn, MSG_SEND_MODEL,
                         {
                             "sender_ip":  self.my_ip,
                             "sender_mac": self.mac_address,
                             "dest_ip":    dest_ip or "",
                             "dest_mac":   dest_mac or "",
                             "metadata":   meta,
                         },
                         compressed)
            msg_type, rsp, _ = receive_message(conn)
            send_ts_end = time.time()
            conn.close()

            send_duration = send_ts_end - send_ts_start
            meta["send_duration_s"] = send_duration

            if msg_type == MSG_ACK:
                ratio = compression_ratio(self._model_bytes, len(compressed))
                logging.info(
                    f"Modèle envoyé → {dest_label}  "
                    f"({len(compressed)/1024:.1f} KB, ratio={ratio:.1f}x)"
                )
                self.model_profiler.record_communication(
                    self._model_bytes / (1024**2),
                    len(compressed) / (1024**2)
                )
                return True, len(compressed), send_duration, send_ts_start, send_ts_end
            logging.warning(f"Push refusé par manager : {rsp}")
        except Exception as exc:
            logging.warning(f"Push vers {dest_label} échoué : {exc}")
        return False, 0, 0.0, 0.0, 0.0

    def _pull_models(self):
        """Récupère les modèles en attente pour ce volontaire."""
        received_states = []
        total_recv = 0
        recv_details = []

        # Plusieurs polls jusqu'à ce que la file soit vide
        for _ in range(10):
            try:
                conn = self._connect_manager()
                send_message(conn, MSG_POLL_MODELS,
                             {
                                 "volunteer_ip":  self.my_ip,
                                 "volunteer_mac": self.mac_address,
                                 "max_models":    5,
                             })
                msg_type, data, payload = receive_message(conn)
                conn.close()

                if msg_type == MSG_MODEL_DELIVERY and payload:
                    meta   = data.get("metadata", {})
                    rcv_m  = create_model(DATASET, NUM_CLASSES).to(self.device)
                    decompress_model(rcv_m, payload, meta)
                    received_states.append(rcv_m.state_dict())
                    total_recv += len(payload)
                    self.model_profiler.record_communication(
                        self._model_bytes / (1024**2),
                        len(payload) / (1024**2)
                    )
                    # Tracer la réception
                    recv_ts = time.time()
                    send_ts_start = meta.get("send_ts_start")
                    transfer_time = None
                    if send_ts_start is not None:
                        try:
                            transfer_time = recv_ts - float(send_ts_start)
                        except Exception:
                            transfer_time = None
                    recv_details.append({
                        "sender": data.get("sender_ip"),
                        "bytes": len(payload),
                        "send_ts_start": send_ts_start,
                        "payload_bytes": meta.get("payload_bytes"),
                        "recv_ts": recv_ts,
                        "send_duration_s": meta.get("send_duration_s"),
                        "transfer_time_s": transfer_time,
                    })
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

        return received_states, total_recv, recv_details

    def _push_stats_to_manager(self):
        """Envoie un résumé compact des stats du volontaire au manager."""
        with self._stats._lock:
            rounds = self._stats.rounds
            if not rounds:
                return
            compact = {
                "current_round":          self._current_round,
                "total_rounds":           len(rounds),
                "max_rounds":             self.max_rounds,
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

    #Utilitaires

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
        
        # Affichage des rapports finaux des profileurs
        try:
            accuracy = getattr(self, "_last_test_acc", 0.0)
            sys_report = self.system_profiler.stop_monitoring() # au cas où
            
            total_sys_report = {
                "cpu_avg": round(sum(self.system_profiler.cpu_history) / max(len(self.system_profiler.cpu_history), 1), 1) if self.system_profiler.cpu_history else 0.0,
                "cpu_peak": max(self.system_profiler.cpu_history) if self.system_profiler.cpu_history else 0.0,
                "ram_peak": max(self.system_profiler.ram_history) if self.system_profiler.ram_history else 0.0,
                "energy_used": round(sys_report.get("energy_used", 0.0), 1)
            }
            model_report = self.model_profiler.generate_report(accuracy)
            
            logging.info("="*60)
            logging.info("RAPPORT FINAL DU SYSTEM PROFILER (MACHINE)")
            logging.info("="*60)
            logging.info(f"  CPU moyen            : {total_sys_report['cpu_avg']}%")
            logging.info(f"  CPU pic              : {total_sys_report['cpu_peak']}%")
            logging.info(f"  RAM pic              : {total_sys_report['ram_peak']:.2f} Go")
            logging.info(f"  Énergie consommée    : {total_sys_report['energy_used']} Joules")
            
            logging.info("="*60)
            logging.info("RAPPORT FINAL DU MODEL PROFILER (MODÈLE)")
            logging.info("="*60)
            logging.info(f"  Temps d'entraînement : {model_report['training_time']} s")
            logging.info(f"  Trafic gradient brut : {model_report['gradient_traffic']:.2f} Mo")
            logging.info(f"  Trafic compressé     : {model_report['compressed_traffic']:.2f} Mo")
            logging.info(f"  Précision finale     : {model_report['final_accuracy']:.2f}%")
            logging.info("="*60)
        except Exception as e:
            logging.warning(f"Impossible de générer les rapports de fin : {e}")
            
        self._running = False
        self._stats.save()
        sys.exit(0)


#CLI

def parse_args():
    # Charger les valeurs depuis les variables d'environnement en guise de valeurs par défaut
    env_id = os.getenv("VOLUNTEER_ID")
    env_coord = os.getenv("COORDINATOR_HOST")
    env_manager = os.getenv("MANAGER_HOST")
    env_n_vol = os.getenv("N_VOLUNTEERS")
    env_my_ip = os.getenv("MY_IP")
    env_cpu = os.getenv("CPU_CORES")
    env_ram = os.getenv("RAM_GB")
    env_net = os.getenv("NETWORK_MBPS")

    p = argparse.ArgumentParser(description="Nœud Volontaire — Apprentissage distribué")
    p.add_argument("--id",           type=int,
                   default=int(env_id) if env_id is not None else None,
                   required=env_id is None,
                   help="Identifiant du volontaire (0-indexé)")
    p.add_argument("--n-volunteers", type=int,
                   default=int(env_n_vol) if env_n_vol is not None else 5,
                   help="Nombre total de volontaires attendus")
    p.add_argument("--coordinator",  type=str,
                   default=env_coord,
                   required=env_coord is None,
                   help="IP/hostname du coordinateur")
    p.add_argument("--manager",      type=str,
                   default=env_manager,
                   required=env_manager is None,
                   help="IP/hostname du manager")
    p.add_argument("--my-ip",        type=str,
                   default=env_my_ip if env_my_ip is not None else "",
                   help="IP publique de cette machine (détection auto si omis)")
    p.add_argument("--cpu-cores",    type=int,
                   default=int(env_cpu) if env_cpu else None,
                   help="Nombre de cœurs CPU alloués (détection auto si omis)")
    p.add_argument("--ram-gb",       type=float,
                   default=float(env_ram) if env_ram else None,
                   help="RAM allouée en GB (détection auto si omis)")
    p.add_argument("--network-mbps", type=float,
                   default=float(env_net) if env_net else None,
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