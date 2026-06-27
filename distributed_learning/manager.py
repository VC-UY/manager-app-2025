#!/usr/bin/env python3
"""
Nœud Manager
─────────────
Rôle :
  1. Reçoit la liste de volontaires du Coordinateur.
  2. Calcule (à la demande) les k voisins XOR de chaque volontaire.
  3. Route les modèles entre volontaires (stockage temporaire par file).
  4. Collecte et affiche les statistiques globales.

Démarrage :
  python manager.py

Variables d'environnement clés :
  MANAGER_HOST    IP d'écoute (défaut 0.0.0.0)
  MANAGER_PORT    Port d'écoute (défaut 9001)
  K_NEIGHBORS     Nombre de voisins XOR (défaut 4)
"""
import logging
import math
import queue
import signal
import socket
import sys
import threading
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else ".")

from src.config import (
    MANAGER_HOST, MANAGER_PORT,
    K_NEIGHBORS, SOCKET_TIMEOUT,
    MAX_CONNECTIONS, MAX_RETRIES, RETRY_DELAY,
    STATS_PRINT_INTERVAL, LOG_LEVEL, STATS_DIR,
    SW_UCB_WINDOW, SW_UCB_CONFIDENCE,
    MODEL_NAME, DATASET, NUM_CLASSES, BATCH_SIZE, COMPRESSION,
    QUANTIZATION_BITS, SPARSIFICATION_RATIO,
    GOSSIP_INTERVAL, GOSSIP_FANOUT,
)
from src.protocol import (
    send_message, receive_message,
    MSG_VOLUNTEER_LIST, MSG_SEND_MODEL, MSG_POLL_MODELS,
    MSG_MODEL_DELIVERY, MSG_REQUEST_NEIGHBORS, MSG_NEIGHBORS_RESPONSE,
    MSG_STATS_REQUEST, MSG_STATS_RESPONSE, MSG_STATS_PUSH,
    MSG_ACK, MSG_ERROR,
)
from src.stats import GlobalStats
from src.volunteer_node import VolunteerNode

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  [MANAGER]       %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Manager:
    def __init__(self):
        self._volunteers: Dict[str, VolunteerNode] = {}  # mac → VolunteerNode
        self._vol_lock  = threading.Lock()

        # Mapping IP courant → MAC (pour supporter les changements d'IP)
        self._ip_to_mac: Dict[str, str] = {}

        # file par MAC destinataire : Queue[(sender_mac, payload_bytes, metadata_dict)]
        self._queues: Dict[str, queue.Queue] = defaultdict(queue.Queue)
        self._q_lock = threading.Lock()

        self._stats   = GlobalStats(results_dir=STATS_DIR)
        self._neighbor_rewards: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=SW_UCB_WINDOW)
        )
        self._neighbor_stats_prev: Dict[str, dict] = {}
        # Estimation des besoins du modèle (ModelProfiler) côté Manager
        try:
            import torch
            from src.model import create_model
            from src.profiler import ModelProfiler
            m = create_model(MODEL_NAME, NUM_CLASSES).to("cpu")
            prof = ModelProfiler(m)
            est = prof.estimate_needs(
                dataset_name=DATASET,
                batch_size=BATCH_SIZE,
                optimizer_type="sgd",
                compression_type=COMPRESSION,
                quantization_bits=QUANTIZATION_BITS,
                sparsification_ratio=SPARSIFICATION_RATIO,
                gossip_interval=GOSSIP_INTERVAL,
                fanout=GOSSIP_FANOUT,
                network_bandwidth_mbps=1000.0
            )
            self._ram_needed = est["ram_needed"]
            logging.info(
                f"Besoins estimés du modèle '{MODEL_NAME}' sur '{DATASET}' : "
                f"RAM nécessaire = {self._ram_needed:.2f} GB"
            )
        except Exception as e:
            logging.warning(f"Impossible d'estimer les besoins du modèle : {e}. Utilisation d'une valeur par défaut (0.5 Go).")
            self._ram_needed = 0.5

        self._running = True

    #Entrée principale

    def run(self):
        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        threads = [
            threading.Thread(target=self._listen,         daemon=True, name="listener"),
            threading.Thread(target=self._stats_reporter, daemon=True, name="reporter"),
        ]
        for t in threads:
            t.start()

        logging.info(f"Manager démarré — écoute {MANAGER_HOST}:{MANAGER_PORT}")

        while self._running:
            time.sleep(1)

    #Serveur TCP

    def _listen(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.settimeout(1.0)
        srv.bind((MANAGER_HOST, MANAGER_PORT))
        srv.listen(MAX_CONNECTIONS)

        while self._running:
            try:
                conn, addr = srv.accept()
                conn.settimeout(SOCKET_TIMEOUT)
                threading.Thread(
                    target=self._dispatch,
                    args=(conn, addr[0]),
                    daemon=True,
                ).start()
            except socket.timeout:
                pass
            except OSError as exc:
                if self._running:
                    logging.error(f"Erreur accept : {exc}")
        srv.close()

    def _dispatch(self, conn: socket.socket, addr: str):
        """Lit le premier message et délègue au bon handler."""
        try:
            msg_type, data, payload = receive_message(conn)

            if   msg_type == MSG_VOLUNTEER_LIST:
                self._on_volunteer_list(data)

            elif msg_type == MSG_SEND_MODEL:
                self._on_send_model(conn, addr, data, payload)

            elif msg_type == MSG_POLL_MODELS:
                self._on_poll(conn, addr, data)

            elif msg_type == MSG_REQUEST_NEIGHBORS:
                self._on_neighbors_request(conn, addr, data)

            elif msg_type == MSG_STATS_REQUEST:
                self._on_stats_request(conn)

            elif msg_type == MSG_STATS_PUSH:
                self._on_stats_push(addr, data)
                send_message(conn, MSG_ACK, {"status": "stats_received"})

            else:
                send_message(conn, MSG_ERROR,
                             {"message": f"Type inconnu : {msg_type}"})
        except Exception as exc:
            logging.warning(f"Erreur dispatch ({addr}) : {exc}")
        finally:
            conn.close()

    def _resolve_mac(self, ip_or_mac: str) -> Optional[str]:
        """Résout une IP ou un MAC vers le MAC connu du volontaire."""
        if ip_or_mac in self._ip_to_mac:
            return self._ip_to_mac[ip_or_mac]
        if ip_or_mac in self._volunteers:
            return ip_or_mac
        for mac, node in self._volunteers.items():
            if node.current_ip == ip_or_mac:
                return mac
        return None

    def _compute_bandwidth_reward(self, mac: str, current_summary: dict) -> float:
        """Calcule un reward de bande passante en Mbps pour SW-UCB."""
        prev = self._neighbor_stats_prev.get(mac)
        node = self._volunteers.get(mac)
        if prev and node is not None:
            delta_bytes = current_summary.get("total_bytes_sent", 0) - prev.get("total_bytes_sent", 0)
            elapsed = time.time() - prev.get("_last_update", time.time())
            if delta_bytes > 0 and elapsed > 0:
                return (delta_bytes * 8) / elapsed / 1_000_000
        return node.resources.network_bandwidth_mbps if node is not None else 0.0

    #Handlers

    def _on_volunteer_list(self, data: dict):
        """Met à jour la liste des volontaires (message du coordinateur).

        FIX BUG 2 : L'ancienne implémentation faisait clear() + reconstruction complète
        à chaque broadcast (toutes les 5s). Si un volontaire demandait ses voisins
        exactement pendant ce clear, il recevait une liste vide → aucun échange.

        Correction : mise à jour DIFFÉRENTIELLE.
        - On ajoute / met à jour les volontaires présents dans le nouveau broadcast.
        - On supprime uniquement ceux qui n'y figurent plus (réellement partis).
        - Les données existantes ne sont jamais effacées brutalement.
        """
        new_list = data.get("volunteers", [])
        new_macs = set()

        with self._vol_lock:
            for vol_data in new_list:
                try:
                    node = VolunteerNode.from_dict(vol_data)
                    mac = node.mac_address
                    new_macs.add(mac)

                    if mac not in self._volunteers:
                        # Nouveau volontaire - vérifier ses ressources
                        if node.resources.ram_gb < self._ram_needed:
                            logging.warning(
                                f"[REFUS] Volontaire MAC={mac} rejeté (RAM disponible = {node.resources.ram_gb} GB < requise = {self._ram_needed:.2f} GB)"
                            )
                            continue
                        
                        self._volunteers[mac] = node
                        logging.info(
                            f"Volontaire ajouté : MAC={mac}  IP={node.current_ip}  "
                            f"CPU={node.resources.cpu_cores} cores  "
                            f"RAM={node.resources.ram_gb}GB  "
                            f"Network={node.resources.network_bandwidth_mbps}Mbps"
                        )
                    else:
                        # Pour un volontaire existant, on vérifie s'il respecte toujours les exigences de RAM
                        if node.resources.ram_gb < self._ram_needed:
                            logging.warning(
                                f"[REFUS] Volontaire MAC={mac} rejeté après mise à jour (RAM disponible = {node.resources.ram_gb} GB < requise = {self._ram_needed:.2f} GB)"
                            )
                            self._volunteers.pop(mac, None)
                            if node.current_ip in self._ip_to_mac:
                                del self._ip_to_mac[node.current_ip]
                            continue

                        existing = self._volunteers[mac]
                        existing.resources  = node.resources
                        existing.current_ip = node.current_ip
                        logging.debug(f"Volontaire mis à jour : MAC={mac}  IP={node.current_ip}")

                    # Maintenir le mapping IP → MAC
                    if node.current_ip:
                        self._ip_to_mac[node.current_ip] = mac

                except Exception as e:
                    logging.error(f"Erreur parsing volontaire : {e}")

            # Retirer les volontaires absents du dernier broadcast
            departed = [mac for mac in list(self._volunteers.keys())
                        if mac not in new_macs]
            for mac in departed:
                gone_node = self._volunteers.pop(mac)
                if gone_node.current_ip and gone_node.current_ip in self._ip_to_mac:
                    del self._ip_to_mac[gone_node.current_ip]
                logging.info(f"Volontaire retiré (absent du broadcast) : {mac}")

        logging.info(f"Liste volontaires synchronisée : {len(self._volunteers)} volontaires actifs")

    def _on_send_model(self, conn: socket.socket,
                       sender_ip: str, data: dict, payload: bytes):
        """Enfile le modèle pour le destinataire.

        FIX BUG 6 (côté manager) : le volontaire envoie maintenant dest_mac ET dest_ip.
        On tente la résolution dans cet ordre :
          1. dest_mac direct (clé dans self._volunteers) — le plus fiable
          2. dest_ip via _ip_to_mac
          3. Scan des current_ip (fallback)
        Idem pour le sender : on préfère sender_mac transmis dans data.
        """
        dest_ip   = data.get("dest_ip", "")
        dest_mac_hint = data.get("dest_mac", "")
        sender_mac_hint = data.get("sender_mac", "")
        metadata  = data.get("metadata", {})

        with self._vol_lock:
            known_macs = list(self._volunteers.keys())

            #Résoudre le MAC du destinataire — priorité au MAC explicite
            dest_mac = None
            if dest_mac_hint and dest_mac_hint in self._volunteers:
                dest_mac = dest_mac_hint
            elif dest_ip in self._ip_to_mac:
                dest_mac = self._ip_to_mac[dest_ip]
            elif dest_ip in self._volunteers:
                dest_mac = dest_ip
            else:
                for mac, node in self._volunteers.items():
                    if node.current_ip == dest_ip:
                        dest_mac = mac
                        break

            if not dest_mac or dest_mac not in known_macs:
                send_message(conn, MSG_ERROR,
                             {"message": f"Destinataire inconnu : {dest_mac_hint or dest_ip}"})
                return

            #Résoudre le sender — priorité au MAC explicite
            sender_mac = None
            if sender_mac_hint and sender_mac_hint in self._volunteers:
                sender_mac = sender_mac_hint
            elif sender_ip in self._ip_to_mac:
                sender_mac = self._ip_to_mac[sender_ip]
            elif sender_ip in self._volunteers:
                sender_mac = sender_ip
            else:
                for mac, node in self._volunteers.items():
                    if node.current_ip == sender_ip:
                        sender_mac = mac
                        break

            if not sender_mac:
                sender_mac = sender_ip  # fallback sur l'IP brute

        metadata = dict(metadata or {})
        metadata["_queued_ts"] = time.time()
        with self._q_lock:
            self._queues[dest_mac].put((sender_mac, payload, metadata))

        send_message(conn, MSG_ACK,
                     {"status": "queued", "dest": dest_mac, "bytes": len(payload)})
        logging.info(
            f"Modèle en file : {sender_mac} → {dest_mac}  "
            f"({len(payload)/1024:.1f} KB)"
        )

    def _on_poll(self, conn: socket.socket, vol_ip: str, data: dict):
        """Livre les modèles en attente.

        FIX BUG 4 : La résolution du MAC était fragile après un clear().
        On utilise maintenant _resolve_mac() qui couvre tous les cas
        (MAC direct, IP dans ip_to_mac, scan des nodes) de façon cohérente.
        On garde en plus vol_ip comme dernier recours.
        """
        # Priorité : MAC explicite fourni par le volontaire > IP déclarée > IP TCP
        candidate = (
            data.get("volunteer_mac")
            or data.get("volunteer_ip")
            or vol_ip
        )

        with self._vol_lock:
            vol_mac = self._resolve_mac(candidate)
            # Dernier recours : essayer directement avec l'IP TCP de connexion
            if vol_mac is None and candidate != vol_ip:
                vol_mac = self._resolve_mac(vol_ip)

        if vol_mac is None:
            # Aucune résolution possible : répondre vide plutôt que planter
            logging.warning(f"_on_poll : impossible de résoudre {candidate} / {vol_ip}")
            send_message(conn, MSG_ACK, {"status": "empty"})
            return

        max_deliver = data.get("max_models", 5)
        items = []

        with self._q_lock:
            q = self._queues.get(vol_mac, queue.Queue())
            while not q.empty() and len(items) < max_deliver:
                items.append(q.get_nowait())

        if not items:
            send_message(conn, MSG_ACK, {"status": "empty"})
            return

        sender, payload, meta = items[0]
        remaining = len(items) - 1

        if remaining > 0:
            with self._q_lock:
                for item in items[1:]:
                    self._queues[vol_mac].put(item)

        send_message(conn, MSG_MODEL_DELIVERY,
                     {"sender_ip": sender, "n_pending": remaining, "metadata": meta},
                     payload)
        logging.info(
            f"Modèle livré : {sender} → {vol_mac}  "
            f"({len(payload)/1024:.1f} KB)  restants={remaining}"
        )
        try:
            recv_ts = time.time()
            payload_bytes = len(payload) if payload else 0
            self._stats.record_exchange(sender, vol_mac, payload_bytes,
                                        metadata=meta, delivered_ts=recv_ts)
        except Exception as e:
            logging.debug(f"Enregistrement échange échoué: {e}")

    def _on_neighbors_request(self, conn: socket.socket,
                               vol_ip: str, data: dict):
        """Retourne la liste de TOUS les autres volontaires avec leur historique
        de bande passante, en excluant le demandeur.

        FIX BUG 3 : L'ancienne implémentation incluait le demandeur dans la liste
        renvoyée. Le volontaire calculait alors XOR(soi-même, soi-même) = 0, ce qui
        faisait de lui-même son voisin le plus proche → aucun échange réel.

        On identifie le MAC du demandeur et on l'exclut de la réponse.
        Si on ne peut pas l'identifier (premier contact), on renvoie tout
        et laisse le volontaire filtrer lui-même via son propre MAC.
        """
        # Identifier le demandeur
        requester_mac_hint = data.get("volunteer_mac") or data.get("volunteer_ip")
        with self._vol_lock:
            requester_mac = (
                self._resolve_mac(requester_mac_hint) if requester_mac_hint else None
            ) or self._resolve_mac(vol_ip)

            vol_list = []
            for mac, node in self._volunteers.items():
                #FIX : exclure le demandeur pour éviter XOR = 0
                if mac == requester_mac:
                    continue
                node_dict = node.to_dict()
                node_dict["bandwidth_history"] = list(self._neighbor_rewards[mac])
                vol_list.append(node_dict)

        send_message(conn, MSG_NEIGHBORS_RESPONSE, {"volunteers": vol_list})
        logging.info(
            f"[Demande voisins] Réponse à {requester_mac or vol_ip} : "
            f"{len(vol_list)} voisins potentiels (demandeur exclu)"
        )

    def _on_stats_push(self, sender_ip: str, data: dict):
        """Enregistre le résumé de stats poussé par un volontaire."""
        vol_ip  = data.get("volunteer_ip", "unknown")
        summary = data.get("summary", {})
        if summary:
            with self._vol_lock:
                vol_mac = self._resolve_mac(vol_ip)
                if vol_mac is None:
                    vol_mac = self._resolve_mac(sender_ip)
                
                node = self._volunteers.get(vol_mac) if vol_mac else None
                real_ip = node.current_ip if node else (sender_ip or vol_ip)

            if vol_mac:
                current_summary = dict(summary)
                current_summary["_last_update"] = time.time()
                reward = self._compute_bandwidth_reward(vol_mac, current_summary)
                self._neighbor_rewards[vol_mac].append(reward)
                self._neighbor_stats_prev[vol_mac] = current_summary
                logging.debug(
                    f"Reward mis à jour pour {vol_mac}: {reward:.2f} Mbps "
                    f"(stats push de {sender_ip})"
                )
            self._stats.update_volunteer_summary(real_ip, summary)
            self._stats.save()
            logging.debug(f"Stats reçues de {real_ip} (round {summary.get('current_round', '?')})")

    def _on_stats_request(self, conn: socket.socket):
        """Répond avec le résumé global des stats."""
        send_message(conn, MSG_STATS_RESPONSE, self._stats.summary())

    #Reporter périodique

    def _stats_reporter(self):
        while self._running:
            time.sleep(STATS_PRINT_INTERVAL)
            self._stats.print_summary()

    #Arrêt

    def _shutdown(self, *_):
        logging.info("Arrêt du manager…")
        self._stats.print_summary()
        self._stats.save()
        self._running = False
        sys.exit(0)


if __name__ == "__main__":
    Manager().run()