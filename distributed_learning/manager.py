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
from typing import Dict, List, Optional

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else ".")

from src.config import (
    MANAGER_HOST, MANAGER_PORT,
    K_NEIGHBORS, SOCKET_TIMEOUT,
    MAX_CONNECTIONS, MAX_RETRIES, RETRY_DELAY,
    STATS_PRINT_INTERVAL, LOG_LEVEL, STATS_DIR,
    SW_UCB_WINDOW, SW_UCB_CONFIDENCE,
)
from src.protocol import (
    send_message, receive_message,
    MSG_VOLUNTEER_LIST, MSG_SEND_MODEL, MSG_POLL_MODELS,
    MSG_MODEL_DELIVERY, MSG_REQUEST_NEIGHBORS, MSG_NEIGHBORS_RESPONSE,
    MSG_STATS_REQUEST, MSG_STATS_RESPONSE, MSG_STATS_PUSH,
    MSG_ACK, MSG_ERROR,
)
from src.topology import get_k_nearest_neighbors
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
        self._neighbor_request_count = 0
        self._running = True

    # ─── Entrée principale ────────────────────────────────────────────────────

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

    # ─── Serveur TCP ──────────────────────────────────────────────────────────

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
                self._on_stats_push(data)
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

    def _score_neighbors(self, neighbor_macs: List[str]) -> List[str]:
        """Classe les voisins les plus proches en priorité SW-UCB sur la bande passante."""
        self._neighbor_request_count += 1
        t = max(1, self._neighbor_request_count)
        scored = []

        for mac in neighbor_macs:
            history = self._neighbor_rewards[mac]
            if history:
                avg = sum(history) / len(history)
                count = len(history)
                bonus = SW_UCB_CONFIDENCE * math.sqrt(2 * math.log(t) / count)
                score = avg + bonus
            else:
                node = self._volunteers.get(mac)
                base = node.resources.network_bandwidth_mbps if node is not None else 0.0
                score = base + SW_UCB_CONFIDENCE * math.sqrt(2 * math.log(t))
            scored.append((mac, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return [mac for mac, _ in scored]

    # ─── Handlers ─────────────────────────────────────────────────────────────

    def _on_volunteer_list(self, data: dict):
        """Met à jour la liste des volontaires (message du coordinateur)."""
        new_list = data.get("volunteers", [])
        with self._vol_lock:
            self._volunteers.clear()
            self._ip_to_mac.clear()
            
            for vol_data in new_list:
                try:
                    node = VolunteerNode.from_dict(vol_data)
                    mac = node.mac_address
                    self._volunteers[mac] = node
                    
                    # Maintenir le mapping IP → MAC
                    if node.current_ip:
                        self._ip_to_mac[node.current_ip] = mac
                    
                    logging.info(
                        f"Volontaire mis à jour: MAC={mac}  IP={node.current_ip}  "
                        f"CPU={node.resources.cpu_cores} cores  "
                        f"RAM={node.resources.ram_gb}GB  "
                        f"Network={node.resources.network_bandwidth_mbps}Mbps"
                    )
                except Exception as e:
                    logging.error(f"Erreur parsing volontaire : {e}")
        
        logging.info(f"Liste volontaires mise à jour : {len(self._volunteers)} volontaires")

    def _on_send_model(self, conn: socket.socket,
                       sender_ip: str, data: dict, payload: bytes):
        """Enfile le modèle pour le destinataire."""
        dest_ip  = data.get("dest_ip", "")
        metadata = data.get("metadata", {})

        with self._vol_lock:
            # Obtenir le MAC du destinataire
            known_macs = list(self._volunteers.keys())
            
            # Si dest_ip est une IP, chercher son MAC
            dest_mac = None
            if dest_ip in self._ip_to_mac:
                dest_mac = self._ip_to_mac[dest_ip]
            elif dest_ip in self._volunteers:
                # C'est peut-être un MAC directement
                dest_mac = dest_ip
            else:
                # Chercher en comparant les IPs
                for mac, node in self._volunteers.items():
                    if node.current_ip == dest_ip:
                        dest_mac = mac
                        break
            
            if not dest_mac or dest_mac not in known_macs:
                send_message(conn, MSG_ERROR,
                             {"message": f"Destinataire inconnu : {dest_ip}"})
                return
            
            # Également résoudre l'IP du sender
            sender_mac = None
            if sender_ip in self._ip_to_mac:
                sender_mac = self._ip_to_mac[sender_ip]
            elif sender_ip in self._volunteers:
                sender_mac = sender_ip
            else:
                for mac, node in self._volunteers.items():
                    if node.current_ip == sender_ip:
                        sender_mac = mac
                        break
            
            if not sender_mac:
                sender_mac = sender_ip  # Utiliser l'IP comme fallback

        with self._q_lock:
            self._queues[dest_mac].put((sender_mac, payload, metadata))

        self._stats.record_exchange(sender_ip, dest_ip, len(payload))
        send_message(conn, MSG_ACK,
                     {"status": "queued", "dest": dest_mac, "bytes": len(payload)})
        logging.info(
            f"Modèle en file : {sender_mac} → {dest_mac}  "
            f"({len(payload)/1024:.1f} KB)"
        )

    def _on_poll(self, conn: socket.socket, vol_ip: str, data: dict):
        """Livre les modèles en attente pour vol_ip.
        Supporte à la fois les IP et les MAC.
        """
        # Obtenir le MAC du volontaire
        vol_mac = data.get("volunteer_mac") or data.get("volunteer_ip", vol_ip)
        
        with self._vol_lock:
            # Si c'est une IP, chercher le MAC correspondant
            if vol_mac in self._ip_to_mac:
                vol_mac = self._ip_to_mac[vol_mac]
            elif vol_mac not in self._volunteers:
                # Chercher en comparant les IPs
                for mac, node in self._volunteers.items():
                    if node.current_ip == vol_mac or node.current_ip == vol_ip:
                        vol_mac = mac
                        break
        
        max_deliver = data.get("max_models", 5)
        items = []

        with self._q_lock:
            q = self._queues.get(vol_mac, queue.Queue())
            while not q.empty() and len(items) < max_deliver:
                items.append(q.get_nowait())

        if not items:
            send_message(conn, MSG_ACK, {"status": "empty"})
            return

        # Livraison du premier
        sender, payload, meta = items[0]
        remaining = len(items) - 1

        # Remettre les extras dans la file
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

    def _on_neighbors_request(self, conn: socket.socket,
                              vol_ip: str, data: dict):
        """Calcule et retourne les k voisins XOR du volontaire.
        Supporte à la fois les IP et les MAC comme identifiants.
        """
        k = data.get("k", K_NEIGHBORS)
        req_ip_or_mac = data.get("volunteer_ip") or data.get("volunteer_mac", vol_ip)

        with self._vol_lock:
            all_macs = list(self._volunteers.keys())
            
            # Si c'est une IP, chercher le MAC
            req_mac = req_ip_or_mac
            if req_mac in self._ip_to_mac:
                req_mac = self._ip_to_mac[req_mac]
            elif req_mac not in self._volunteers:
                # Chercher en comparant les IPs
                for mac, node in self._volunteers.items():
                    if node.current_ip == req_ip_or_mac:
                        req_mac = mac
                        break

            if req_mac not in self._volunteers:
                send_message(conn, MSG_NEIGHBORS_RESPONSE, {"neighbors": []})
                logging.warning(f"Neighbors request inconnu : {req_ip_or_mac}")
                return

            neighbors = get_k_nearest_neighbors(req_mac, all_macs, k)
            ordered = self._score_neighbors(neighbors)
            neighbors_info = [self._volunteers[mac].to_dict() for mac in ordered]

        send_message(conn, MSG_NEIGHBORS_RESPONSE, {"neighbors": neighbors_info})
        logging.debug(f"Voisins de {req_mac} : {ordered}")

    def _on_stats_push(self, data: dict):
        """Enregistre le résumé de stats poussé par un volontaire."""
        vol_ip = data.get("volunteer_ip", "unknown")
        summary = data.get("summary", {})
        if summary:
            with self._vol_lock:
                vol_mac = self._resolve_mac(vol_ip)
            if vol_mac:
                current_summary = dict(summary)
                current_summary["_last_update"] = time.time()
                reward = self._compute_bandwidth_reward(vol_mac, current_summary)
                self._neighbor_rewards[vol_mac].append(reward)
                self._neighbor_stats_prev[vol_mac] = current_summary
                logging.debug(
                    f"Reward mis à jour pour {vol_mac}: {reward:.2f} Mbps "
                    f"(stats push de {vol_ip})"
                )
            self._stats.update_volunteer_summary(vol_ip, summary)
            logging.debug(f"Stats reçues de {vol_ip} (round {summary.get('current_round', '?')})")

    def _on_stats_request(self, conn: socket.socket):
        """Répond avec le résumé global des stats."""
        send_message(conn, MSG_STATS_RESPONSE, self._stats.summary())

    # ─── Reporter périodique ──────────────────────────────────────────────────

    def _stats_reporter(self):
        while self._running:
            time.sleep(STATS_PRINT_INTERVAL)
            self._stats.print_summary()

    # ─── Arrêt ────────────────────────────────────────────────────────────────

    def _shutdown(self, *_):
        logging.info("Arrêt du manager…")
        self._stats.print_summary()
        self._stats.save()
        self._running = False
        sys.exit(0)


if __name__ == "__main__":
    Manager().run()
