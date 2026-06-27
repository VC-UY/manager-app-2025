#!/usr/bin/env python3
"""
Nœud Coordinateur
─────────────────
Rôle :
  1. Écoute les heartbeats des volontaires (connexion TCP persistante).
  2. Maintient la liste des volontaires actifs.
  3. Diffuse cette liste au Manager toutes les ~5 secondes.
  4. Supprime les volontaires silencieux depuis plus de HEARTBEAT_TIMEOUT secondes.

Démarrage :
  python coordinator.py

Variables d'environnement clés :
  COORDINATOR_HOST        IP d'écoute (défaut 0.0.0.0)
  COORDINATOR_PORT        Port d'écoute (défaut 9000)
  MANAGER_EXTERNAL_HOST   IP publique du manager (défaut 127.0.0.1)
  MANAGER_PORT            Port du manager (défaut 9001)
  HEARTBEAT_TIMEOUT       Secondes d'inactivité avant expulsion (défaut 35)
"""
import logging
import signal
import socket
import sys
import threading
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else ".")

from src.config import (
    COORDINATOR_HOST, COORDINATOR_PORT,
    MANAGER_EXTERNAL_HOST, MANAGER_PORT,
    HEARTBEAT_INTERVAL, HEARTBEAT_TIMEOUT,
    SOCKET_TIMEOUT, MAX_CONNECTIONS,
    MAX_RETRIES, RETRY_DELAY, LOG_LEVEL,
)
from src.protocol import (
    send_message, receive_message,
    MSG_HEARTBEAT, MSG_ACK,
    MSG_VOLUNTEER_LIST, MSG_DISCONNECT,
)
from src.volunteer_node import VolunteerNode

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  [COORDINATEUR]  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Coordinator:
    def __init__(self):
        self._volunteers: dict[str, VolunteerNode] = {}   # mac → VolunteerNode
        self._lock     = threading.Lock()
        self._running  = True

    # ─── Entrée principale ────────────────────────────────────────────────────

    def run(self):
        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        threads = [
            threading.Thread(target=self._listen_volunteers, daemon=True, name="listener"),
            threading.Thread(target=self._purge_inactive,    daemon=True, name="purge"),
            threading.Thread(target=self._broadcast_loop,    daemon=True, name="broadcast"),
        ]
        for t in threads:
            t.start()

        logging.info(f"Coordinateur démarré — écoute {COORDINATOR_HOST}:{COORDINATOR_PORT}")
        logging.info(f"Manager cible : {MANAGER_EXTERNAL_HOST}:{MANAGER_PORT}")

        while self._running:
            time.sleep(1)

    # ─── Écoute volontaires ───────────────────────────────────────────────────

    def _listen_volunteers(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.settimeout(1.0)
        # On lie à "0.0.0.0" pour écouter sur toutes les interfaces et éviter l'erreur OSError [Errno 99]
        srv.bind(("0.0.0.0", COORDINATOR_PORT))
        srv.listen(MAX_CONNECTIONS)

        while self._running:
            try:
                conn, addr = srv.accept()
                conn.settimeout(SOCKET_TIMEOUT)
                threading.Thread(
                    target=self._handle_volunteer,
                    args=(conn, addr[0]),
                    daemon=True,
                ).start()
            except socket.timeout:
                pass
            except OSError as exc:
                if self._running:
                    logging.error(f"Erreur accept : {exc}")
        srv.close()

    def _handle_volunteer(self, conn: socket.socket, ip: str):
        # Le volontaire déclare son MAC et ses ressources dans le heartbeat initial
        # FIX BUG 1 : mac_address doit être mis à jour dès le premier heartbeat valide.
        # Dans la version originale, `mac_address` restait None car on utilisait
        # la variable locale `mac` dans la boucle sans jamais assigner mac_address.
        # Résultat : le bloc `finally` ne retirait JAMAIS le volontaire du dictionnaire.
        mac_address = None
        node = None

        try:
            while self._running:
                msg_type, data, _ = receive_message(conn)

                if msg_type == MSG_HEARTBEAT:
                    mac = data.get("mac_address", "")

                    if not mac:
                        logging.warning(f"Heartbeat sans MAC depuis {ip}, ignoré")
                        continue

                    # ✅ FIX : assigner mac_address dès qu'on a un MAC valide
                    if mac_address is None:
                        mac_address = mac

                    # Créer ou mettre à jour le volontaire
                    with self._lock:
                        if mac not in self._volunteers:
                            # Nouveau volontaire
                            try:
                                node = VolunteerNode.from_dict(data)
                                node.current_ip = ip
                                self._volunteers[mac] = node
                                logging.info(
                                    f"Volontaire connecté : MAC={mac}  "
                                    f"IP={ip}  "
                                    f"Ressources: CPU={node.resources.cpu_cores} cores, "
                                    f"RAM={node.resources.ram_gb}GB, "
                                    f"Network={node.resources.network_bandwidth_mbps}Mbps  "
                                    f"(total : {len(self._volunteers)})"
                                )
                            except Exception as e:
                                logging.error(f"Impossible de parser volontaire {mac}: {e}")
                                mac_address = None  # annuler si parsing échoue
                                continue
                        else:
                            # Mise à jour du volontaire existant
                            node = self._volunteers[mac]
                            node.current_ip = ip
                            try:
                                updated = VolunteerNode.from_dict(data)
                                node.resources = updated.resources
                                logging.debug(f"Ressources mises à jour pour {mac}")
                            except Exception as e:
                                logging.warning(f"Erreur mise à jour ressources {mac}: {e}")

                        node.last_heartbeat = time.time()

                    send_message(conn, MSG_ACK, {"ts": time.time(), "mac": mac})

                elif msg_type == MSG_DISCONNECT:
                    break

        except (ConnectionError, OSError, EOFError) as exc:
            logging.info(f"Volontaire {mac_address or ip} déconnecté : {exc}")
        except Exception as exc:
            logging.warning(f"Erreur volontaire {mac_address or ip} : {exc}")
        finally:
            # ✅ FIX : mac_address est maintenant correctement assigné,
            # donc le volontaire sera bien retiré du dictionnaire à la déconnexion.
            if mac_address:
                with self._lock:
                    self._volunteers.pop(mac_address, None)
                logging.info(
                    f"Volontaire retiré : {mac_address}  "
                    f"(total : {len(self._volunteers)})"
                )
            else:
                logging.info(f"Volontaire déconnecté {ip} (aucun MAC enregistré)")
            conn.close()

    # ─── Nettoyage ────────────────────────────────────────────────────────────

    def _purge_inactive(self):
        while self._running:
            time.sleep(HEARTBEAT_TIMEOUT // 2)
            now = time.time()
            with self._lock:
                stale = [mac for mac, node in self._volunteers.items()
                         if now - node.last_heartbeat > HEARTBEAT_TIMEOUT]
                for mac in stale:
                    del self._volunteers[mac]
                    logging.warning(f"Volontaire expiré (timeout) : {mac}")

    # ─── Diffusion vers le Manager ────────────────────────────────────────────

    def _broadcast_loop(self):
        while self._running:
            time.sleep(5)
            with self._lock:
                vol_list = [node.to_dict() for node in self._volunteers.values()]
            if not vol_list:
                continue
            self._send_to_manager(vol_list)

    def _send_to_manager(self, vol_list: list):
        for attempt in range(MAX_RETRIES):
            try:
                conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                conn.settimeout(SOCKET_TIMEOUT)
                conn.connect((MANAGER_EXTERNAL_HOST, MANAGER_PORT))
                send_message(conn, MSG_VOLUNTEER_LIST, {"volunteers": vol_list})
                conn.close()
                logging.debug(f"Liste envoyée au manager : {len(vol_list)} volontaires")
                return
            except Exception as exc:
                logging.warning(f"Envoi manager échoué (essai {attempt + 1}) : {exc}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)

    # ─── Arrêt ────────────────────────────────────────────────────────────────

    def _shutdown(self, *_):
        logging.info("Arrêt du coordinateur…")
        self._running = False
        sys.exit(0)


if __name__ == "__main__":
    Coordinator().run()