"""
Service d'orchestration pour workflows DISTRIBUTED_LEARNING.

Démarre le manager DL (TCP gossip) et le bridge VC-UY → manager DL
(synchronisation de la liste des volontaires depuis la présence Manager).
Sans Docker — les volontaires exécutent run_volunteer_vcuy.py via vc-uyr.
"""
from __future__ import annotations

import hashlib
import logging
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_DL_ROOT = Path(__file__).resolve().parents[2].parent / "distributed_learning" / "distributed_learning"
_EXAMPLES = Path(__file__).resolve().parent / "examples" / "distributed_learning"

# Registre global workflow_id → {manager_thread, bridge_thread, port, stop_event}
_ACTIVE: Dict[str, dict] = {}
_LOCK = threading.Lock()


def _framework_paths() -> list[str]:
    paths = []
    if _EXAMPLES.is_dir():
        paths.append(str(_EXAMPLES))
    if _DL_ROOT.is_dir():
        paths.append(str(_DL_ROOT))
    return paths


def _pick_port(workflow_id: str, base: int = 9100) -> int:
    digest = int(hashlib.sha256(str(workflow_id).encode()).hexdigest()[:6], 16)
    port = base + (digest % 800)
    for attempt in range(20):
        candidate = port + attempt
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("0.0.0.0", candidate))
                return candidate
            except OSError:
                continue
    return base


def _vcuy_volunteer_to_dl_node(vol: dict, index: int) -> dict:
    """Convertit un volontaire VC-UY en dict VolunteerNode DL."""
    vid = str(vol.get("volunteer_id") or index)
    resources = vol.get("resources") or {}
    ip = resources.get("ip_address") or "127.0.0.1"
    ram_mb = int(resources.get("memory_mb") or 1024)
    cpu = int(resources.get("cpu_cores") or 1)

    mac_hash = hashlib.sha256(vid.encode()).hexdigest()[:12].upper()
    mac = ":".join(mac_hash[i : i + 2] for i in range(0, 12, 2))

    return {
        "mac_address": mac,
        "current_ip": ip,
        "last_heartbeat": time.time(),
        "resources": {
            "cpu_cores": cpu,
            "cpu_freq_ghz": 2.0,
            "ram_gb": max(0.5, ram_mb / 1024.0),
            "network_bandwidth_mbps": float(resources.get("network_mbps") or 50.0),
            "battery": 100.0,
            "disk_free_gb": max(1.0, int(resources.get("disk_space_mb") or 10240) / 1024.0),
            "cpu_load": 0.0,
        },
    }


def _bridge_loop(workflow_id: str, manager_host: str, manager_port: int, stop: threading.Event) -> None:
    for path in _framework_paths():
        if path not in sys.path:
            sys.path.insert(0, path)

    from src.protocol import MSG_VOLUNTEER_LIST, send_message, receive_message

    while not stop.is_set():
        try:
            volunteers = []
            try:
                from volunteers.presence import get_online_volunteers_data

                volunteers = get_online_volunteers_data()
            except Exception as exc:
                logger.debug("Bridge DL: présence indisponible: %s", exc)

            dl_nodes = [_vcuy_volunteer_to_dl_node(v, i) for i, v in enumerate(volunteers)]
            if dl_nodes:
                conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                conn.settimeout(10)
                conn.connect((manager_host, manager_port))
                send_message(conn, MSG_VOLUNTEER_LIST, {"volunteers": dl_nodes})
                try:
                    receive_message(conn)
                except Exception:
                    pass
                conn.close()
                logger.debug(
                    "Bridge DL workflow %s: %s volontaire(s) synchronisé(s)",
                    workflow_id,
                    len(dl_nodes),
                )
        except Exception as exc:
            logger.warning("Bridge DL workflow %s: %s", workflow_id, exc)

        stop.wait(5)


def _run_manager(manager_port: int, stop: threading.Event) -> None:
    for path in _framework_paths():
        if path not in sys.path:
            sys.path.insert(0, path)

    os.environ["MANAGER_HOST"] = "0.0.0.0"
    os.environ["MANAGER_PORT"] = str(manager_port)

    try:
        import importlib
        import manager_core as _mgr_mod

        # Recharger à chaque workflow pour appliquer les hotfixes docker cp
        # sans redémarrer tout le process Django.
        importlib.reload(_mgr_mod)
        Manager = _mgr_mod.Manager

        mgr = Manager()
        mgr._running = True
        threads = [
            threading.Thread(target=mgr._listen, daemon=True, name="dl-manager-listener"),
            threading.Thread(target=mgr._stats_reporter, daemon=True, name="dl-manager-stats"),
        ]
        for thread in threads:
            thread.start()
        logger.info("Manager DL démarré sur 0.0.0.0:%s", manager_port)
        while not stop.is_set():
            stop.wait(1)
        mgr._running = False
    except Exception as exc:
        logger.error("Manager DL arrêté avec erreur: %s", exc, exc_info=True)


def start_for_workflow(workflow, public_host: Optional[str] = None) -> int:
    """
    Démarre manager DL + bridge pour un workflow.
    Retourne le port TCP du manager DL.
    """
    workflow_id = str(workflow.id)
    with _LOCK:
        if workflow_id in _ACTIVE:
            return _ACTIVE[workflow_id]["port"]

        port = _pick_port(workflow_id)
        stop = threading.Event()

        mgr_thread = threading.Thread(
            target=_run_manager,
            args=(port, stop),
            daemon=True,
            name=f"dl-mgr-{workflow_id[:8]}",
        )
        mgr_thread.start()
        time.sleep(1.5)

        host = public_host or "127.0.0.1"
        bridge_thread = threading.Thread(
            target=_bridge_loop,
            args=(workflow_id, host, port, stop),
            daemon=True,
            name=f"dl-bridge-{workflow_id[:8]}",
        )
        bridge_thread.start()

        _ACTIVE[workflow_id] = {
            "port": port,
            "host": host,
            "stop": stop,
            "manager_thread": mgr_thread,
            "bridge_thread": bridge_thread,
        }
        logger.info(
            "Service DL workflow %s: manager=%s:%s",
            workflow_id,
            host,
            port,
        )
        return port


def stop_for_workflow(workflow_id: str) -> None:
    with _LOCK:
        entry = _ACTIVE.pop(str(workflow_id), None)
    if not entry:
        return
    entry["stop"].set()
    logger.info("Service DL arrêté pour workflow %s", workflow_id)


def get_manager_endpoint(workflow_id: str) -> Optional[tuple[str, int]]:
    with _LOCK:
        entry = _ACTIVE.get(str(workflow_id))
    if not entry:
        return None
    return entry["host"], entry["port"]
