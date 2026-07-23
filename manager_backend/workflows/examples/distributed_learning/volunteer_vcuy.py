"""
Adaptateur volontaire VC-UY : exécution gossip sans coordinateur DL ni Docker.
Le bridge Manager VC-UY alimente le manager DL avec la liste des volontaires.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def run_volunteer(
    *,
    volunteer_id: int,
    n_volunteers: int,
    manager_host: str,
    manager_port: int = 9001,
    my_ip: str = "",
    cpu_cores: Optional[int] = None,
    ram_gb: Optional[float] = None,
    network_bandwidth_mbps: Optional[float] = None,
) -> None:
    """Lance le cycle gossip en mode VC-UY (sans heartbeat coordinateur DL)."""
    os.environ["MANAGER_PORT"] = str(manager_port)
    os.environ["MANAGER_HOST"] = str(manager_host)

    import volunteer_core as _core
    # Recharger config après setenv (évite MANAGER_PORT figé à l'import)
    import importlib
    import src.config as _cfg
    importlib.reload(_cfg)
    importlib.reload(_core)

    vol = _core.Volunteer(
        volunteer_id=volunteer_id,
        n_volunteers=n_volunteers,
        coordinator_host=manager_host,
        manager_host=manager_host,
        manager_port=manager_port,
        my_ip=my_ip,
        cpu_cores=cpu_cores,
        ram_gb=ram_gb,
        network_bandwidth_mbps=network_bandwidth_mbps,
    )

    skip_coord = os.getenv("VCUY_SKIP_DL_COORDINATOR", "").lower() in ("1", "true", "yes")
    if skip_coord:
        logging.info(
            "[Volontaire %s] Mode VC-UY — pas de coordinateur DL (bridge Manager actif)",
            volunteer_id,
        )
        _run_without_coordinator(vol, _core)
    else:
        vol.run()


def _run_without_coordinator(vol, _core) -> None:
    """Boucle gossip identique à volunteer_core.run() sans thread heartbeat."""
    time.sleep(5)

    max_rounds = _core.MAX_ROUNDS
    gossip_interval = _core.GOSSIP_INTERVAL

    while vol._running and (max_rounds <= 0 or vol.round_num < max_rounds):
        try:
            vol._run_gossip_round()
        except Exception as exc:
            logging.exception("[Volontaire %s] Erreur round : %s", vol.vol_id, exc)
        time.sleep(gossip_interval)

    vol._shutdown()
