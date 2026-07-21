#!/usr/bin/env python3
"""
Point d'entrée vc-uyr pour l'apprentissage distribué gossip (sans Docker).
Lit dl_config.json, configure l'environnement, exécute le nœud volontaire,
puis écrit les artefacts dans vc_OUTPUT.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [DL-VCUY]  %(levelname)-8s  %(message)s",
)


def _load_config() -> dict:
    candidates = [
        SCRIPT_DIR / "dl_config.json",
    ]
    vc_input = os.environ.get("vc_INPUT") or os.environ.get("INPUT_DIR")
    if vc_input:
        candidates.insert(0, Path(vc_input) / "dl_config.json")
    for path in candidates:
        if path.is_file():
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
    raise FileNotFoundError("dl_config.json introuvable dans le bundle ou vc_INPUT")


def _apply_config(cfg: dict) -> Path:
    output_dir = Path(
        os.environ.get("vc_OUTPUT")
        or os.environ.get("OUTPUT_DIR")
        or (SCRIPT_DIR / "output")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_dir = output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    os.environ["VCUY_SKIP_DL_COORDINATOR"] = "1"
    os.environ["STATS_DIR"] = str(stats_dir)

    env_map = cfg.get("env") or {}
    for key, value in env_map.items():
        os.environ[str(key)] = str(value)

    slot = int(cfg.get("volunteer_slot", 0))
    os.environ.setdefault("VOLUNTEER_ID", str(slot))
    os.environ.setdefault("N_VOLUNTEERS", str(cfg.get("n_volunteers", 3)))
    os.environ.setdefault("MANAGER_HOST", str(cfg.get("manager_host", "127.0.0.1")))
    os.environ.setdefault("MANAGER_PORT", str(cfg.get("manager_port", 9101)))
    os.environ.setdefault("MY_IP", str(cfg.get("my_ip", "")))

    return output_dir


def _export_results(output_dir: Path, slot: int) -> None:
    stats_root = output_dir / "stats"
    vol_dir = stats_root / f"volunteer_{slot}"
    if vol_dir.is_dir():
        dest = output_dir / f"volunteer_{slot}"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(vol_dir, dest)

    summary = {
        "volunteer_slot": slot,
        "stats_dir": str(stats_root),
        "outputs": [p.name for p in output_dir.iterdir() if p.is_file()],
    }
    with open(output_dir / "dl_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    # Modèle final si présent
    for name in ("model_final.pt", "final_model.pt"):
        candidate = vol_dir / name if vol_dir.is_dir() else None
        if candidate and candidate.is_file():
            shutil.copy2(candidate, output_dir / "model_final.pt")
            break


def main() -> int:
    cfg = _load_config()
    output_dir = _apply_config(cfg)
    slot = int(cfg.get("volunteer_slot", 0))

    from volunteer_vcuy import run_volunteer

    run_volunteer(
        volunteer_id=slot,
        n_volunteers=int(cfg.get("n_volunteers", 3)),
        manager_host=str(cfg.get("manager_host", "127.0.0.1")),
        manager_port=int(cfg.get("manager_port", 9101)),
        my_ip=str(cfg.get("my_ip", "")),
    )

    _export_results(output_dir, slot)
    logging.info("Volontaire DL terminé — sorties dans %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
