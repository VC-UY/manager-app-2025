#!/usr/bin/env python3
"""
Lance le système complet sur une seule machine (idéal pour tester avant déploiement réel).
Démarre : Manager → Coordinateur → N Volontaires (avec IPs simulées).

Usage :
    python launch_experiment.py --n-volunteers 5
    python launch_experiment.py --n-volunteers 3 --dataset cifar10 --compression sparsification
    python launch_experiment.py --n-volunteers 4 --partition non-iid --k 2

Les IPs simulées sont du type 10.0.0.1, 10.0.0.2 … pour permettre
le calcul XOR réel entre volontaires même en test local.
"""

import argparse
import os
import signal
import subprocess
import sys
import time

PROCS: list = []


def launch(name: str, cmd: list, env_extra: dict, log_dir: str = "logs"):
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(log_dir, f"{name}.log")
    env = {**os.environ, **env_extra}
    with open(logfile, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
    PROCS.append((name, proc))
    print(f"  [OK] {name:<28s}  PID={proc.pid:<6d}  → logs/{name}.log")
    return proc


def stop_all():
    print("\nArrêt de tous les processus…")
    for name, proc in PROCS:
        try:
            proc.terminate()
        except Exception:
            pass
    time.sleep(2)
    for name, proc in PROCS:
        try:
            proc.kill()
        except Exception:
            pass
    print("Tous les processus arrêtés.")


def wait(secs: float, label: str):
    print(f"  Attente {secs:.0f}s ({label})…")
    time.sleep(secs)


def main():
    parser = argparse.ArgumentParser(
        description="Lancement local du système d'apprentissage distribué frugal"
    )
    parser.add_argument("--n-volunteers",     type=int,   default=3,
                        help="Nombre de volontaires (défaut: 3)")
    parser.add_argument("--dataset",          default="mnist",
                        choices=["mnist", "cifar10"],
                        help="Dataset (défaut: mnist)")
    parser.add_argument("--partition",        default="iid",
                        choices=["iid", "non-iid"],
                        help="Partition des données (défaut: iid)")
    parser.add_argument("--compression",      default="quantization",
                        choices=["quantization", "sparsification", "none"],
                        help="Méthode de compression (défaut: quantization)")
    parser.add_argument("--k",                type=int,   default=3,
                        help="Nombre de voisins XOR par nœud (défaut: 3)")
    parser.add_argument("--gossip-interval",  type=int,   default=30,
                        help="Secondes entre rounds gossip (défaut: 30)")
    parser.add_argument("--local-epochs",     type=int,   default=3,
                        help="Epochs d'entraînement local par round (défaut: 3)")
    parser.add_argument("--max-rounds",       type=int,   default=15,
                        help="Nombre maximum de rounds (défaut: 30, 0 = sans limite)")
    parser.add_argument("--sparsity",         type=float, default=0.05,
                        help="Ratio top-k pour sparsification (défaut: 0.05)")
    parser.add_argument("--bits",             type=int,   default=8,
                        help="Bits pour quantification (défaut: 8)")
    parser.add_argument("--fake-ip-base",     default="10.0.0.",
                        help="Préfixe IP simulé (défaut: 10.0.0.)")
    args = parser.parse_args()

    python = sys.executable
    os.makedirs("logs",    exist_ok=True)
    os.makedirs("results", exist_ok=True)

    signal.signal(signal.SIGINT,  lambda s, f: (stop_all(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda s, f: (stop_all(), sys.exit(0)))

    # Variables d'environnement partagées
    base_env = {
        "MANAGER_HOST":          "127.0.0.1",
        "MANAGER_PORT":          "9001",
        "MANAGER_EXTERNAL_HOST": "127.0.0.1",
        "COORDINATOR_HOST":      "127.0.0.1",
        "COORDINATOR_PORT":      "9000",
        "COORDINATOR_EXTERNAL_HOST": "127.0.0.1",
        "K_NEIGHBORS":           str(args.k),
        "GOSSIP_INTERVAL":       str(args.gossip_interval),
        "GOSSIP_FANOUT":         "1",
        "LOCAL_EPOCHS":          str(args.local_epochs),
        "DATASET":               args.dataset,
        "DATA_PARTITION":        args.partition,
        "COMPRESSION":           args.compression,
        "QUANTIZATION_BITS":     str(args.bits),
        "SPARSIFICATION_RATIO":  str(args.sparsity),
        "MAX_ROUNDS":            str(args.max_rounds),
        "HEARTBEAT_INTERVAL":    "10",
        "HEARTBEAT_TIMEOUT":     "35",
        "SOCKET_TIMEOUT":        "30",
        "MAX_RETRIES":           "3",
        "RETRY_DELAY":           "3",
        "STATS_DIR":             "./results",
        "STATS_PRINT_INTERVAL":  "60",
        "LOG_LEVEL":             "INFO",
    }

    sep = "=" * 60
    print(sep)
    print("  LANCEMENT — SYSTÈME D'APPRENTISSAGE DISTRIBUÉ FRUGAL")
    print(sep)
    print(f"  Volontaires     : {args.n_volunteers}")
    print(f"  Dataset         : {args.dataset}  ({args.partition})")
    print(f"  Compression     : {args.compression}")
    print(f"  Voisins XOR (k) : {args.k}")
    print(f"  Gossip interval : {args.gossip_interval}s")
    print(f"  Epochs locaux   : {args.local_epochs}")
    print(f"  Max rounds      : {args.max_rounds} (0 = sans limite)")
    print(sep)

    # 1. Manager (toujours en premier — le coordinateur lui enverra des messages)
    launch("manager", [python, "manager.py"], base_env)
    wait(2, "démarrage manager")

    # 2. Coordinateur
    launch("coordinator", [python, "coordinator.py"], base_env)
    wait(3, "démarrage coordinateur")

    # 3. Volontaires avec IPs simulées pour activer la topologie XOR en local
    for i in range(args.n_volunteers):
        fake_ip = f"{args.fake_ip_base}{i + 1}"
        launch(
            f"volunteer_{i}",
            [
                python, "volunteer.py",
                "--id",           str(i),
                "--n-volunteers", str(args.n_volunteers),
                "--coordinator",  "127.0.0.1",
                "--manager",      "127.0.0.1",
                "--my-ip",        fake_ip,
            ],
            base_env,
        )
        time.sleep(0.8)

    print()
    print(f"  Système opérationnel avec {args.n_volunteers} volontaires.")
    print(f"  Stats : ./results/  (mises à jour toutes les 60s)")
    print(f"  Monitoring en temps réel : python monitor.py")
    print(f"  Arrêt propre : Ctrl+C")
    print(sep)

    # Surveiller les processus
    try:
        while True:
            time.sleep(5)
            dead = [(n, p) for n, p in PROCS if p.poll() is not None]
            if dead:
                for name, proc in dead:
                    print(f"  [WARN] Processus terminé (code {proc.returncode}) : {name}")
                    print(f"         Voir logs/{name}.log pour détails.")
    except KeyboardInterrupt:
        stop_all()


if __name__ == "__main__":
    main()
