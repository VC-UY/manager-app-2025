#!/usr/bin/env python3
"""
Outil de monitoring en temps réel du système d'apprentissage distribué.
Se connecte au manager et affiche les statistiques périodiquement.

Usage :
    python monitor.py                            # manager local (127.0.0.1:9001)
    python monitor.py --manager 192.168.1.11    # manager distant
    python monitor.py --interval 5              # rafraîchissement toutes les 5s
    python monitor.py --no-clear                # sans effacer l'écran (pour log)
    python monitor.py --export stats_export.json  # exporter et quitter
"""

import argparse
import json
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.protocol import (
    send_message, receive_message,
    MSG_STATS_REQUEST, MSG_STATS_RESPONSE,
)


def fetch_stats(host: str, port: int, timeout: int = 10) -> dict:
    """Interroge le manager et retourne le résumé des stats."""
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    conn.settimeout(timeout)
    conn.connect((host, port))
    try:
        send_message(conn, MSG_STATS_REQUEST, {})
        msg_type, data, _ = receive_message(conn)
        if msg_type == MSG_STATS_RESPONSE:
            return data
        return {}
    finally:
        conn.close()


def bar(value: float, width: int = 20, char: str = "█") -> str:
    """Génère une barre de progression ASCII."""
    filled = int(value * width)
    return char * filled + "░" * (width - filled)


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def display(stats: dict, refresh: int, host: str, port: int):
    SEP = "═" * 70
    s = SEP
    print(s)
    print(f"  MONITORING — APPRENTISSAGE DISTRIBUÉ FRUGAL     [refresh #{refresh}]")
    print(s)

    runtime  = stats.get("runtime_s", 0)
    n_vol    = stats.get("n_active_volunteers", 0)
    n_exch   = stats.get("total_model_exchanges", 0)
    bw_mb    = stats.get("total_bytes_routed", 0) / 1024 / 1024
    thr_kbs  = stats.get("throughput_KB_per_s", 0)

    h, m = divmod(int(runtime), 3600)
    m, sec = divmod(m, 60)
    print(f"  Manager         : {host}:{port}")
    print(f"  Durée           : {h:02d}h {m:02d}m {sec:02d}s")
    print(f"  Volontaires     : {n_vol}")
    print(f"  Échanges totaux : {n_exch}")
    print(f"  BW totale       : {bw_mb:.3f} MB")
    print(f"  Débit           : {thr_kbs:.2f} KB/s")

    summaries = stats.get("volunteer_summaries", {})
    if summaries:
        print(f"\n  ─── Détail par volontaire ({len(summaries)}) ───────────────────────────────")
        hdr = f"  {'IP Volontaire':<18} {'Round':>5}  {'Acc Test':>8}  {'Précision':22}  {'BW ↑ KB':>8}  {'Durée train':>11}"
        print(hdr)
        print(f"  {'─'*18} {'─'*5}  {'─'*8}  {'─'*22}  {'─'*8}  {'─'*11}")
        for ip, vs in sorted(summaries.items()):
            acc     = vs.get("best_test_acc", vs.get("final_test_acc", 0))
            rounds  = vs.get("total_rounds", vs.get("current_round", 0))
            bw_kb   = vs.get("total_bytes_sent", 0) / 1024
            dur_s   = vs.get("total_train_duration_s", 0)
            b       = bar(min(acc, 1.0))
            print(
                f"  {ip:<18} {rounds:>5}  {acc:>7.1%}  {b}  {bw_kb:>8.1f}  {dur_s:>9.1f}s"
            )

        # Résumé agrégé
        all_acc = [vs.get("best_test_acc", vs.get("final_test_acc", 0))
                   for vs in summaries.values()]
        if all_acc:
            print(f"\n  Précision moyenne : {sum(all_acc)/len(all_acc):.2%}"
                  f"   Min: {min(all_acc):.2%}   Max: {max(all_acc):.2%}")
    else:
        print("\n  En attente des statistiques des volontaires…")
        print("  (Les stats arrivent après le premier round de gossip)")

    print(s)
    print(f"  Mis à jour : {time.strftime('%Y-%m-%d %H:%M:%S')}  |  Ctrl+C pour quitter")


def main():
    parser = argparse.ArgumentParser(description="Monitoring du système distribué")
    parser.add_argument("--manager",   default="127.0.0.1",
                        help="IP/hostname du manager (défaut: 127.0.0.1)")
    parser.add_argument("--port",      type=int, default=9001,
                        help="Port du manager (défaut: 9001)")
    parser.add_argument("--interval",  type=int, default=10,
                        help="Secondes entre rafraîchissements (défaut: 10)")
    parser.add_argument("--no-clear",  action="store_true",
                        help="Ne pas effacer l'écran (utile pour redirection)")
    parser.add_argument("--export",    default=None,
                        help="Exporter les stats dans un fichier JSON et quitter")
    args = parser.parse_args()

    # Mode export one-shot
    if args.export:
        try:
            stats = fetch_stats(args.manager, args.port)
            with open(args.export, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            print(f"Stats exportées dans : {args.export}")
        except Exception as exc:
            print(f"Erreur export : {exc}", file=sys.stderr)
            sys.exit(1)
        return

    # Mode monitoring continu
    count = 0
    consecutive_failures = 0
    print(f"Connexion au manager {args.manager}:{args.port} …")

    while True:
        try:
            stats = fetch_stats(args.manager, args.port)
            count += 1
            consecutive_failures = 0

            if not args.no_clear:
                clear_screen()
            display(stats, count, args.manager, args.port)

        except ConnectionRefusedError:
            consecutive_failures += 1
            msg = f"[{time.strftime('%H:%M:%S')}] Manager non joignable sur {args.manager}:{args.port}"
            if consecutive_failures == 1 or consecutive_failures % 6 == 0:
                print(f"{msg}. Nouvelle tentative dans {args.interval}s…")

        except socket.timeout:
            print(f"[{time.strftime('%H:%M:%S')}] Timeout manager. Nouvelle tentative…")

        except KeyboardInterrupt:
            print("\nMonitoring arrêté.")
            break

        except Exception as exc:
            print(f"[{time.strftime('%H:%M:%S')}] Erreur inattendue : {exc}")

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
