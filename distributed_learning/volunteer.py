#!/usr/bin/env python3
"""
Nœud Volontaire — version complète
===================================

Intégrations :
1. Estimation des BESOINS du modèle vs RESSOURCES disponibles (ModelProfiler + SystemProfiler)
   AVANT entraînement -> verdicts ram_ok / bandwidth_ok / timing_ok.
2. Entraînement local sécurisé (snapshot + clip_grad_norm_ + isfinite + rollback).
3. Sélection adaptative SW-UCB des voisins (un selector par volontaire).
4. Agrégation FedAvg robuste avec filtre anti-NaN.
5. Profilage avancé (RSS / PSS / USS / CPU / Throttle / ETE / IPC) à chaque round.
6. Enregistrement des coûts de communication par round (ModelProfiler).

Démarrage :
    python volunteer.py --id 0 --n-volunteers 5 \
        --coordinator 192.168.1.10 --manager 192.168.1.11 [--my-ip 192.168.1.20]
"""
import argparse
import json
import logging
import os
import signal
import socket
import sys
import threading
import time
from typing import Dict, List, Optional

import psutil
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
    SW_UCB_WINDOW, SW_UCB_CONFIDENCE,
)
from src.protocol import (
    send_message, receive_message,
    MSG_HEARTBEAT, MSG_ACK, MSG_DISCONNECT,
    MSG_SEND_MODEL, MSG_POLL_MODELS, MSG_MODEL_DELIVERY,
    MSG_REQUEST_NEIGHBORS, MSG_NEIGHBORS_RESPONSE,
    MSG_STATS_PUSH, MSG_ERROR,
)
from src.model import create_model, model_parameter_bytes
from src.dataset import load_dataset
from src.compression import compress_model, decompress_model, average_models, compression_ratio
from src.stats import StatsTracker
from src.volunteer_node import VolunteerNode, get_mac_address, get_resource_info
from src.peer_sampling import SWUCBSelector, get_peer_sample
from src.profiler import SystemProfiler, ModelProfiler
from src.advanced_profiler import AdvancedProfiler


logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  [VOLONTAIRE]    %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Volunteer:
    """Nœud volontaire d'apprentissage distribué (Gossip Learning)."""

    def __init__(self, volunteer_id: int, n_volunteers: int,
                 coordinator_host: str, manager_host: str,
                 my_ip: str = "",
                 cpu_cores: Optional[int] = None,
                 ram_gb: Optional[float] = None,
                 network_bandwidth_mbps: Optional[float] = None):
        self.vol_id = volunteer_id
        self.n_volunteers = n_volunteers
        self.coord_host = coordinator_host
        self.manager_host = manager_host
        self.my_ip = my_ip or self._detect_ip(coordinator_host)

        # ── PROFILEUR AVANCÉ : baseline AVANT chargement des données/modèle ──
        self.adv_profiler = AdvancedProfiler(sample_interval=0.5, max_samples=300)
        self.adv_profiler.capture_baseline()

        # Identité réseau
        self.my_mac = get_mac_address()
        self.resources = get_resource_info(
            cpu_cores=cpu_cores, ram_gb=ram_gb,
            network_bandwidth_mbps=network_bandwidth_mbps,
        )

        # Modèle, données, optimiseur
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_model(DATASET, NUM_CLASSES).to(self.device)
        self.train_loader, self.test_loader = load_dataset(
            dataset=DATASET, data_dir="./data",
            volunteer_id=self.vol_id, n_volunteers=self.n_volunteers,
            partition=DATA_PARTITION, batch_size=BATCH_SIZE,
        )

        # ─────────────────────────────────────────────────────────────────────
        # ESTIMATION DES BESOINS DU MODÈLE vs RESSOURCES DISPONIBLES
        # ─────────────────────────────────────────────────────────────────────
        # On instancie le ModelProfiler ET on l'utilise réellement pour
        # vérifier que la machine peut accueillir l'entraînement.
        self.model_profiler = ModelProfiler(self.model)
        self._estimate_resources_vs_needs()

        # SW-UCB : sélecteur adaptatif propre à ce volontaire
        self.selector = SWUCBSelector(
            window=SW_UCB_WINDOW,
            confidence=SW_UCB_CONFIDENCE,
            my_mac=self.my_mac,
        )
        self.round_num = 0

        # État
        self._running = True
        self._stats = StatsTracker(volunteer_ip=self.my_ip, results_dir=STATS_DIR)
        self._heartbeat_sock: Optional[socket.socket] = None

        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

        # Démarre le suivi global du ModelProfiler (compteurs cumulatifs)
        self.model_profiler.start_training_tracking()

        logging.info(f"[Volontaire {self.vol_id}] MAC={self.my_mac} IP={self.my_ip} "
                     f"device={self.device}")

    # ─── Estimation des ressources ──────────────────────────────────────────
    def _estimate_resources_vs_needs(self) -> Dict:
        """
        Compare ce que la MACHINE offre vs ce que le MODÈLE demande.
        Produit verdicts ram_ok / bandwidth_ok / timing_ok et logge des
        avertissements si la configuration est risquée.

        Sauvegarde le rapport dans
            results/volunteer_X/resource_estimation.json
        """
        # 1) Ressources disponibles sur la machine (snapshot SystemProfiler)
        available = SystemProfiler.get_available_resources(
            network_bandwidth_mbps=self.resources.network_bandwidth_mbps
        )

        # 2) Besoins du modèle (ModelProfiler.estimate_needs)
        needs = self.model_profiler.estimate_needs(
            dataset_name=DATASET,
            batch_size=BATCH_SIZE,
            optimizer_type="sgd",
            compression_type=COMPRESSION,
            quantization_bits=QUANTIZATION_BITS,
            sparsification_ratio=SPARSIFICATION_RATIO,
            gossip_interval=GOSSIP_INTERVAL,
            fanout=GOSSIP_FANOUT,
            network_bandwidth_mbps=self.resources.network_bandwidth_mbps,
        )

        # 3) Verdicts
        warnings_list: List[str] = []

        ram_ok = needs["ram_needed"] <= 0.9 * available["ram_free"]
        if not ram_ok:
            msg = (f"RAM CRITIQUE : besoin {needs['ram_needed']:.2f} Go "
                   f"vs disponible {available['ram_free']:.2f} Go "
                   f"(risque d'OOM, l'entraînement peut s'arrêter brutalement)")
            warnings_list.append(msg)
            logging.critical(f"[Volontaire {self.vol_id}] {msg}")

        bandwidth_ok = needs["min_bandwidth_needed_mbps"] <= available["bw"]
        if not bandwidth_ok:
            msg = (f"BANDE PASSANTE INSUFFISANTE : besoin "
                   f"{needs['min_bandwidth_needed_mbps']:.1f} Mbps "
                   f"vs disponible {available['bw']:.1f} Mbps "
                   f"(les gossip rounds risquent de prendre du retard)")
            warnings_list.append(msg)
            logging.warning(f"[Volontaire {self.vol_id}] {msg}")

        total_train_time = needs["epoch_time_estimate"] * LOCAL_EPOCHS
        timing_ok = total_train_time <= 0.8 * GOSSIP_INTERVAL
        if not timing_ok:
            msg = (f"TIMING SERRÉ : {LOCAL_EPOCHS} epochs ≈ {total_train_time:.1f}s "
                   f"vs GOSSIP_INTERVAL={GOSSIP_INTERVAL}s "
                   f"(l'entraînement risque de ne pas tenir dans le cycle gossip)")
            warnings_list.append(msg)
            logging.warning(f"[Volontaire {self.vol_id}] {msg}")

        # 4) Synthèse
        report = {
            "vol_id": self.vol_id,
            "mac": self.my_mac,
            "timestamp": time.time(),
            "available_resources": available,
            "estimated_needs": needs,
            "verdicts": {
                "ram_ok": ram_ok,
                "bandwidth_ok": bandwidth_ok,
                "timing_ok": timing_ok,
                "warnings": warnings_list,
            },
            "config_used": {
                "dataset": DATASET,
                "batch_size": BATCH_SIZE,
                "local_epochs": LOCAL_EPOCHS,
                "gossip_interval": GOSSIP_INTERVAL,
                "gossip_fanout": GOSSIP_FANOUT,
                "compression": COMPRESSION,
                "quantization_bits": QUANTIZATION_BITS,
                "sparsification_ratio": SPARSIFICATION_RATIO,
            },
        }

        # Log synthétique en INFO
        logging.info(
            f"[Volontaire {self.vol_id}] [ESTIMATION] "
            f"Param={needs['parameter_size']}MB "
            f"RAM_needed={needs['ram_needed']:.2f}Go / RAM_free={available['ram_free']:.2f}Go "
            f"BW_needed={needs['min_bandwidth_needed_mbps']:.1f}Mbps / BW_avail={available['bw']:.1f}Mbps "
            f"EpochTime≈{needs['epoch_time_estimate']:.1f}s | "
            f"verdicts: ram={ram_ok} bw={bandwidth_ok} timing={timing_ok}"
        )

        # Sauvegarde JSON
        out_dir = os.path.join(STATS_DIR, f"volunteer_{self.vol_id}")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "resource_estimation.json")
        try:
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logging.info(f"[Volontaire {self.vol_id}] Estimation sauvegardée -> {out_file}")
        except Exception as exc:
            logging.warning(f"[Volontaire {self.vol_id}] Sauvegarde estimation KO : {exc}")

        return report

    # ─── Cycle de vie ────────────────────────────────────────────────────────
    def run(self):
        hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        hb_thread.start()
        time.sleep(3)

        while self._running and (MAX_ROUNDS <= 0 or self.round_num < MAX_ROUNDS):
            try:
                self._run_gossip_round()
            except Exception as exc:
                logging.exception(f"[Volontaire {self.vol_id}] Erreur round : {exc}")
            time.sleep(GOSSIP_INTERVAL)

        self._shutdown()

    def _on_signal(self, signum, frame):
        logging.info(f"[Volontaire {self.vol_id}] Signal {signum} reçu, arrêt.")
        self._running = False

    def _shutdown(self):
        # Rapport final ModelProfiler
        try:
            final_test_acc = self._evaluate_test()
            final_report = self.model_profiler.generate_report(test_accuracy=final_test_acc)
            out_dir = os.path.join(STATS_DIR, f"volunteer_{self.vol_id}")
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, "model_profile_final.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(final_report, f, indent=2, ensure_ascii=False)
            logging.info(f"[Volontaire {self.vol_id}] Rapport final sauvegardé -> {out_file}")
        except Exception as exc:
            logging.warning(f"[Volontaire {self.vol_id}] Rapport final KO : {exc}")

        try:
            if self._heartbeat_sock:
                send_message(self._heartbeat_sock, MSG_DISCONNECT, {"mac": self.my_mac})
                self._heartbeat_sock.close()
        except Exception:
            pass
        logging.info(f"[Volontaire {self.vol_id}] Arrêt propre terminé.")

    # ─── Réseau ──────────────────────────────────────────────────────────────
    def _detect_ip(self, remote: str) -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((remote, 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def _heartbeat_loop(self):
        while self._running:
            try:
                self._heartbeat_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._heartbeat_sock.settimeout(SOCKET_TIMEOUT)
                self._heartbeat_sock.connect((self.coord_host, COORDINATOR_PORT))
                logging.info(f"[Volontaire {self.vol_id}] Connecté au coordinateur "
                             f"{self.coord_host}:{COORDINATOR_PORT}")
                while self._running:
                    payload = {
                        "mac": self.my_mac, "ip": self.my_ip,
                        "resources": self.resources.to_dict(),
                        "timestamp": time.time(),
                    }
                    send_message(self._heartbeat_sock, MSG_HEARTBEAT, payload)
                    try:
                        receive_message(self._heartbeat_sock)
                    except Exception:
                        pass
                    time.sleep(HEARTBEAT_INTERVAL)
            except Exception as exc:
                logging.warning(f"[Volontaire {self.vol_id}] Heartbeat KO : {exc}. "
                                f"Retry dans {RETRY_DELAY}s.")
                time.sleep(RETRY_DELAY)

    def _open_manager_conn(self) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(SOCKET_TIMEOUT)
        s.connect((self.manager_host, MANAGER_PORT))
        return s

    def _fetch_active_volunteers(self) -> List[str]:
        try:
            s = self._open_manager_conn()
            send_message(s, MSG_REQUEST_NEIGHBORS, {"mac": self.my_mac, "k": K_NEIGHBORS})
            msg_type, data, _ = receive_message(s)
            s.close()
            if msg_type == MSG_NEIGHBORS_RESPONSE:
                vol_list = data.get("volunteers", data.get("neighbors", []))
                # Extraire toutes les adresses MAC des autres volontaires
                all_macs = [v.get("mac_address") for v in vol_list if v.get("mac_address") and v.get("mac_address") != self.my_mac]
                # Effectuer un peer sampling aléatoire local pour obtenir k voisins
                sampled = get_peer_sample(self.my_mac, all_macs, K_NEIGHBORS)
                logging.info(f"[Volontaire {self.vol_id}] Peer sampling local : "
                             f"sélectionné {sampled} parmi {len(all_macs)} pairs totaux.")
                return sampled
        except Exception as exc:
            logging.warning(f"[Volontaire {self.vol_id}] fetch_active_volunteers KO : {exc}")
        return []

    def _send_model_to_peer(self, peer_mac: str, payload: bytes, meta: Dict) -> bool:
        try:
            s = self._open_manager_conn()
            send_message(s, MSG_SEND_MODEL,
                         {"from": self.my_mac, "to": peer_mac, "meta": meta},
                         payload=payload)
            msg_type, _, _ = receive_message(s)
            s.close()
            return msg_type == MSG_ACK
        except Exception as exc:
            logging.warning(f"[Volontaire {self.vol_id}] Envoi vers {peer_mac} KO : {exc}")
            return False

    def _poll_received_models(self) -> List[Dict[str, torch.Tensor]]:
        states: List[Dict[str, torch.Tensor]] = []
        try:
            s = self._open_manager_conn()
            send_message(s, MSG_POLL_MODELS, {"mac": self.my_mac})
            while True:
                try:
                    msg_type, data, payload = receive_message(s)
                except Exception:
                    break
                if msg_type != MSG_MODEL_DELIVERY or not payload:
                    break
                tmp_model = create_model(DATASET, NUM_CLASSES)
                try:
                    decompress_model(tmp_model, payload, data.get("meta", {}))
                    states.append(tmp_model.state_dict())
                except Exception as exc:
                    logging.warning(f"[Volontaire {self.vol_id}] Décompression KO : {exc}")
            s.close()
        except Exception as exc:
            logging.warning(f"[Volontaire {self.vol_id}] poll_received_models KO : {exc}")
        return states

    # ─── SW-UCB ──────────────────────────────────────────────────────────────
    def _select_neighbors(self, candidates: List[str]) -> List[str]:
        candidates = [c for c in candidates if c != self.my_mac]
        if not candidates:
            return []
        return self.selector.select(candidates, k=GOSSIP_FANOUT,
                                    current_round=self.round_num)

    def _record_transfer_reward(self, peer_mac: str, n_bytes: int,
                                duration_s: float, success: bool) -> None:
        self.selector.update_from_transfer(
            arm=peer_mac, bytes_sent=n_bytes, duration_s=duration_s,
            success=success, current_round=self.round_num,
        )

    # ─── Entraînement local sécurisé ─────────────────────────────────────────
    @staticmethod
    def _is_model_finite(model: nn.Module) -> bool:
        for p in model.parameters():
            if not torch.isfinite(p).all():
                return False
        return True

    @staticmethod
    def _snapshot_state(model: nn.Module) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    @staticmethod
    def _rollback(model: nn.Module, snap: Dict[str, torch.Tensor]) -> None:
        model.load_state_dict(snap)
        logging.warning("[Volontaire] Rollback : snapshot précédent restauré.")

    @staticmethod
    def _filter_received_states(received_states):
        clean = []
        for i, s in enumerate(received_states):
            ok = all(
                (not v.is_floating_point()) or torch.isfinite(v).all()
                for v in s.values()
            )
            if ok:
                clean.append(s)
            else:
                logging.warning(f"[Volontaire] Modèle reçu #{i} corrompu -> rejeté.")
        return clean

    def _train_local_safe(self, max_grad_norm: float = 1.0):
        snap = self._snapshot_state(self.model)
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(),
                              lr=LEARNING_RATE, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        t0 = time.time()
        total_loss, total_correct, total_seen, n_skipped = 0.0, 0, 0, 0

        for epoch in range(LOCAL_EPOCHS):
            epoch_start = time.time()
            for batch_idx, (x, y) in enumerate(self.train_loader):
                batch_start = time.time()
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                out = self.model(x)
                loss = criterion(out, y)

                if not torch.isfinite(loss):
                    n_skipped += 1
                    logging.warning(f"[Volontaire {self.vol_id}] Loss non finie "
                                    f"(epoch {epoch} batch {batch_idx}) -> skip.")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               max_norm=max_grad_norm)
                optimizer.step()

                if not self._is_model_finite(self.model):
                    n_skipped += 1
                    logging.warning(f"[Volontaire {self.vol_id}] NaN après step "
                                    f"(batch {batch_idx}) -> rollback batch.")
                    self._rollback(self.model, snap)
                    continue

                with torch.no_grad():
                    total_loss += loss.item() * x.size(0)
                    total_correct += (out.argmax(dim=1) == y).sum().item()
                    total_seen += x.size(0)

                # ── ModelProfiler : durée batch + loss ────────────────────
                self.model_profiler.record_batch(
                    batch_duration=time.time() - batch_start,
                    loss_val=loss.item(),
                )

            # ── ModelProfiler : durée epoch + accuracy ────────────────────
            epoch_acc = total_correct / max(1, total_seen)
            self.model_profiler.record_epoch(
                epoch_duration=time.time() - epoch_start,
                accuracy=epoch_acc,
            )

        duration = time.time() - t0
        avg_loss = total_loss / max(1, total_seen)
        avg_acc = total_correct / max(1, total_seen)

        if not self._is_model_finite(self.model):
            logging.error(f"[Volontaire {self.vol_id}] Modèle corrompu en fin de round "
                          f"-> rollback complet.")
            self._rollback(self.model, snap)
            avg_loss = float("nan")

        return avg_loss, avg_acc, duration, n_skipped

    # ─── Évaluation ─────────────────────────────────────────────────────────
    def _evaluate_test(self) -> float:
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                correct += (out.argmax(dim=1) == y).sum().item()
                total += y.size(0)
        return correct / max(1, total)

    # ─── Sauvegardes ────────────────────────────────────────────────────────
    def _save_selector_stats(self):
        out_dir = os.path.join(STATS_DIR, f"volunteer_{self.vol_id}")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"swucb_round_{self.round_num:03d}.json")
        try:
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(self.selector.get_stats(), f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logging.warning(f"[Volontaire {self.vol_id}] save_selector_stats KO : {exc}")

    def _save_profile_report(self, report: Dict):
        out_dir = os.path.join(STATS_DIR, f"volunteer_{self.vol_id}")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"profile_round_{self.round_num:03d}.json")
        try:
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logging.warning(f"[Volontaire {self.vol_id}] save_profile_report KO : {exc}")

    # ─── Boucle d'un round gossip ───────────────────────────────────────────
    def _run_gossip_round(self):
        self.round_num += 1
        round_start = time.time()
        logging.info(f"[Volontaire {self.vol_id}] === Round gossip #{self.round_num} ===")

        # PROFILAGE AVANCÉ : démarrage monitoring pour tout le round
        self.adv_profiler.start_monitoring()

        # 1. Entraînement local sécurisé
        train_loss, train_acc, train_dur, n_skipped = self._train_local_safe(max_grad_norm=1.0)
        logging.info(f"[Volontaire {self.vol_id}] Train : loss={train_loss:.4f} "
                     f"acc={train_acc:.4f} dur={train_dur:.2f}s skipped={n_skipped}")

        # 2. Sélection adaptative des voisins
        candidates = self._fetch_active_volunteers()
        peers = self._select_neighbors(candidates)

        if peers:
            compressed_bytes, meta = compress_model(
                self.model, method=COMPRESSION,
                bits=QUANTIZATION_BITS, ratio=SPARSIFICATION_RATIO,
            )
            n_bytes = len(compressed_bytes)
            orig_size = model_parameter_bytes(self.model)
            ratio = compression_ratio(orig_size, n_bytes)
            logging.info(f"[Volontaire {self.vol_id}] Compression {COMPRESSION} : "
                         f"{orig_size}->{n_bytes} octets (ratio={ratio:.2f}x)")

            for peer_mac in peers:
                t0 = time.time()
                success = self._send_model_to_peer(peer_mac, compressed_bytes, meta)
                duration = time.time() - t0
                self._record_transfer_reward(peer_mac, n_bytes, duration, success)

                # ── ModelProfiler : enregistre le coût de communication ──
                self.model_profiler.record_communication(
                    uncompressed_mb=orig_size / (1024 ** 2),
                    compressed_mb=n_bytes / (1024 ** 2),
                )
        else:
            logging.warning(f"[Volontaire {self.vol_id}] Aucun voisin -> envoi sauté.")

        # 3. Réception + filtrage + agrégation
        received_states = self._poll_received_models()
        received_states = self._filter_received_states(received_states)
        if received_states:
            average_models(self.model, received_states, local_weight=0.5)
            logging.info(f"[Volontaire {self.vol_id}] FedAvg : "
                         f"{len(received_states)} modèles agrégés.")

        # 4. Évaluation
        test_acc = self._evaluate_test()

        # PROFILAGE AVANCÉ : IPC + arrêt monitoring
        self.adv_profiler.measure_ipc(duration_s=1.5)
        adv_metrics = self.adv_profiler.stop_monitoring()
        adv_report = self.adv_profiler.get_full_report()
        adv_report["round_num"] = self.round_num
        adv_report["train_loss"] = train_loss
        adv_report["train_acc"] = train_acc
        adv_report["test_acc"] = test_acc

        # Log synthétique
        throttle = adv_metrics.get("throttle_ratio")
        throttle_str = f"{throttle*100:.1f}%" if throttle is not None else "N/A"
        ipc = adv_metrics.get("ipc")
        ipc_str = f"{ipc:.2f}" if ipc is not None else "N/A"
        round_dur = time.time() - round_start
        logging.info(
            f"[Volontaire {self.vol_id}] [PROFILE] "
            f"RSS_peak={adv_metrics.get('rss_peak_kb', 0)/1024:.1f}MB "
            f"RSS_delta={adv_metrics.get('rss_delta_kb', 0)/1024:.1f}MB "
            f"CPU_avg={adv_metrics.get('cpu_avg_pct', 0):.1f}% "
            f"Throttle={throttle_str} "
            f"ETE={adv_metrics.get('ete_seconds', 0):.2f}s "
            f"IPC={ipc_str} | "
            f"TestAcc={test_acc:.4f} round_dur={round_dur:.2f}s"
        )

        self._save_profile_report(adv_report)
        self._save_selector_stats()


# ─── CLI ─────────────────────────────────────────────────────────────────────
def parse_args():
    env_id = os.getenv("VOLUNTEER_ID")
    env_coord = os.getenv("COORDINATOR_HOST")
    env_manager = os.getenv("MANAGER_HOST")
    env_n_vol = os.getenv("N_VOLUNTEERS")
    env_my_ip = os.getenv("MY_IP")
    env_cpu = os.getenv("CPU_CORES")
    env_ram = os.getenv("RAM_GB")
    env_net = os.getenv("NETWORK_MBPS")

    p = argparse.ArgumentParser(description="Nœud Volontaire (Gossip + SW-UCB + Profilage complet)")
    p.add_argument("--id", type=int, default=int(env_id) if env_id else 0)
    p.add_argument("--n-volunteers", type=int,
                   default=int(env_n_vol) if env_n_vol else 5)
    p.add_argument("--coordinator", default=env_coord or "127.0.0.1")
    p.add_argument("--manager", default=env_manager or "127.0.0.1")
    p.add_argument("--my-ip", default=env_my_ip or "")
    p.add_argument("--cpu-cores", type=int,
                   default=int(env_cpu) if env_cpu else None)
    p.add_argument("--ram-gb", type=float,
                   default=float(env_ram) if env_ram else None)
    p.add_argument("--network-mbps", type=float,
                   default=float(env_net) if env_net else None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Volunteer(
        volunteer_id=args.id,
        n_volunteers=args.n_volunteers,
        coordinator_host=args.coordinator,
        manager_host=args.manager,
        my_ip=args.my_ip,
        cpu_cores=args.cpu_cores,
        ram_gb=args.ram_gb,
        network_bandwidth_mbps=args.network_mbps,
    ).run()
