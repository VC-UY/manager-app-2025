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
import math
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
    MODEL_NAME, DATASET, NUM_CLASSES, DATA_PARTITION,
    COMPRESSION, QUANTIZATION_BITS, SPARSIFICATION_RATIO,
    HEARTBEAT_INTERVAL, SOCKET_TIMEOUT,
    MAX_RETRIES, RETRY_DELAY, LOG_LEVEL, STATS_DIR,
    SW_UCB_WINDOW, SW_UCB_CONFIDENCE, ADAPTIVE_LR_METHOD,
    ADPSGD_ENABLED, ADPSGD_TOPOLOGY, ADPSGD_ROLE, ADPSGD_ALPHA,
    PEER_TIMEOUT, ADPSGD_SKIP_FACTOR_MAX, ADPSGD_STALENESS_THRESHOLD,
)
from src.adpsgd import (
    BipartiteTopology, StaleModelReader, ADPSGDStats,
    adpsgd_average, build_topology, get_neighbor_macs,
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
        # Si on a un volunteer_id > 0, on le reflète dans l'adresse MAC
        # pour éviter les collisions lors de simulations locales sur la même machine.
        base_mac = get_mac_address()
        if self.vol_id > 0:
            parts = base_mac.split(":")
            if len(parts) == 6:
                parts[-1] = f"{self.vol_id:02X}"
                self.my_mac = ":".join(parts)
            else:
                self.my_mac = f"{base_mac[:-2]}{self.vol_id:02X}"
        else:
            self.my_mac = base_mac
        self.resources = get_resource_info(
            cpu_cores=cpu_cores, ram_gb=ram_gb,
            network_bandwidth_mbps=network_bandwidth_mbps,
        )

        # Modèle, données, optimiseur
        # Seed both CPU and GPU to ensure identical initialization of weights across all volunteers
        torch.manual_seed(42)
        import numpy as np
        import random
        np.random.seed(42)
        random.seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_model(MODEL_NAME, NUM_CLASSES).to(self.device)
        self.train_loader, self.test_loader = load_dataset(
            dataset=DATASET, data_dir=os.path.join(os.path.dirname(__file__), "data"),
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

        # Initialisation du learning rate adaptatif
        self.current_lr = LEARNING_RATE
        
        # AdaStair : rounds où le LR sera divisé par 2
        self.rstair_rounds = sorted(list(set(
            r for r in [int(MAX_ROUNDS * 0.50), int(MAX_ROUNDS * 0.75), int(MAX_ROUNDS * 0.85)]
            if r > 0
        )))
        
        # AdaLoss : patience (en rounds) avant division par 2
        self.rloss_patience = [
            max(1, int(MAX_ROUNDS * 0.25)),
            max(1, int(MAX_ROUNDS * 0.15)),
            max(1, int(MAX_ROUNDS * 0.10))
        ]
        self.adaloss_alpha = 0
        self.adaloss_counter = 0
        self.adaloss_last_loss = float('inf')

        # ─── AD-PSGD ──────────────────────────────────────────────────────
        # Effective role : overridable via ADPSGD_ROLE env, sinon pair=active / impair=passive
        _effective_id = self.vol_id
        if ADPSGD_ROLE == "active":
            _effective_id = 0   # pair → active
        elif ADPSGD_ROLE == "passive":
            _effective_id = 1   # impair → passive

        self.adpsgd_topo = build_topology(
            volunteer_id=_effective_id,
            n_volunteers=self.n_volunteers,
            topology_type=ADPSGD_TOPOLOGY,
        )
        # Lecteur de snapshot stale (x̂_k = x_{k-τ})
        self.adpsgd_stale_reader = StaleModelReader(self.model)
        # Statistiques AD-PSGD par round
        self.adpsgd_stats = ADPSGDStats(self.adpsgd_topo)

        logging.info(
            f"[Volontaire {self.vol_id}] AD-PSGD "
            f"{'ACTIVÉ' if ADPSGD_ENABLED else 'DÉSACTIVÉ'} | "
            f"role={self.adpsgd_topo.role} | "
            f"topology={ADPSGD_TOPOLOGY} | "
            f"alpha={ADPSGD_ALPHA} | "
            f"neighbors={self.adpsgd_topo.get_neighbors()}"
        )

        # État
        self._running = True
        self._stats = StatsTracker(volunteer_ip=self.my_ip, results_dir=STATS_DIR)
        self.last_neighbors_list = []
        self.round_recv_details = []
        self._heartbeat_sock: Optional[socket.socket] = None

        # Suivi d'inactivité des pairs pour arrêt automatique
        self.last_active_peer_time = time.time()
        self.peer_timeout = PEER_TIMEOUT

        # Facteur de saut adaptatif (AD-PSGD)
        self.adpsgd_skip_factor = 1
        self.adpsgd_skip_counter = 0

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
        out_dir = os.path.join(STATS_DIR, f"volunteer_{self.vol_id}")
        os.makedirs(out_dir, exist_ok=True)

        # ── Rapport final AdvancedProfiler (rss_baseline + métriques session) ──
        try:
            # Si un monitoring est encore actif (arrêt en cours de round), on le stoppe
            if self.adv_profiler._monitoring:
                self.adv_profiler.stop_monitoring()

            adv_final_report = self.adv_profiler.get_full_report()
            adv_final_report["session_info"] = {
                "volunteer_id": self.vol_id,
                "total_rounds": self.round_num,
                "mac": self.my_mac,
            }
            # Inclure explicitement la baseline capturée en début de session
            adv_final_report["baseline_captured"] = self.adv_profiler.baseline

            adv_out = os.path.join(out_dir, "advanced_profile_final.json")
            with open(adv_out, "w", encoding="utf-8") as f:
                json.dump(adv_final_report, f, indent=2, ensure_ascii=False, default=str)
            logging.info(f"[Volontaire {self.vol_id}] Rapport AdvProfiler final -> {adv_out}")
        except Exception as exc:
            logging.warning(f"[Volontaire {self.vol_id}] Rapport AdvProfiler final KO : {exc}")

        # ── Rapport final ModelProfiler ──────────────────────────────────────
        try:
            final_test_acc = self._evaluate_test()
            final_report = self.model_profiler.generate_report(test_accuracy=final_test_acc)
            out_file = os.path.join(out_dir, "model_profile_final.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(final_report, f, indent=2, ensure_ascii=False)
            logging.info(f"[Volontaire {self.vol_id}] Rapport ModelProfiler final -> {out_file}")
        except Exception as exc:
            logging.warning(f"[Volontaire {self.vol_id}] Rapport ModelProfiler final KO : {exc}")

        # ── Sauvegarde et push final des statistiques ─────────────────────────
        try:
            self._stats.save()
            self._push_stats_to_manager()
        except Exception as exc:
            logging.warning(f"[Volontaire {self.vol_id}] Sauvegarde/push final des stats KO : {exc}")

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
                # TCP keepalive pour détecter les connexions mortes rapidement
                self._heartbeat_sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                self._heartbeat_sock.connect((self.coord_host, COORDINATOR_PORT))
                logging.info(f"[Volontaire {self.vol_id}] Connecté au coordinateur "
                             f"{self.coord_host}:{COORDINATOR_PORT}")
                while self._running:
                    payload = {
                        "mac_address": self.my_mac,
                        "current_ip": self.my_ip,
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
                try:
                    self._heartbeat_sock.close()
                except Exception:
                    pass
                time.sleep(RETRY_DELAY)

    def _open_manager_conn(self) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(SOCKET_TIMEOUT)
        s.connect((self.manager_host, MANAGER_PORT))
        return s

    def _fetch_active_volunteers(self) -> List[str]:
        try:
            s = self._open_manager_conn()
            send_message(s, MSG_REQUEST_NEIGHBORS, {
                "volunteer_mac": self.my_mac,
                "volunteer_ip": self.my_ip,
                "k": K_NEIGHBORS
            })
            msg_type, data, _ = receive_message(s)
            s.close()
            if msg_type == MSG_NEIGHBORS_RESPONSE:
                vol_list = data.get("volunteers", data.get("neighbors", []))
                self.last_neighbors_list = vol_list
                
                # Mise à jour dynamique du rôle AD-PSGD
                assigned_role = data.get("assigned_role")
                if assigned_role and ADPSGD_ENABLED:
                    self.adpsgd_topo.role = assigned_role
                    logging.info(f"[AD-PSGD] Rôle dynamique réassigné par le Manager : {assigned_role}")

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
            # Trouver l'IP du destinataire dans la liste des voisins si possible
            dest_ip = ""
            for cand in self.last_neighbors_list:
                if cand.get("mac_address") == peer_mac:
                    dest_ip = cand.get("current_ip", "")
                    break

            s = self._open_manager_conn()
            send_message(s, MSG_SEND_MODEL,
                         {
                             "sender_mac": self.my_mac,
                             "dest_mac": peer_mac,
                             "dest_ip": dest_ip,
                             "metadata": meta
                         },
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
            send_message(s, MSG_POLL_MODELS, {
                "volunteer_mac": self.my_mac,
                "volunteer_ip": self.my_ip
            })
            while True:
                try:
                    msg_type, data, payload = receive_message(s)
                except Exception:
                    break
                if msg_type != MSG_MODEL_DELIVERY or not payload:
                    break

                # Comptabiliser la réception
                payload_len = len(payload)
                self.round_bytes_received += payload_len
                self.round_n_models_received += 1

                meta = data.get("metadata", {})
                sender_mac = data.get("sender_ip", "unknown")  # Le manager renvoie sender_ip (qui contient le MAC du sender)
                send_ts_start = meta.get("send_ts_start")
                recv_ts = time.time()
                transfer_time = (recv_ts - send_ts_start) if send_ts_start else None

                self.round_recv_details.append({
                    "sender": sender_mac,
                    "bytes": payload_len,
                    "send_ts_start": send_ts_start,
                    "payload_bytes": payload_len,
                    "recv_ts": recv_ts,
                    "send_duration_s": meta.get("send_duration_s"),
                    "transfer_time_s": transfer_time
                })

                tmp_model = create_model(MODEL_NAME, NUM_CLASSES)
                try:
                    decompress_model(tmp_model, payload, meta)
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
        # Snapshot initial (utilisé uniquement pour le rollback de fin de round)
        round_snap = self._snapshot_state(self.model)
        self.model.train()

        # ── Warmup linéaire pour les grands modèles (ResNet, VGG) ─────────
        # Les premiers batches utilisent un LR réduit puis montent linéairement
        # vers LEARNING_RATE pour éviter la divergence sur poids aléatoires.
        WARMUP_BATCHES = 10
        base_lr = self.current_lr

        initial_lr = base_lr / 10.0 if self.round_num <= 1 else base_lr

        optimizer = optim.SGD(self.model.parameters(),
                              lr=initial_lr, momentum=0.9,
                              weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        t0 = time.monotonic()
        total_loss, total_correct, total_seen, n_skipped = 0.0, 0, 0, 0
        global_batch_idx = 0   # compteur de batches global sur toutes les epochs

        for epoch in range(LOCAL_EPOCHS):
            epoch_start = time.monotonic()
            n_batches = len(self.train_loader)
            for batch_idx, (x, y) in enumerate(self.train_loader):
                # ── Arrêt propre si SIGTERM reçu pendant l'entraînement ────
                if not self._running:
                    logging.info(f"[Volontaire {self.vol_id}] Arrêt demandé pendant "
                                 f"l'entraînement (epoch {epoch}, batch {batch_idx}/{n_batches}).")
                    break
                batch_start = time.monotonic()
                x, y = x.to(self.device), y.to(self.device)

                # ── Warmup LR linéaire sur les premiers batches ────────────
                if self.round_num <= 1 and global_batch_idx < WARMUP_BATCHES:
                    warmup_lr = base_lr * (0.1 + 0.9 * global_batch_idx / WARMUP_BATCHES)
                    for pg in optimizer.param_groups:
                        pg['lr'] = warmup_lr
                elif self.round_num <= 1 and global_batch_idx == WARMUP_BATCHES:
                    for pg in optimizer.param_groups:
                        pg['lr'] = base_lr
                global_batch_idx += 1

                optimizer.zero_grad()
                out = self.model(x)
                loss = criterion(out, y)

                if not torch.isfinite(loss):
                    n_skipped += 1
                    logging.warning(f"[Volontaire {self.vol_id}] Loss non finie "
                                    f"(epoch {epoch} batch {batch_idx}) -> skip batch.")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               max_norm=max_grad_norm)

                # Vérifier si les gradients sont finis avant de faire le step
                grads_finite = all(
                    torch.isfinite(p.grad).all()
                    for p in self.model.parameters()
                    if p.grad is not None
                )

                if not grads_finite:
                    n_skipped += 1
                    logging.warning(f"[Volontaire {self.vol_id}] Gradients non finis "
                                    f"(batch {batch_idx}) -> skip step.")
                    continue

                optimizer.step()

                with torch.no_grad():
                    total_loss += loss.item() * x.size(0)
                    total_correct += (out.argmax(dim=1) == y).sum().item()
                    total_seen += x.size(0)

                # ── ModelProfiler : durée batch + loss ────────────────────
                self.model_profiler.record_batch(
                    batch_duration=time.monotonic() - batch_start,
                    loss_val=loss.item(),
                )

                # ── Log de progression tous les 500 batches ────────────────
                if batch_idx > 0 and batch_idx % 500 == 0:
                    elapsed = time.monotonic() - epoch_start
                    eta = elapsed / batch_idx * (n_batches - batch_idx)
                    logging.info(
                        f"[Volontaire {self.vol_id}] Epoch {epoch+1}/{LOCAL_EPOCHS} "
                        f"batch {batch_idx}/{n_batches} "
                        f"loss={loss.item():.4f} acc={total_correct/max(1,total_seen):.4f} "
                        f"elapsed={elapsed:.0f}s ETA={eta:.0f}s"
                    )

            # ── Arrêt propre entre les epochs ─────────────────────────────
            if not self._running:
                break

            # ── ModelProfiler : durée epoch + accuracy ────────────────────
            epoch_acc = total_correct / max(1, total_seen)
            self.model_profiler.record_epoch(
                epoch_duration=time.monotonic() - epoch_start,
                accuracy=epoch_acc,
            )

        duration = time.monotonic() - t0

        if total_seen == 0:
            # Tous les batches ont été skippés -> rollback complet au début du round
            logging.error(f"[Volontaire {self.vol_id}] Aucun batch valide sur l'epoch entière "
                          f"-> rollback complet au snapshot de début de round.")
            self._rollback(self.model, round_snap)
            return float("nan"), 0.0, duration, n_skipped

        avg_loss = total_loss / total_seen
        avg_acc = total_correct / total_seen

        if not self._is_model_finite(self.model):
            logging.error(f"[Volontaire {self.vol_id}] Modèle corrompu en fin de round "
                          f"-> rollback complet.")
            self._rollback(self.model, round_snap)
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

    def _evaluate_loss_test(self) -> float:
        self.model.eval()
        total_loss, total = 0.0, 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = criterion(out, y)
                total_loss += loss.item() * y.size(0)
                total += y.size(0)
        return total_loss / max(1, total)

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

    def _push_stats_to_manager(self):
        """Envoie le résumé complet des stats (avec tous les rounds) au manager."""
        try:
            summary_data = self._stats.summary()
            if not summary_data:
                return
            s = self._open_manager_conn()
            send_message(s, MSG_STATS_PUSH, {
                "volunteer_ip": self.my_ip,
                "summary": summary_data,
            })
            msg_type, _, _ = receive_message(s)
            s.close()
        except Exception as exc:
            logging.warning(f"[Volontaire {self.vol_id}] Push stats to manager KO : {exc}")

    # ─── Boucle d'un round gossip ───────────────────────────────────────────
    def _run_gossip_round(self):
        self.round_num += 1
        round_start = time.time()
        round_start_mono = time.monotonic()
        logging.info(f"[Volontaire {self.vol_id}] === Round gossip #{self.round_num} ===")

        # Réinitialisation des compteurs de com pour ce round
        self.round_bytes_received = 0
        self.round_n_models_received = 0
        self.round_recv_details = []
        round_bytes_sent = 0
        round_sent_details = []
        ratio = 1.0

        # PROFILAGE AVANCÉ : démarrage monitoring pour tout le round
        self.adv_profiler.start_monitoring()

        # 1. Entraînement local sécurisé
        train_loss, train_acc, train_dur, n_skipped = self._train_local_safe(max_grad_norm=1.0)
        logging.info(f"[Volontaire {self.vol_id}] Train : loss={train_loss:.4f} "
                     f"acc={train_acc:.4f} dur={train_dur:.2f}s skipped={n_skipped}")

        # 2. Sélection adaptative des voisins
        candidates = self._fetch_active_volunteers()
        
        # Arrêt automatique s'il n'y a plus de pair actif pendant la durée PEER_TIMEOUT
        if candidates:
            self.last_active_peer_time = time.time()
        else:
            inactive_duration = time.time() - self.last_active_peer_time
            logging.warning(
                f"[Volontaire {self.vol_id}] Aucun pair actif disponible. "
                f"Durée d'inactivité : {inactive_duration:.1f}s / limite : {self.peer_timeout}s"
            )
            if inactive_duration >= self.peer_timeout:
                logging.error(
                    f"[Volontaire {self.vol_id}] ARRÊT AUTOMATIQUE : "
                    f"Aucun pair actif disponible pendant {inactive_duration:.1f}s (seuil de {self.peer_timeout}s dépassé). Cause : Solitude / Pas de pairs actifs."
                )
                self._running = False
                return

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
                t0_mono = time.monotonic()
                t0_wall = time.time()
                # On ajoute le timestamp de début d'envoi dans les métadonnées
                peer_meta = dict(meta or {})
                peer_meta["send_ts_start"] = t0_wall
                success = self._send_model_to_peer(peer_mac, compressed_bytes, peer_meta)
                duration = time.monotonic() - t0_mono
                self._record_transfer_reward(peer_mac, n_bytes, duration, success)
                
                if success:
                    round_bytes_sent += n_bytes
                    # Trouver l'IP associée à peer_mac dans candidates si disponible
                    peer_ip = "unknown"
                    for cand in self.last_neighbors_list:
                        if cand.get("mac_address") == peer_mac:
                            peer_ip = cand.get("current_ip", "unknown")
                            break
                    round_sent_details.append({
                        "dest_mac": peer_mac,
                        "dest_ip": peer_ip,
                        "bytes": n_bytes,
                        "send_duration_s": duration,
                        "send_ts_start": t0_wall,
                        "send_ts_end": t0_wall + duration
                    })

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

        if ADPSGD_ENABLED:
            # ── AD-PSGD averaging step (Algorithme 1, lignes 5-6) ──────────
            # Capture du snapshot x̂ AVANT averaging (= lecture stale)
            x_hat = self.adpsgd_stale_reader.capture(self.model, self.round_num)
            self.adpsgd_stats.reset()

            if received_states:
                # On prend le PREMIER modèle reçu comme voisin i' (averaging pairwise)
                # Conforme à l'article : x_i ← (x_i + x_i') / 2
                neighbor_state = received_states[0]

                # Staleness avant averaging
                staleness_before = self.adpsgd_stale_reader.compute_staleness(self.model)

                # Si rôle passif, appliquer le facteur de saut adaptatif
                is_passive = (self.adpsgd_topo.role == BipartiteTopology.ROLE_PASSIVE)
                should_skip = False
                if is_passive:
                    # Ajuster le skip factor en fonction de la staleness mesurée
                    threshold = ADPSGD_STALENESS_THRESHOLD
                    max_skip = ADPSGD_SKIP_FACTOR_MAX
                    
                    if staleness_before > threshold:
                        # Grande staleness -> rafraîchir le modèle -> réduire le skip factor
                        old_factor = self.adpsgd_skip_factor
                        self.adpsgd_skip_factor = max(1, self.adpsgd_skip_factor - 1)
                        if old_factor != self.adpsgd_skip_factor:
                            logging.info(f"[AD-PSGD] Staleness élevée ({staleness_before:.6f} > {threshold}) : réduction du skip factor à {self.adpsgd_skip_factor}")
                    else:
                        # Petite staleness -> modèle frais -> augmenter le skip factor
                        old_factor = self.adpsgd_skip_factor
                        self.adpsgd_skip_factor = min(max_skip, self.adpsgd_skip_factor + 1)
                        if old_factor != self.adpsgd_skip_factor:
                            logging.info(f"[AD-PSGD] Staleness faible ({staleness_before:.6f} <= {threshold}) : augmentation du skip factor à {self.adpsgd_skip_factor}")
                    
                    # Décider si on saute l'averaging
                    self.adpsgd_skip_counter += 1
                    self.adpsgd_stats.skip_factor = self.adpsgd_skip_factor

                    if self.adpsgd_skip_counter < self.adpsgd_skip_factor:
                        should_skip = True
                        logging.info(f"[AD-PSGD] Rôle passif : averaging sauté par politique de saut (compteur={self.adpsgd_skip_counter}/{self.adpsgd_skip_factor})")
                    else:
                        self.adpsgd_skip_counter = 0

                if should_skip:
                    self.adpsgd_stats.record_skip()
                    logging.info(f"[Volontaire {self.vol_id}] [AD-PSGD] Averaging sauté par saut adaptatif (staleness={staleness_before:.6f}, factor={self.adpsgd_skip_factor}).")
                else:
                    # Averaging symétrique AD-PSGD
                    adpsgd_average(self.model, neighbor_state, alpha=ADPSGD_ALPHA)

                    # Staleness après averaging (mesure l'impact de l'averaging)
                    staleness_after = self.adpsgd_stale_reader.compute_staleness(self.model)

                    # Identifier le voisin (par index de topologie ou par défaut)
                    sampled_neighbor = self.adpsgd_topo.sample_neighbor()

                    self.adpsgd_stats.record_averaging(
                        neighbor_id=sampled_neighbor if sampled_neighbor is not None else -1,
                        alpha=ADPSGD_ALPHA,
                        staleness=staleness_after,
                    )
                    logging.info(
                        f"[Volontaire {self.vol_id}] [AD-PSGD] Averaging avec voisin "
                        f"(stale_before={staleness_before:.4f} stale_after={staleness_after:.4f} "
                        f"alpha={ADPSGD_ALPHA}). {len(received_states)} modèle(s) reçu(s)."
                    )

                    # Si plusieurs modèles reçus, les incorporer avec FedAvg résiduel
                    if len(received_states) > 1:
                        extra = self._filter_received_states(received_states[1:])
                        if extra:
                            average_models(self.model, extra, local_weight=0.7)
                            logging.info(
                                f"[Volontaire {self.vol_id}] [AD-PSGD] {len(extra)} modèle(s) "
                                f"supplémentaire(s) incorporé(s) via FedAvg résiduel."
                            )
            else:
                self.adpsgd_stats.record_skip()
                logging.info(
                    f"[Volontaire {self.vol_id}] [AD-PSGD] Aucun modèle reçu → "
                    f"averaging sauté (role={self.adpsgd_topo.role})."
                )
        else:
            # Comportement Gossip classique (FedAvg)
            if received_states:
                average_models(self.model, received_states, local_weight=0.5)
                logging.info(f"[Volontaire {self.vol_id}] FedAvg : "
                             f"{len(received_states)} modèles agrégés.")

        # 4. Évaluation
        test_acc = self._evaluate_test()

        # ── Ajustement adaptatif du Learning Rate (AdaStair / AdaLoss) ─────
        if ADAPTIVE_LR_METHOD == "adastair":
            if self.round_num in self.rstair_rounds:
                self.current_lr = self.current_lr / 2.0
                logging.info(f"[AdaStair] Round {self.round_num} atteint. "
                             f"Le learning rate est divisé par 2. Nouveau LR: {self.current_lr:.6f}")
        elif ADAPTIVE_LR_METHOD == "adaloss":
            patience = self.rloss_patience[self.adaloss_alpha]
            loss_t = self._evaluate_loss_test()
            
            # Si c'est le premier round avec AdaLoss, initialiser self.adaloss_last_loss
            if self.adaloss_last_loss == float('inf'):
                self.adaloss_last_loss = loss_t
                logging.info(f"[AdaLoss] Initialisation de la loss de référence : {loss_t:.4f}")
            else:
                if loss_t >= self.adaloss_last_loss:
                    self.adaloss_counter += 1
                    logging.info(f"[AdaLoss] La loss n'a pas diminué ({loss_t:.4f} >= {self.adaloss_last_loss:.4f}). "
                                 f"Patience: {self.adaloss_counter}/{patience}")
                else:
                    logging.info(f"[AdaLoss] La loss a diminué ({loss_t:.4f} < {self.adaloss_last_loss:.4f}). "
                                 f"Réinitialisation du compteur de patience.")
                    self.adaloss_counter = 0
                    self.adaloss_last_loss = loss_t
                
                if self.adaloss_counter >= patience:
                    self.current_lr = self.current_lr / 2.0
                    self.adaloss_alpha = min(self.adaloss_alpha + 1, len(self.rloss_patience) - 1)
                    self.adaloss_counter = 0
                    self.adaloss_last_loss = loss_t
                    logging.info(f"[AdaLoss] Patience dépassée ! Le learning rate est divisé par 2. "
                                 f"Nouveau LR: {self.current_lr:.6f}. Nouvelle patience index={self.adaloss_alpha} ({self.rloss_patience[self.adaloss_alpha]} rounds)")

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
        round_dur = time.monotonic() - round_start_mono
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

        # 5. Enregistrement des statistiques structurées
        try:
            # Calcul du meilleur test acc à ce jour
            all_rounds = self._stats.rounds
            best_acc_so_far = max([r.test_acc for r in all_rounds] + [test_acc])
            best_acc_ts = time.time()

            # Extraire les voisins triés avec leur rank et score UCB
            try:
                all_macs = [item.get("mac_address") for item in self.last_neighbors_list if item.get("mac_address")]
                ucb_scores = self.selector._compute_ucb(all_macs, self.round_num)
            except Exception:
                ucb_scores = {}

            neighbors_info = []
            for v in self.last_neighbors_list:
                mac = v.get("mac_address")
                if not mac:
                    continue
                score = ucb_scores.get(mac, float("inf"))
                neighbors_info.append({
                    "mac_address": mac,
                    "current_ip": v.get("current_ip"),
                    "resources": v.get("resources"),
                    "last_heartbeat": v.get("last_heartbeat"),
                    "bandwidth_history": v.get("bandwidth_history"),
                    "sw_ucb_score": score if math.isfinite(score) else 100.0,
                    "sw_ucb_rank": 0
                })
            
            neighbors_info.sort(key=lambda x: x["sw_ucb_score"], reverse=True)
            for rank, item in enumerate(neighbors_info, start=1):
                item["sw_ucb_rank"] = rank

            # Estimation de l'énergie et niveau de batterie
            try:
                import psutil
                bat = psutil.sensors_battery()
                battery = bat.percent if bat is not None else 100.0
            except Exception:
                battery = 100.0

            cpu_avg = adv_metrics.get("cpu_avg_pct", 0.0)
            cpu_load_fraction = cpu_avg / 100.0
            # Estimation de l'énergie avec 65W TDP et 10W idle
            watts_avg = 10.0 + cpu_load_fraction * (65.0 - 10.0)
            energy_used = watts_avg * round_dur

            orig_size = model_parameter_bytes(self.model)

            # Métriques AD-PSGD à inclure dans les stats
            adpsgd_metrics = self.adpsgd_stats.to_dict() if ADPSGD_ENABLED else {}

            self._stats.record(
                round_num=self.round_num,
                train_loss=train_loss,
                train_acc=train_acc,
                test_acc=test_acc,
                train_duration_s=train_dur,
                bytes_sent=round_bytes_sent,
                bytes_received=self.round_bytes_received,
                n_models_received=self.round_n_models_received,
                compression_ratio=ratio,
                learning_rate=self.current_lr,
                
                neighbors_info=neighbors_info,
                sent_details=round_sent_details,
                recv_details=self.round_recv_details,
                
                round_start_ts=round_start,
                round_end_ts=time.time(),
                round_duration_s=round_dur,
                best_test_acc_so_far=best_acc_so_far,
                best_test_acc_ts=best_acc_ts,
                
                cpu_percent_peak=adv_metrics.get("cpu_avg_pct", 0.0),
                cpu_percent_mean=adv_metrics.get("cpu_avg_pct", 0.0),
                ram_usage_gb_peak=adv_metrics.get("rss_peak_kb", 0.0) / (1024 * 1024),
                ram_usage_gb_mean=adv_metrics.get("rss_avg_kb", 0.0) / (1024 * 1024),
                battery_level=battery,
                energy_used_joules=energy_used,
                gradient_size_mb=orig_size / (1024 * 1024),
                batch_time_avg_s=adv_metrics.get("batch_time_avg_s", 0.0),
                
                rss_baseline_kb=adv_metrics.get("rss_baseline_kb", 0),
                rss_peak_kb=adv_metrics.get("rss_peak_kb", 0),
                rss_avg_kb=adv_metrics.get("rss_avg_kb", 0.0),
                rss_delta_kb=adv_metrics.get("rss_delta_kb", 0),
                pss_peak_kb=adv_metrics.get("pss_peak_kb", 0),
                pss_avg_kb=adv_metrics.get("pss_avg_kb", 0.0),
                uss_peak_kb=adv_metrics.get("uss_peak_kb", 0),
                uss_avg_kb=adv_metrics.get("uss_avg_kb", 0.0),
                rss_profile=adv_metrics.get("rss_profile", []),
                cpu_avg_pct=adv_metrics.get("cpu_avg_pct", 0.0),
                cpu_max_mhz=adv_metrics.get("cpu_max_mhz", 0.0),
                cpu_avg_freq_mhz=adv_metrics.get("cpu_avg_freq_mhz", 0.0),
                throttle_ratio=adv_metrics.get("throttle_ratio", 0.0),
                ete_seconds=adv_metrics.get("ete_seconds", 0.0),
                n_samples=adv_metrics.get("n_samples", 0),
                ipc=adv_metrics.get("ipc", None),
                adpsgd=adpsgd_metrics,
            )

            # Sauvegarde locale et envoi au manager
            self._stats.save()
            self._push_stats_to_manager()

        except Exception as exc:
            logging.error(f"[Volontaire {self.vol_id}] Erreur lors de l'enregistrement/envoi des stats : {exc}", exc_info=True)


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
