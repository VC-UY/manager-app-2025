"""
Profiler Système et Modèle pour le suivi des ressources.
"""
import os
import time
import psutil
import torch
import threading
import logging
from typing import Dict, Any, List, Optional

class SystemProfiler:
    """
    Profileur système pour vérifier les ressources disponibles sur la machine,
    suivre la consommation matérielle en temps réel et générer des rapports.
    """
    def __init__(self, tdp_watts: float = 65.0, power_idle_watts: float = 10.0):
        self.tdp = tdp_watts
        self.power_idle = power_idle_watts
        self._monitoring = False
        self._monitor_thread = None
        self.cpu_history = []
        self.ram_history = []
        self.battery_history = []
        self.start_time = 0.0
        self.results = {}

    @staticmethod
    def get_available_resources(network_bandwidth_mbps: float = 1000.0) -> Dict[str, Any]:
        """Mesures d'avant-entraînement."""
        # RAM disponible en Go
        ram_free = psutil.virtual_memory().available / (1024**3)
        # CPU charge actuelle
        cpu_load = psutil.cpu_percent(interval=0.1)
        # Cœurs CPU
        cpu_cores = psutil.cpu_count(logical=True) or 1
        # Fréquence CPU max
        try:
            freq = psutil.cpu_freq()
            cpu_freq = freq.max / 1000.0 if freq else 2.0
        except:
            cpu_freq = 2.0
        # Batterie
        try:
            bat = psutil.sensors_battery()
            battery = bat.percent if bat is not None else 100.0
        except:
            battery = 100.0
        # Espace disque libre
        try:
            disk_free = psutil.disk_usage('.').free / (1024**3)
        except:
            disk_free = 0.0

        return {
            "ram_free": round(ram_free, 2),
            "cpu_load": round(cpu_load, 1),
            "cpu_cores": cpu_cores,
            "cpu_freq": round(cpu_freq, 2),
            "bw": network_bandwidth_mbps,
            "battery": round(battery, 1),
            "disk_free": round(disk_free, 2)
        }

    def start_monitoring(self, interval: float = 0.5):
        """Démarre le suivi en temps réel dans un thread."""
        self.cpu_history.clear()
        self.ram_history.clear()
        self.battery_history.clear()
        self.start_time = time.time()
        self._monitoring = True
        
        def monitor_loop():
            # Initial sampling
            psutil.cpu_percent(interval=None)
            while self._monitoring:
                try:
                    cpu = psutil.cpu_percent(interval=None)
                    ram = psutil.virtual_memory().used / (1024**3)
                    bat = psutil.sensors_battery()
                    battery = bat.percent if bat is not None else 100.0
                    
                    self.cpu_history.append(cpu)
                    self.ram_history.append(ram)
                    self.battery_history.append(battery)
                except Exception as e:
                    logging.debug(f"Erreur monitoring: {e}")
                time.sleep(interval)

        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self) -> Dict[str, Any]:
        """Arrête le suivi et génère le rapport."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        
        duration = time.time() - self.start_time
        if not self.cpu_history:
            self.cpu_history = [psutil.cpu_percent(interval=0.1)]
        if not self.ram_history:
            self.ram_history = [psutil.virtual_memory().used / (1024**3)]

        cpu_avg = sum(self.cpu_history) / len(self.cpu_history)
        cpu_peak = max(self.cpu_history)
        ram_peak = max(self.ram_history)
        
        # Estimation de l'énergie utilisée (Joules = Watts * secondes)
        cpu_load_fraction = cpu_avg / 100.0
        watts_avg = self.power_idle + cpu_load_fraction * (self.tdp - self.power_idle)
        energy_used = watts_avg * duration # Joules

        return {
            "cpu_avg": round(cpu_avg, 1),
            "cpu_peak": round(cpu_peak, 1),
            "ram_peak": round(ram_peak, 2),
            "energy_used": round(energy_used, 1) # en Joules
        }

    def __enter__(self):
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.results = self.stop_monitoring()


class ModelProfiler:
    """
    Profileur de modèle pour estimer les besoins en ressources du modèle,
    mesurer son comportement pendant l'entraînement et fournir des rapports.
    """
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.param_bytes = sum(p.numel() for p in model.parameters()) * 4 # float32
        self.param_count = sum(p.numel() for p in model.parameters())
        self.device = next(model.parameters()).device
        
        # Variables de suivi pendant l'entraînement
        self.batch_times = []
        self.epoch_times = []
        self.losses = []
        self.accuracies = []
        self.gradient_traffic = 0.0 # Mo cumulés non compressés
        self.compressed_traffic = 0.0 # Mo cumulés compressés
        self.start_time = 0.0

    def estimate_needs(self, 
                       dataset_name: str, 
                       batch_size: int, 
                       optimizer_type: str = "sgd",
                       compression_type: str = "none",
                       quantization_bits: int = 8,
                       sparsification_ratio: float = 0.1,
                       gossip_interval: float = 60.0,
                       fanout: int = 1,
                       network_bandwidth_mbps: float = 1000.0) -> Dict[str, Any]:
        """Estimer les besoins matériels du modèle (avant entraînement)."""
        # Taille des paramètres en Mo
        param_size_mb = self.param_bytes / (1024**2)
        # Taille des gradients en Mo
        gradient_size_mb = param_size_mb # mêmes dimensions

        # Mémoire nécessaire : Modèle + Gradients + Optimiseur + Activations
        opt_factor = 2 if optimizer_type.lower() == "sgd" else 3
        optimizer_mem_mb = opt_factor * param_size_mb

        # Activations memory: estimation dynamique
        input_shape = (batch_size, 1, 28, 28) if dataset_name.lower() == "mnist" else (batch_size, 3, 32, 32)
        dummy_input = torch.zeros(input_shape, device=self.device)
        
        torch.cuda.empty_cache()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated()
            try:
                out = self.model(dummy_input)
                loss = out.sum()
                loss.backward()
                peak_mem = torch.cuda.max_memory_allocated()
                act_mem_mb = (peak_mem - mem_before) / (1024**2)
            except Exception:
                act_mem_mb = (batch_size * 0.1)
        else:
            act_mem_mb = (batch_size * 0.2)

        ram_needed_mb = param_size_mb + gradient_size_mb + optimizer_mem_mb + act_mem_mb
        ram_needed_gb = ram_needed_mb / 1024.0

        # Coût de communication par round (en Mo)
        if compression_type.lower() == "none":
            comp_size_mb = param_size_mb
        elif compression_type.lower() == "quantization":
            comp_size_mb = param_size_mb * (quantization_bits / 32.0)
        elif compression_type.lower() == "sparsification":
            comp_size_mb = param_size_mb * sparsification_ratio
        else:
            comp_size_mb = param_size_mb

        # Coût total de communication par round pour envoyer le modèle à fanout voisins
        communication_cost_mb = comp_size_mb * fanout
        
        # Bande passante minimale requise en Mbps de sorte à envoyer en 10% du gossip_interval
        target_send_time = gossip_interval * 0.1
        min_bw_mbps = (comp_size_mb * 8) / max(0.1, target_send_time)

        # Estimation du temps d'une époque (sur base d'une simulation rapide)
        images_per_volunteer = 60000 // 5 if dataset_name.lower() == "mnist" else 50000 // 5
        batches_per_epoch = images_per_volunteer // batch_size
        
        t0 = time.time()
        opt = torch.optim.SGD(self.model.parameters(), lr=0.01)
        for _ in range(5):
            opt.zero_grad()
            out = self.model(dummy_input)
            loss = out.sum()
            loss.backward()
            opt.step()
        t1 = time.time()
        step_time_avg = (t1 - t0) / 5.0
        epoch_time_estimate = step_time_avg * batches_per_epoch

        return {
            "ram_needed": round(ram_needed_gb, 2),
            "gradient_size": round(gradient_size_mb, 2),
            "parameter_size": round(param_size_mb, 2),
            "epoch_time_estimate": round(epoch_time_estimate, 1),
            "communication_cost_mb": round(communication_cost_mb, 2),
            "min_bandwidth_needed_mbps": round(min_bw_mbps, 2)
        }

    def start_training_tracking(self):
        self.start_time = time.time()
        self.batch_times.clear()
        self.epoch_times.clear()
        self.losses.clear()
        self.accuracies.clear()

    def record_batch(self, batch_duration: float, loss_val: float):
        self.batch_times.append(batch_duration)
        self.losses.append(loss_val)

    def record_epoch(self, epoch_duration: float, accuracy: float):
        self.epoch_times.append(epoch_duration)
        self.accuracies.append(accuracy)

    def record_communication(self, uncompressed_mb: float, compressed_mb: float):
        self.gradient_traffic += uncompressed_mb
        self.compressed_traffic += compressed_mb

    def generate_report(self, test_accuracy: float) -> Dict[str, Any]:
        """Génère le rapport du modèle post-entraînement."""
        training_time = time.time() - self.start_time
        return {
            "training_time": round(training_time, 1),
            "gradient_traffic": round(self.gradient_traffic, 2),
            "compressed_traffic": round(self.compressed_traffic, 2),
            "final_accuracy": round(test_accuracy * 100, 2)
        }
