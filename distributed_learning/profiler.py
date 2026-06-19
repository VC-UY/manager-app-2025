#!/usr/bin/env python3
"""
Script CLI autonome pour exécuter le profileur de ressources système et de modèle.
Permet de valider le comportement hors-ligne du profileur.
"""
import argparse
import sys
import os
import time
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Importer les profileurs et le créateur de modèle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import create_model
from src.profiler import SystemProfiler, ModelProfiler

def run_cli():
    parser = argparse.ArgumentParser(description="Profileur de ressources Système & Modèle")
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"],
                        help="Nom du dataset (défaut: mnist)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Taille du batch (défaut: 32)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Nombre d'époques d'entraînement (défaut: 1)")
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"],
                        help="Type d'optimiseur (défaut: sgd)")
    parser.add_argument("--compression", default="quantization", choices=["none", "quantization", "sparsification"],
                        help="Méthode de compression (défaut: quantization)")
    parser.add_argument("--sparsity", type=float, default=0.1,
                        help="Ratio de sparsification (défaut: 0.1)")
    parser.add_argument("--bits", type=int, default=8,
                        help="Nombre de bits pour la quantification (défaut: 8)")
    parser.add_argument("--gossip-interval", type=int, default=60,
                        help="Intervalle de gossip en secondes (défaut: 60)")
    parser.add_argument("--tdp", type=float, default=65.0,
                        help="TDP du processeur en Watts (défaut: 65.0)")
    parser.add_argument("--idle-watts", type=float, default=10.0,
                        help="Consommation au repos en Watts (défaut: 10.0)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initialisation sur le device : {device}")

    # Création du modèle
    model = create_model(args.dataset, 10).to(device)

    # 1. AVANT L'ENTRAÎNEMENT (Estimation)
    print("\n" + "="*50)
    print(" 1. ESTIMATIONS AVANT L'ENTRAÎNEMENT (Profilers)")
    print("="*50)

    # Profiler système (ressources disponibles)
    sys_avail = SystemProfiler.get_available_resources()
    print("--- System Profiler (Ressources disponibles) ---")
    for k, v in sys_avail.items():
        print(f"  {k:<12}: {v}")

    # Profiler modèle (besoins estimés)
    model_prof = ModelProfiler(model)
    model_est = model_prof.estimate_needs(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        optimizer_type=args.optimizer,
        compression_type=args.compression,
        quantization_bits=args.bits,
        sparsification_ratio=args.sparsity,
        gossip_interval=args.gossip_interval,
        fanout=1,
        network_bandwidth_mbps=sys_avail["bw"]
    )
    print("\n--- Model Profiler (Besoins requis) ---")
    for k, v in model_est.items():
        print(f"  {k:<30}: {v}")

    # Décision du manager (Locale)
    print("\n--- Décision ---")
    ram_ok = sys_avail["ram_free"] >= model_est["ram_needed"]
    print(f"  RAM Disponible : {sys_avail['ram_free']} Go")
    print(f"  RAM Requise    : {model_est['ram_needed']} Go")
    if ram_ok:
        print("  => [ACCEPTÉ] La machine a assez de ressources pour cette tâche.")
    else:
        print("  => [REFUSÉ] Ressources RAM insuffisantes. Arrêt.")
        sys.exit(1)

    # 2. PENDANT L'ENTRAÎNEMENT (Monitoring temps réel)
    print("\n" + "="*50)
    print(" 2. ENTRAÎNEMENT & MONITORING TEMPS RÉEL")
    print("="*50)
    
    # Créer un mini dataset factice pour simuler l'entraînement rapidement
    # mnist: 1x28x28, cifar10: 3x32x32
    channels = 1 if args.dataset == "mnist" else 3
    height = 28 if args.dataset == "mnist" else 32
    width = height
    
    # Simuler 256 images pour l'entraînement local factice
    num_samples = 256
    x_dummy = torch.randn(num_samples, channels, height, width)
    y_dummy = torch.randint(0, 10, (num_samples,))
    dummy_dataset = TensorDataset(x_dummy, y_dummy)
    train_loader = DataLoader(dummy_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    sys_prof = SystemProfiler(tdp_watts=args.tdp, power_idle_watts=args.idle_watts)
    model_prof.start_training_tracking()

    print(f"[*] Démarrage de l'entraînement simulé ({args.epochs} époques)...")
    
    # Démarrer le monitor de ressources système
    with sys_prof:
        for epoch in range(args.epochs):
            t_epoch_start = time.time()
            model.train()
            correct, total = 0, 0
            
            for batch_idx, (X, y) in enumerate(train_loader):
                t_batch_start = time.time()
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(X)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                
                batch_duration = time.time() - t_batch_start
                model_prof.record_batch(batch_duration, loss.item())
                
                correct += out.argmax(1).eq(y).sum().item()
                total += len(y)
                
            epoch_duration = time.time() - t_epoch_start
            accuracy = correct / total
            model_prof.record_epoch(epoch_duration, accuracy)

            # Simuler de la bande passante consommée par époque (ex: envoi du modèle)
            # Taille des paramètres en Mo
            param_mb = model_prof.param_bytes / (1024**2)
            comp_factor = (args.bits/32.0) if args.compression == "quantization" else (args.sparsity if args.compression == "sparsification" else 1.0)
            model_prof.record_communication(param_mb, param_mb * comp_factor)
            
            # Afficher des métriques temps réel
            # Lire les métriques intermédiaires du système
            cpu_now = psutil.cpu_percent(interval=None)
            ram_now = psutil.virtual_memory().used / (1024**3)
            try:
                bat = psutil.sensors_battery()
                bat_now = bat.percent if bat is not None else 100.0
            except:
                bat_now = 100.0
            
            print(f"  Époque {epoch+1:2d} | CPU: {cpu_now:.1f}% | RAM: {ram_now:.2f} Go | Batterie: {bat_now:.1f}% | Acc: {accuracy:.2%} | Loss: {loss.item():.4f}")
            time.sleep(0.5)

    # 3. APRÈS L'ENTRAÎNEMENT (Rapports finaux)
    print("\n" + "="*50)
    print(" 3. RAPPORTS POST-ENTRAÎNEMENT")
    print("="*50)

    # Rapport Système
    sys_report = sys_prof.results
    print("--- System Profiler Report (Machine) ---")
    for k, v in sys_report.items():
        unit = " %" if "cpu" in k else (" Go" if "ram" in k else " Joules")
        print(f"  {k:<12}: {v}{unit}")

    # Rapport Modèle (Précision finale sur test factice)
    model_report = model_prof.generate_report(accuracy)
    print("\n--- Model Profiler Report (Modèle) ---")
    for k, v in model_report.items():
        unit = " s" if "time" in k else (" Mo" if "traffic" in k else " %")
        print(f"  {k:<20}: {v}{unit}")

if __name__ == "__main__":
    run_cli()
