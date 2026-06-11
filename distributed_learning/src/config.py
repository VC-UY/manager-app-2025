#!/usr/bin/env python3
"""

Ce module centralise toutes les constantes et variables d'environnement
utilisées par les nœuds (Coordinateur, Manager, Volontaire).
"""
import os
import logging

# =============================================================================
# RÉSEAU — COORDINATEUR
# =============================================================================
COORDINATOR_HOST = os.getenv("COORDINATOR_HOST", "0.0.0.0")
COORDINATOR_PORT = int(os.getenv("COORDINATOR_PORT", "9000"))

# =============================================================================
# RÉSEAU — MANAGER
# =============================================================================
MANAGER_HOST = os.getenv("MANAGER_HOST", "0.0.0.0")
MANAGER_PORT = int(os.getenv("MANAGER_PORT", "9001"))
MANAGER_EXTERNAL_HOST = os.getenv("MANAGER_EXTERNAL_HOST", "127.0.0.1")

# =============================================================================
# TOPOLOGIE — VOISINAGE
# =============================================================================
K_NEIGHBORS = int(os.getenv("K_NEIGHBORS", "3"))

# =============================================================================
# GOSSIP — MODÈLES
# =============================================================================
GOSSIP_INTERVAL = int(os.getenv("GOSSIP_INTERVAL", "60"))      # secondes
GOSSIP_FANOUT = int(os.getenv("GOSSIP_FANOUT", "1"))           # nombre de pairs

# =============================================================================
# APPRENTISSAGE — LOCAL
# =============================================================================
LOCAL_EPOCHS = int(os.getenv("LOCAL_EPOCHS", "3"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.01"))
MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", "15"))

# =============================================================================
# DONNÉES
# =============================================================================
DATASET = os.getenv("DATASET", "mnist").lower()               # mnist, cifar10
NUM_CLASSES = int(os.getenv("NUM_CLASSES", "10"))
DATA_PARTITION = os.getenv("DATA_PARTITION", "iid")       # iid, non-iid
N_VOLUNTEERS = int(os.getenv("N_VOLUNTEERS", "5"))

# =============================================================================
# COMPRESSION
# =============================================================================
COMPRESSION = os.getenv("COMPRESSION", "quantization").lower()        # none, quantization, sparsification
QUANTIZATION_BITS = int(os.getenv("QUANTIZATION_BITS", "8"))
SPARSIFICATION_RATIO = float(os.getenv("SPARSIFICATION_RATIO", "0.9"))

# =============================================================================
# HEARTBEAT — DÉTECTION DE DÉFAILLANCE
# =============================================================================
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "10"))      # secondes
HEARTBEAT_TIMEOUT = int(os.getenv("HEARTBEAT_TIMEOUT", "35"))        # secondes

# =============================================================================
# SOCKET — RÉSEAU
# =============================================================================
SOCKET_TIMEOUT = int(os.getenv("SOCKET_TIMEOUT", "60"))       # secondes
MAX_CONNECTIONS = int(os.getenv("MAX_CONNECTIONS", "100"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "5"))              # secondes

# =============================================================================
# STATISTIQUES
# =============================================================================
# Par défaut : /app/results en Docker, ./results en local
_default_stats_dir = "/app/results" if os.path.exists("/app") else "./results"
STATS_DIR = os.getenv("STATS_DIR", _default_stats_dir)
STATS_PRINT_INTERVAL = int(os.getenv("STATS_PRINT_INTERVAL", "10"))  # secondes

# =============================================================================
# SLIDING WINDOW UCB (Contextual Bandits)
# =============================================================================
SW_UCB_WINDOW = int(os.getenv("SW_UCB_WINDOW", "5"))         # taille de la fenêtre
SW_UCB_CONFIDENCE = float(os.getenv("SW_UCB_CONFIDENCE", "0.95"))

# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Valider LOG_LEVEL
valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
if LOG_LEVEL not in valid_levels:
    LOG_LEVEL = "INFO"

# =============================================================================
# HARDWARE HINTS (optionnel, remontées au Coordinateur)
# =============================================================================
CPU_CORES = os.getenv("CPU_CORES", "")
RAM_GB = os.getenv("RAM_GB", "")
NETWORK_MBPS = os.getenv("NETWORK_MBPS", "")

# =============================================================================
# IDS ET IDENTIFIANTS
# =============================================================================
VOLUNTEER_ID = int(os.getenv("VOLUNTEER_ID", "0"))
MY_IP = os.getenv("MY_IP", "")

# =============================================================================
# AFFICHAGE DEBUG
# =============================================================================
if __name__ == "__main__":
    print("Configuration du système d'apprentissage distribué")
    print("=" * 60)
    print(f"COORDINATOR_HOST={COORDINATOR_HOST}")
    print(f"COORDINATOR_PORT={COORDINATOR_PORT}")
    print(f"MANAGER_HOST={MANAGER_HOST}")
    print(f"MANAGER_PORT={MANAGER_PORT}")
    print(f"K_NEIGHBORS={K_NEIGHBORS}")
    print(f"DATASET={DATASET}")
    print(f"LOG_LEVEL={LOG_LEVEL}")
    print("=" * 60)
