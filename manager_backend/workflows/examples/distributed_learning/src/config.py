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
COORDINATOR_HOST = os.getenv("COORDINATOR_HOST", "192.168.68.143")
COORDINATOR_PORT = int(os.getenv("COORDINATOR_PORT", "9000"))

# =============================================================================
# RÉSEAU — MANAGER
# =============================================================================
MANAGER_HOST = os.getenv("MANAGER_HOST", "192.168.68.143")
MANAGER_PORT = int(os.getenv("MANAGER_PORT", "9001"))
MANAGER_EXTERNAL_HOST = os.getenv("MANAGER_EXTERNAL_HOST", "192.168.68.143")

# =============================================================================
# TOPOLOGIE — VOISINAGE
# =============================================================================
K_NEIGHBORS = int(os.getenv("K_NEIGHBORS", "3"))

# =============================================================================
# GOSSIP — MODÈLES
# =============================================================================
GOSSIP_INTERVAL = int(os.getenv("GOSSIP_INTERVAL", "60"))      # secondes
if GOSSIP_INTERVAL <= 0:
    logging.warning("GOSSIP_INTERVAL invalide (%s), valeur par défaut 60s utilisée.", os.getenv("GOSSIP_INTERVAL", "60"))
    GOSSIP_INTERVAL = 60
elif GOSSIP_INTERVAL > 600:
    logging.warning("GOSSIP_INTERVAL trop grand (%ss), il est ramené à 60s pour éviter un blocage de l'expérience.", GOSSIP_INTERVAL)
    GOSSIP_INTERVAL = 60
GOSSIP_FANOUT = int(os.getenv("GOSSIP_FANOUT", "1"))           # nombre de pairs

# =============================================================================
# APPRENTISSAGE — LOCAL
# =============================================================================
LOCAL_EPOCHS = int(os.getenv("LOCAL_EPOCHS", "1"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", "10"))

# Méthode d'ajustement adaptatif du LR (none | adastair | adaloss)
ADAPTIVE_LR_METHOD = os.getenv("ADAPTIVE_LR_METHOD", "adaloss").lower().strip()

# =============================================================================
# AD-PSGD — Asynchronous Decentralized Parallel Stochastic Gradient Descent
# Lian et al. (2018) https://arxiv.org/abs/1710.06952
# =============================================================================

# Active ou non (si False : comportement Gossip classique inchangé)
ADPSGD_ENABLED = os.getenv("ADPSGD_ENABLED", "true").lower().strip() in ("1", "true", "yes")

# Topologie du ring de communication
#   ring          : voisins immédiats seulement (simple, O(n) dissémination)
#   exponential   : sauts exponentiels 2^k+1 (recommandé, O(log n) dissémination)
ADPSGD_TOPOLOGY = os.getenv("ADPSGD_TOPOLOGY", "exponential").lower().strip()

# Rôle dans la partition bipartie (pour éviter les deadlocks)
#   auto   : déduit de VOLUNTEER_ID (pair=active, impair=passive)
#   active : ce nœud initie toujours l'averaging
#   passive: ce nœud répond uniquement quand un actif le contacte
ADPSGD_ROLE = os.getenv("ADPSGD_ROLE", "auto").lower().strip()

# Poids de l'averaging symétrique : x_i ← alpha * x_i + (1 - alpha) * x_j
# L'article utilise alpha = 0.5 (moyenne exacte, doubly stochastic)
ADPSGD_ALPHA = float(os.getenv("ADPSGD_ALPHA", "0.5"))
if not (0.0 < ADPSGD_ALPHA < 1.0):
    logging.warning(f"ADPSGD_ALPHA invalide ({ADPSGD_ALPHA}), remis à 0.5.")
    ADPSGD_ALPHA = 0.5

# Facteur de saut adaptatif pour le rôle passif
ADPSGD_SKIP_FACTOR_MAX = int(os.getenv("ADPSGD_SKIP_FACTOR_MAX", "5"))
ADPSGD_STALENESS_THRESHOLD = float(os.getenv("ADPSGD_STALENESS_THRESHOLD", "0.05"))

# =============================================================================
# MODÈLE
# =============================================================================
# Architectures disponibles : resnet18 | resnet50 | resnet101 | resnet152 | vgg19
MODEL_NAME = os.getenv("MODEL_NAME", "resnet18").lower()

# =============================================================================
# DONNÉES
# =============================================================================
# Datasets disponibles : cifar10 | cifar100 | imagenet
DATASET = os.getenv("DATASET", "cifar10").lower()

# Nombre de classes — déduit automatiquement du dataset si non surchargé
_NUM_CLASSES_MAP = {"cifar10": 10, "cifar100": 100, "imagenet": 1000}
_default_num_classes = _NUM_CLASSES_MAP.get(DATASET, 10)
NUM_CLASSES = int(os.getenv("NUM_CLASSES", str(_default_num_classes)))

DATA_PARTITION = os.getenv("DATA_PARTITION", "iid")       # iid, non-iid
N_VOLUNTEERS = int(os.getenv("N_VOLUNTEERS", "2"))

# =============================================================================
# COMPRESSION
# =============================================================================
COMPRESSION = os.getenv("COMPRESSION", "jointsq").lower()        # none, quantization, sparsification
QUANTIZATION_BITS = int(os.getenv("QUANTIZATION_BITS", "8"))
SPARSIFICATION_RATIO = float(os.getenv("SPARSIFICATION_RATIO", "0.9"))

# =============================================================================
# HEARTBEAT — DÉTECTION DE DÉFAILLANCE
# =============================================================================
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "10"))      # secondes
HEARTBEAT_TIMEOUT = int(os.getenv("HEARTBEAT_TIMEOUT", "35"))        # secondes

# Seuil d'inactivité (sans pair) pour l'arrêt automatique
PEER_TIMEOUT = int(os.getenv("PEER_TIMEOUT", "60"))                  # secondes

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
    print(f"MODEL_NAME={MODEL_NAME}")
    print(f"DATASET={DATASET}")
    print(f"NUM_CLASSES={NUM_CLASSES}")
    print(f"LOG_LEVEL={LOG_LEVEL}")
    print("=" * 60)