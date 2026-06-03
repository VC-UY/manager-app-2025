"""
Configuration centrale - toutes les valeurs sont surchargeables par variables d'environnement.
"""
import os

# ─── Réseau Coordinateur ───────────────────────────────────────────────────────
COORDINATOR_HOST = os.getenv("COORDINATOR_HOST", "192.168.1.106")
COORDINATOR_PORT = int(os.getenv("COORDINATOR_PORT", "9000"))
COORDINATOR_EXTERNAL_HOST = os.getenv("COORDINATOR_EXTERNAL_HOST", "192.168.1.106")

# ─── Réseau Manager ───────────────────────────────────────────────────────────
MANAGER_HOST = os.getenv("MANAGER_HOST", "192.168.1.106")
MANAGER_PORT = int(os.getenv("MANAGER_PORT", "9001"))
MANAGER_EXTERNAL_HOST = os.getenv("MANAGER_EXTERNAL_HOST", "192.168.1.106")

# ─── Topologie ────────────────────────────────────────────────────────────────
K_NEIGHBORS = int(os.getenv("K_NEIGHBORS", "4"))

# ─── Exploration adaptative SW-UCB ───────────────────────────────────────────
SW_UCB_WINDOW     = int(os.getenv("SW_UCB_WINDOW", "8"))
SW_UCB_CONFIDENCE = float(os.getenv("SW_UCB_CONFIDENCE", "1.0"))

# ─── Gossip ───────────────────────────────────────────────────────────────────
GOSSIP_INTERVAL = int(os.getenv("GOSSIP_INTERVAL", "60"))    # secondes entre rounds
GOSSIP_FANOUT   = int(os.getenv("GOSSIP_FANOUT", "2"))       # voisins contactés par round

# ─── Entraînement local ───────────────────────────────────────────────────────
LOCAL_EPOCHS  = int(os.getenv("LOCAL_EPOCHS", "3"))
MAX_ROUNDS    = int(os.getenv("MAX_ROUNDS", "30"))  # 0 = no limit
BATCH_SIZE    = int(os.getenv("BATCH_SIZE", "32"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.01"))
DATASET       = os.getenv("DATASET", "mnist")        # 'mnist' | 'cifar10'
NUM_CLASSES   = int(os.getenv("NUM_CLASSES", "10"))
DATA_PARTITION = os.getenv("DATA_PARTITION", "iid")  # 'iid' | 'non-iid'

# ─── Compression ──────────────────────────────────────────────────────────────
COMPRESSION          = os.getenv("COMPRESSION", "quantization")  # 'none' | 'quantization' | 'sparsification'
QUANTIZATION_BITS    = int(os.getenv("QUANTIZATION_BITS", "8"))
SPARSIFICATION_RATIO = float(os.getenv("SPARSIFICATION_RATIO", "0.05"))  # top-5 %

# ─── Heartbeat ────────────────────────────────────────────────────────────────
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "10"))  # secondes
HEARTBEAT_TIMEOUT  = int(os.getenv("HEARTBEAT_TIMEOUT", "35"))   # secondes avant expulsion

# ─── Réseau / robustesse ──────────────────────────────────────────────────────
SOCKET_TIMEOUT   = int(os.getenv("SOCKET_TIMEOUT", "60"))
MAX_RETRIES      = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY      = float(os.getenv("RETRY_DELAY", "5.0"))
MAX_CONNECTIONS  = int(os.getenv("MAX_CONNECTIONS", "200"))
MAX_MODEL_BYTES  = int(os.getenv("MAX_MODEL_BYTES", str(200 * 1024 * 1024)))  # 200 MB

# ─── Statistiques ─────────────────────────────────────────────────────────────
STATS_PRINT_INTERVAL = int(os.getenv("STATS_PRINT_INTERVAL", "30"))
STATS_DIR            = os.getenv("STATS_DIR", "./results")

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
