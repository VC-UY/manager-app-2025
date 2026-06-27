#!/usr/bin/env bash
# =============================================================================
# deploy.sh — Déploiement du système d'apprentissage distribué sur machines SSH
#
# Usage :
#   chmod +x deploy.sh
#   ./deploy.sh                          # avec les IPs par défaut
#   COORD_IP=10.0.1.5 MANAGER_IP=10.0.1.6 ./deploy.sh
#
# Pré-requis : SSH sans mot de passe (clés) configuré vers toutes les machines
# =============================================================================

set -e  # Arrêt immédiat si une commande échoue

# ─── Configuration IPs ────────────────────────────────────────────────────────
COORD_IP="${COORD_IP:-192.168.1.10}"
MANAGER_IP="${MANAGER_IP:-192.168.1.11}"
VOL_IPS="${VOL_IPS:-192.168.1.20 192.168.1.21 192.168.1.22}"  # séparés par espace

SSH_USER="${SSH_USER:-ubuntu}"
PROJECT_DIR="${PROJECT_DIR:-~/distributed_learning}"
PYTHON="${PYTHON:-$PROJECT_DIR/venv/bin/python}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"

# ─── Paramètres d'expérience ─────────────────────────────────────────────────
DATASET="${DATASET:-cifar10}"
DATA_PARTITION="${DATA_PARTITION:-iid}"
COMPRESSION="${COMPRESSION:-quantization}"
K_NEIGHBORS="${K_NEIGHBORS:-3}"
GOSSIP_INTERVAL="${GOSSIP_INTERVAL:-60}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-3}"

# ─── Fonctions utilitaires ────────────────────────────────────────────────────
ssh_run() {
    local host="$1"
    shift
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$SSH_USER@$host" "$@"
}

ssh_bg() {
    local host="$1"
    local name="$2"
    shift 2
    ssh_run "$host" "mkdir -p $LOG_DIR && nohup $@ > $LOG_DIR/${name}.log 2>&1 &"
    echo "  [OK] $name démarré sur $host"
}

# ─── Vérifications préliminaires ─────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  DÉPLOIEMENT — APPRENTISSAGE DISTRIBUÉ FRUGAL"
echo "════════════════════════════════════════════════════════════"
echo "  Coordinateur : $COORD_IP"
echo "  Manager      : $MANAGER_IP"
echo "  Volontaires  : $VOL_IPS"
echo "  Dataset      : $DATASET ($DATA_PARTITION)"
echo "  Compression  : $COMPRESSION"
echo "  Voisins k    : $K_NEIGHBORS"
echo "════════════════════════════════════════════════════════════"

# Vérifier la connectivité SSH
echo ""
echo "1. Vérification connectivité SSH…"
for ip in $COORD_IP $MANAGER_IP $VOL_IPS; do
    if ssh_run "$ip" "echo ok" > /dev/null 2>&1; then
        echo "  [OK] $ip"
    else
        echo "  [ERREUR] Impossible de se connecter à $ip"
        exit 1
    fi
done

# ─── Arrêter tout processus existant ─────────────────────────────────────────
echo ""
echo "2. Arrêt des processus existants…"
for ip in $COORD_IP $MANAGER_IP $VOL_IPS; do
    ssh_run "$ip" "pkill -f 'python.*coordinator.py' 2>/dev/null || true"
    ssh_run "$ip" "pkill -f 'python.*manager.py' 2>/dev/null || true"
    ssh_run "$ip" "pkill -f 'python.*volunteer.py' 2>/dev/null || true"
    echo "  [OK] $ip nettoyé"
done
sleep 2

# ─── Créer les dossiers logs/results ─────────────────────────────────────────
echo ""
echo "3. Création des répertoires…"
for ip in $COORD_IP $MANAGER_IP $VOL_IPS; do
    ssh_run "$ip" "mkdir -p $LOG_DIR $PROJECT_DIR/results"
    echo "  [OK] $ip"
done

# ─── Lancement Manager (d'abord — le coordinateur lui envoie des messages) ───
echo ""
echo "4. Démarrage du Manager ($MANAGER_IP:9001)…"
ssh_bg "$MANAGER_IP" "manager" \
    "MANAGER_HOST=0.0.0.0 MANAGER_PORT=9001 \
     MANAGER_EXTERNAL_HOST=$MANAGER_IP \
     K_NEIGHBORS=$K_NEIGHBORS \
     STATS_PRINT_INTERVAL=30 STATS_DIR=$PROJECT_DIR/results \
     LOG_LEVEL=INFO \
     $PYTHON $PROJECT_DIR/manager.py"
sleep 3

# ─── Lancement Coordinateur ───────────────────────────────────────────────────
echo ""
echo "5. Démarrage du Coordinateur ($COORD_IP:9000)…"
ssh_bg "$COORD_IP" "coordinator" \
    "COORDINATOR_HOST=0.0.0.0 COORDINATOR_PORT=9000 \
     COORDINATOR_EXTERNAL_HOST=$COORD_IP \
     MANAGER_EXTERNAL_HOST=$MANAGER_IP MANAGER_PORT=9001 \
     HEARTBEAT_TIMEOUT=35 HEARTBEAT_INTERVAL=10 \
     LOG_LEVEL=INFO \
     $PYTHON $PROJECT_DIR/coordinator.py"
sleep 4

# ─── Lancement des Volontaires ────────────────────────────────────────────────
echo ""
echo "6. Démarrage des Volontaires…"
VOL_ARRAY=($VOL_IPS)
N_VOL="${#VOL_ARRAY[@]}"

for i in "${!VOL_ARRAY[@]}"; do
    ip="${VOL_ARRAY[$i]}"
    ssh_bg "$ip" "volunteer_$i" \
        "DATASET=$DATASET DATA_PARTITION=$DATA_PARTITION \
         COMPRESSION=$COMPRESSION K_NEIGHBORS=$K_NEIGHBORS \
         GOSSIP_INTERVAL=$GOSSIP_INTERVAL GOSSIP_FANOUT=1 \
         LOCAL_EPOCHS=$LOCAL_EPOCHS LEARNING_RATE=0.01 \
         HEARTBEAT_INTERVAL=10 SOCKET_TIMEOUT=30 \
         STATS_DIR=$PROJECT_DIR/results LOG_LEVEL=INFO \
         $PYTHON $PROJECT_DIR/volunteer.py \
         --id $i --n-volunteers $N_VOL \
         --coordinator $COORD_IP --manager $MANAGER_IP \
         --my-ip $ip"
    sleep 1
done

# ─── Résumé ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Système déployé avec $N_VOL volontaires."
echo ""
echo "  Suivi des logs :"
echo "    ssh $SSH_USER@$MANAGER_IP 'tail -f $LOG_DIR/manager.log'"
echo "    ssh $SSH_USER@$COORD_IP   'tail -f $LOG_DIR/coordinator.log'"
for i in "${!VOL_ARRAY[@]}"; do
    ip="${VOL_ARRAY[$i]}"
    echo "    ssh $SSH_USER@$ip 'tail -f $LOG_DIR/volunteer_${i}.log'"
done
echo ""
echo "  Monitoring depuis cette machine :"
echo "    python monitor.py --manager $MANAGER_IP"
echo ""
echo "  Récupérer les stats à la fin :"
echo "    python monitor.py --manager $MANAGER_IP --export resultats.json"
echo "════════════════════════════════════════════════════════════"
