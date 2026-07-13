#!/usr/bin/env bash
# =============================================================================
# entrypoint.sh — Maps Docker env vars to volunteer.py CLI arguments
#
# Required environment variables:
#   VOLUNTEER_ID      — unique 0-indexed ID of this volunteer
#   COORDINATOR_HOST  — IP/hostname of the coordinator
#   MANAGER_HOST      — IP/hostname of the manager
#
# Optional (defaults match src/config.py):
#   N_VOLUNTEERS, MY_IP, DATASET, DATA_PARTITION, COMPRESSION,
#   K_NEIGHBORS, GOSSIP_INTERVAL, GOSSIP_FANOUT, LOCAL_EPOCHS,
#   BATCH_SIZE, LEARNING_RATE, NUM_CLASSES, LOG_LEVEL, STATS_DIR,
#   HEARTBEAT_INTERVAL, HEARTBEAT_TIMEOUT, SOCKET_TIMEOUT,
#   CPU_CORES, RAM_GB, NETWORK_MBPS
# =============================================================================

set -e

# ─── Validate required vars ─────────────────────────────────────────────────
if [ -z "$VOLUNTEER_ID" ]; then
    echo "ERROR: VOLUNTEER_ID is required (--id)" >&2
    exit 1
fi
if [ -z "$COORDINATOR_HOST" ]; then
    echo "ERROR: COORDINATOR_HOST is required (--coordinator)" >&2
    exit 1
fi
if [ -z "$MANAGER_HOST" ]; then
    echo "ERROR: MANAGER_HOST is required (--manager)" >&2
    exit 1
fi

# ─── Build argument list ────────────────────────────────────────────────────
ARGS=(
    --id "$VOLUNTEER_ID"
    --coordinator "$COORDINATOR_HOST"
    --manager "$MANAGER_HOST"
)

# Optional args — only passed if the env var is set
[ -n "${N_VOLUNTEERS}" ]   && ARGS+=(--n-volunteers "$N_VOLUNTEERS")
[ -n "${MY_IP}" ]           && ARGS+=(--my-ip "$MY_IP")
[ -n "${CPU_CORES}" ]       && ARGS+=(--cpu-cores "$CPU_CORES")
[ -n "${RAM_GB}" ]          && ARGS+=(--ram-gb "$RAM_GB")
[ -n "${NETWORK_MBPS}" ]    && ARGS+=(--network-mbps "$NETWORK_MBPS")

# ─── Export env vars that volunteer.py reads directly from src/config.py ────
export DATASET="${DATASET:-cifar10}"
export DATA_PARTITION="${DATA_PARTITION:-iid}"
export COMPRESSION="${COMPRESSION:-quantization}"
export K_NEIGHBORS="${K_NEIGHBORS:-3}"
export GOSSIP_INTERVAL="${GOSSIP_INTERVAL:-60}"
export GOSSIP_FANOUT="${GOSSIP_FANOUT:-1}"
export LOCAL_EPOCHS="${LOCAL_EPOCHS:-3}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export LEARNING_RATE="${LEARNING_RATE:-0.001}"
export NUM_CLASSES="${NUM_CLASSES:-10}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export STATS_DIR="${STATS_DIR:-/app/results}"
export HEARTBEAT_INTERVAL="${HEARTBEAT_INTERVAL:-10}"
export HEARTBEAT_TIMEOUT="${HEARTBEAT_TIMEOUT:-35}"
export SOCKET_TIMEOUT="${SOCKET_TIMEOUT:-60}"

# ─── Create results dir ─────────────────────────────────────────────────────
mkdir -p "$STATS_DIR"
chown volunteer:volunteer "$STATS_DIR" 2>/dev/null || true
chmod a+rwx "$STATS_DIR" 2>/dev/null || true

echo "=============================================="
echo "  Distributed Learning Volunteer $VOLUNTEER_ID"
echo "  Coordinator: $COORDINATOR_HOST:9000"
echo "  Manager:     $MANAGER_HOST:9001"
echo "  Dataset:     $DATASET ($DATA_PARTITION)"
echo "  Compression: $COMPRESSION"
echo "=============================================="

# ─── Launch ─────────────────────────────────────────────────────────────────
exec python3 /app/volunteer.py "${ARGS[@]}"
