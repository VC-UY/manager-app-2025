#!/usr/bin/env bash
# Deploy Manager: JAMAIS toucher coordinator-* (conflits de noms Docker).
set -euo pipefail

COMPOSE_FILE="${1:-/opt/vc-uy/deploy/docker-compose.prod.yml}"
ENV_FILE="${2:-/opt/vc-uy/deploy/.env.production}"

# 1) Purge uniquement les noms hashés conflictuels (pas le proxy canonique)
docker ps -a --format '{{.ID}} {{.Names}}' \
  | awk '/[0-9a-f]{6,}_deploy-coordinator/ {print $1}' \
  | xargs -r docker rm -f 2>/dev/null || true
docker ps -a --format '{{.ID}} {{.Names}}' \
  | awk '$2 ~ /^[0-9a-f]+_deploy-/ {print $1}' \
  | xargs -r docker rm -f 2>/dev/null || true

# 2) Manager ne dépend plus de coordinator-* (compose sur VPS)
python3 - <<PY
from pathlib import Path
import re
p = Path("${COMPOSE_FILE}")
t = p.read_text()
pat = re.compile(
    r"(  manager-backend:[\s\S]*?)    depends_on:\n(?:      - .*\n)+",
    re.M,
)
rep = r"\1    depends_on:\n      - redis\n      - mongodb\n"
t2, n = pat.subn(rep, t, count=1)
if n:
    p.write_text(t2)
    print("patched manager-backend depends_on -> redis,mongodb")
else:
    print("manager-backend depends_on already safe or pattern missed")
PY

# 3) Up STRICTEMENT manager (pas de deps)
docker compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" \
  up -d --no-deps --remove-orphans manager-backend manager-frontend
