#!/usr/bin/env bash
# Relance docker compose up avec retry (évite removal/conflict de noms).
set -euo pipefail

COMPOSE_FILE="${1:?compose file}"
ENV_FILE="${2:?env file}"
shift 2
SERVICES=("$@")

MAX_ATTEMPTS="${COMPOSE_UP_RETRIES:-6}"
WAIT_SECONDS="${COMPOSE_UP_WAIT:-12}"
NO_DEPS="${COMPOSE_UP_NO_DEPS:-1}"

compose() {
  docker compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" "$@"
}

cleanup_service() {
  local svc="$1"
  compose stop "${svc}" 2>/dev/null || true
  compose rm -sf "${svc}" 2>/dev/null || true
  local cid
  cid="$(compose ps -aq "${svc}" 2>/dev/null || true)"
  if [[ -n "${cid}" ]]; then
    docker rm -f ${cid} 2>/dev/null || true
  fi
  # Noms canoniques + reliquats hashés (ex: 4962c40bbe98_deploy-coordinator-proxy-1)
  docker ps -aq --filter "name=deploy-${svc}" | xargs -r docker rm -f 2>/dev/null || true
  docker ps -aq --filter "name=${svc}-" | xargs -r docker rm -f 2>/dev/null || true
  docker ps -a --format '{{.ID}} {{.Names}}' \
    | awk -v s="${svc}" 'index($2, s) {print $1}' \
    | xargs -r docker rm -f 2>/dev/null || true
}

cleanup_stale_renames() {
  docker ps -a --format '{{.ID}} {{.Names}}' \
    | awk '/^[a-f0-9]+[ \t]+[0-9a-f]+_deploy-/ {print $1}' \
    | xargs -r docker rm -f 2>/dev/null || true
  docker ps -a --filter "label=com.docker.compose.project=deploy" --filter "status=created" -q \
    | xargs -r docker rm -f 2>/dev/null || true
  docker container prune -f --filter "label=com.docker.compose.project=deploy" 2>/dev/null || true
}

for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
  echo "=== docker compose up (tentative ${attempt}/${MAX_ATTEMPTS}) services=${SERVICES[*]} no_deps=${NO_DEPS} ==="
  UP_ARGS=(up -d --remove-orphans)
  if [[ "${NO_DEPS}" == "1" ]]; then
    UP_ARGS+=(--no-deps)
  fi
  set +e
  compose "${UP_ARGS[@]}" "${SERVICES[@]}"
  rc=$?
  set -e
  if [[ ${rc} -eq 0 ]]; then
    echo "=== docker compose up OK ==="
    exit 0
  fi

  echo "compose up échoué (rc=${rc}) — nettoyage agressif des conflits de noms..."
  for svc in "${SERVICES[@]}"; do
    cleanup_service "${svc}"
  done
  cleanup_stale_renames
  sleep "${WAIT_SECONDS}"
done

echo "Échec docker compose up après ${MAX_ATTEMPTS} tentatives"
exit 1
