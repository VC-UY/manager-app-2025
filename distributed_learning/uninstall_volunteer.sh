#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "=========================================================="
echo "  Désinstallation du nœud volontaire Distributed Learning"
echo "  Répertoire : $ROOT_DIR"
echo "=========================================================="

docker compose down --rmi all --volumes --remove-orphans || true

if [ -f .env ]; then
    echo "Suppression de .env"
    rm -f .env
fi

if [ -d results ]; then
    echo "Suppression du dossier results/"
    rm -rf results
fi

echo "Suppression des caches Python __pycache__"
find . -type d -name '__pycache__' -prune -exec rm -rf {} + || true

echo "Suppression terminée. Aucune trace du nœud volontaire ne doit subsister dans ce dossier."
