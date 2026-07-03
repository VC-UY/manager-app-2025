#!/usr/bin/env bash
# Construit les images Docker necessaires pour les tests ML et OpenMalaria.
# A executer sur CHAQUE machine volontaire (et optionnellement sur le manager).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
EXAMPLES_DIR="$ROOT_DIR/manager_backend/workflows/examples"

echo "==> Build vcuy-ml-train:latest (entrainement de modele)"
docker build -t vcuy-ml-train:latest "$EXAMPLES_DIR/ml_training"

echo "==> Build malaria-exp:latest (simulation malaria)"
docker build -t malaria-exp:latest "$EXAMPLES_DIR/openmalaria_worker"

echo "==> Build vcuy-matrix:latest (matrices, optionnel)"
if [ -d "$EXAMPLES_DIR/matrix_worker" ]; then
  docker build -t vcuy-matrix:latest "$EXAMPLES_DIR/matrix_worker"
fi

echo
echo "Images disponibles:"
docker images | grep -E 'vcuy-ml-train|malaria-exp|vcuy-matrix' || true
echo
echo "OK. Ces images doivent etre presentes sur la machine volontaire."
