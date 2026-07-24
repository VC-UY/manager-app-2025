"""
Construction de bundles self-contained (.tar.gz + run.sh) pour vc-uyr.
Remplace les images Docker comme artefact d'exécution des tâches.
"""

from __future__ import annotations

import logging
import shutil
import tarfile
from pathlib import Path

logger = logging.getLogger(__name__)

RUNTIME_META = {"runtime": "vc-uyr", "bundle": True}

# Contrat Ashley : run.sh → écrit dans $vc_OUTPUT (result.txt + progress.txt).
DEFAULT_RUN_SH = """#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export OUTPUT_DIR="${vc_OUTPUT:-$SCRIPT_DIR/output}"
export INPUT_DIR="${vc_INPUT:-$SCRIPT_DIR}"
mkdir -p "$OUTPUT_DIR"

# Copie uniquement les fichiers à la racine de vc_INPUT (pas les sous-dossiers).
# Un flatten récursif écrase la stdlib Python 3.14+ (ex. compression.py vs compression/).
if [ -n "${vc_INPUT:-}" ] && [ -d "$vc_INPUT" ]; then
  find "$vc_INPUT" -maxdepth 1 -type f 2>/dev/null | while read -r src; do
    base="$(basename "$src")"
    if [ ! -f "$SCRIPT_DIR/$base" ]; then
      cp -f "$src" "$SCRIPT_DIR/$base" || true
    fi
  done
fi

__COMMAND__

# Contrat Ashley — fichiers attendus dans vc_OUTPUT
if [ ! -f "$OUTPUT_DIR/progress.txt" ]; then
  echo "100" > "$OUTPUT_DIR/progress.txt"
fi
if [ ! -f "$OUTPUT_DIR/result.txt" ]; then
  echo "ok" > "$OUTPUT_DIR/result.txt"
fi
"""


def write_run_sh(dest_dir: str | Path, command: str) -> Path:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    run_sh = dest / "run.sh"
    cmd = (command or "true").strip() or "true"
    # Permet au runtime local d'injecter le Python du venv (torch, etc.)
    if cmd.startswith("python3 "):
        cmd = "${VCUY_PYTHON:-python3} " + cmd[len("python3 ") :]
    elif cmd == "python3":
        cmd = "${VCUY_PYTHON:-python3}"
    content = DEFAULT_RUN_SH.replace("__COMMAND__", cmd)
    run_sh.write_text(content, encoding="utf-8")
    run_sh.chmod(0o755)
    return run_sh


def create_task_bundle(
    staging_dir: str | Path,
    bundle_path: str | Path,
    command: str,
    extra_files: list[str | Path] | None = None,
) -> Path:
    """
    Crée staging_dir/run.sh + archive bundle_path (.tar.gz) contenant
    tous les fichiers du staging (self-contained).
    """
    staging = Path(staging_dir)
    staging.mkdir(parents=True, exist_ok=True)
    write_run_sh(staging, command)

    if extra_files:
        for src in extra_files:
            src_path = Path(src)
            if src_path.is_file():
                shutil.copy2(src_path, staging / src_path.name)

    bundle = Path(bundle_path)
    bundle.parent.mkdir(parents=True, exist_ok=True)
    if bundle.exists():
        bundle.unlink()

    with tarfile.open(bundle, "w:gz") as tar:
        for item in sorted(staging.rglob("*")):
            if not item.is_file():
                continue
            if item.resolve() == bundle.resolve():
                continue
            tar.add(str(item), arcname=item.relative_to(staging).as_posix())

    logger.info("Bundle créé: %s (%s octets)", bundle, bundle.stat().st_size)
    return bundle


def package_files_as_bundle(
    *,
    files: list[str | Path],
    command: str,
    bundle_path: str | Path,
    worker_scripts: list[str | Path] | None = None,
) -> Path:
    """
    Empaquette des fichiers d'entrée (+ scripts worker optionnels) dans un
    bundle .tar.gz prêt pour vc-uyr.
    """
    staging = Path(str(bundle_path) + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    for src in files:
        src_path = Path(src)
        if src_path.is_file():
            shutil.copy2(src_path, staging / src_path.name)

    for src in worker_scripts or []:
        src_path = Path(src)
        if src_path.is_file():
            shutil.copy2(src_path, staging / src_path.name)

    # Normalise "python foo.py" -> "python3 foo.py"
    cmd = (command or "true").strip()
    if cmd.startswith("python "):
        cmd = "python3 " + cmd[len("python ") :]

    bundle = create_task_bundle(staging, bundle_path, cmd)
    try:
        shutil.rmtree(staging)
    except OSError:
        pass
    return bundle
