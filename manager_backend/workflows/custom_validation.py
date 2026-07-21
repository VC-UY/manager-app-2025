"""
Validation stricte des workflows CUSTOM — pas de tâches factices.
Exécution uniquement via bundles runtime vc-uyr (plus de Docker).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

FAKE_COMMANDS = {
    "",
    "true",
    "false",
    'echo "vc-uy custom task"',
    "echo 'vc-uy custom task'",
    "echo vc-uy custom task",
    "echo \"vc-uy custom task\"",
}

RUNTIME_BUNDLE_META = {"runtime": "vc-uyr", "bundle": True}


def _normalize_command(cmd: str) -> str:
    return " ".join((cmd or "").strip().lower().split())


def _is_fake_command(cmd: str) -> bool:
    n = _normalize_command(cmd)
    if not n:
        return True
    if n in FAKE_COMMANDS:
        return True
    # echo seul sans script réel
    if n.startswith("echo ") and "vc-uy" in n:
        return True
    return False


def _normalize_runtime_info(docker: Any) -> Dict[str, Any]:
    """Normalise docker_info (nom legacy) vers runtime vc-uyr + bundle."""
    if isinstance(docker, str):
        # Ancien format image Docker — refus implicite en forçant vc-uyr
        return dict(RUNTIME_BUNDLE_META)
    if not isinstance(docker, dict):
        return dict(RUNTIME_BUNDLE_META)
    if docker.get("image_name") or docker.get("name"):
        # Image Docker legacy explicitement refusée
        return dict(RUNTIME_BUNDLE_META)
    out = dict(RUNTIME_BUNDLE_META)
    out.update({k: v for k, v in docker.items() if k in ("runtime", "bundle")})
    out["runtime"] = "vc-uyr"
    out["bundle"] = True
    return out


def validate_custom_metadata(metadata: Dict[str, Any] | None) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Valide et normalise les métadonnées CUSTOM.
    Retourne (ok, erreur, metadata_normalisée).
    """
    meta = dict(metadata or {})
    task_specs: List[dict] = list(meta.get("tasks") or [])

    if task_specs:
        for i, spec in enumerate(task_specs):
            if not isinstance(spec, dict):
                return False, f"Tâche #{i + 1} invalide (objet attendu).", meta
            cmd = (spec.get("command") or "").strip()
            if _is_fake_command(cmd):
                return (
                    False,
                    (
                        f"Tâche #{i + 1}: commande réelle obligatoire "
                        "(ex. python train.py). Les commandes factices sont refusées."
                    ),
                    meta,
                )
            spec["docker_info"] = _normalize_runtime_info(
                spec.get("docker_info") or meta.get("docker_info")
            )
        meta["docker_info"] = dict(RUNTIME_BUNDLE_META)
        meta["bundle"] = True
        meta["runtime"] = "vc-uyr"
        return True, "", meta

    command = (meta.get("command") or "").strip()
    if _is_fake_command(command):
        return (
            False,
            (
                "Workflow personnalisé: indiquez une commande réelle à exécuter "
                "(ex. python app.py). Impossible de créer un workflow vide."
            ),
            meta,
        )

    meta["docker_info"] = _normalize_runtime_info(meta.get("docker_info"))
    meta["bundle"] = True
    meta["runtime"] = "vc-uyr"

    try:
        num_tasks = int(meta.get("num_tasks") or 8)
    except (TypeError, ValueError):
        num_tasks = 8
    if num_tasks < 7 or num_tasks > 64:
        return False, "num_tasks doit être entre 7 et 64 pour un workflow personnalisé.", meta
    meta["num_tasks"] = num_tasks
    meta["command"] = command

    return True, "", meta
