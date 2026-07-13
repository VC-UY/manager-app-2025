"""
Validation stricte des workflows CUSTOM — pas de tâches factices.
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
            docker = spec.get("docker_info") or meta.get("docker_info") or {}
            uses_runtime = bool(
                docker.get("bundle")
                or docker.get("runtime") == "vc-uyr"
                or spec.get("bundle")
                or meta.get("bundle")
            )
            image = (
                docker.get("image_name")
                or (
                    f"{docker.get('name')}:{docker.get('tag', 'latest')}"
                    if docker.get("name")
                    else ""
                )
            )
            if not uses_runtime and (not image or image in ("vcuy-custom:latest", ":latest")):
                return (
                    False,
                    (
                        f"Tâche #{i + 1}: fournissez un bundle vc-uyr "
                        "(docker_info.runtime=vc-uyr / bundle=true) "
                        "ou une image legacy temporaire."
                    ),
                    meta,
                )
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

    docker = meta.get("docker_info") or {}
    if isinstance(docker, str):
        name, _, tag = docker.partition(":")
        docker = {"name": name, "tag": tag or "latest", "image_name": docker if ":" in docker else f"{docker}:latest"}
        meta["docker_info"] = docker

    uses_runtime = bool(docker.get("bundle") or docker.get("runtime") == "vc-uyr" or meta.get("bundle"))
    image_name = (
        docker.get("image_name")
        or (
            f"{docker.get('name')}:{docker.get('tag', 'latest')}"
            if docker.get("name")
            else ""
        )
    )
    if uses_runtime:
        docker["runtime"] = "vc-uyr"
        docker["bundle"] = True
        meta["docker_info"] = docker
    elif not image_name or image_name in ("vcuy-custom:latest", ":latest", "latest"):
        # Par défaut: exécution via bundle vc-uyr (plus de Docker)
        docker = {"runtime": "vc-uyr", "bundle": True}
        meta["docker_info"] = docker
    else:
        docker.setdefault("name", image_name.split(":")[0])
        docker.setdefault("tag", image_name.split(":")[-1] if ":" in image_name else "latest")
        docker["image_name"] = image_name
        meta["docker_info"] = docker

    try:
        num_tasks = int(meta.get("num_tasks") or 8)
    except (TypeError, ValueError):
        num_tasks = 8
    if num_tasks < 7 or num_tasks > 64:
        return False, "num_tasks doit être entre 7 et 64 pour un workflow personnalisé.", meta
    meta["num_tasks"] = num_tasks
    meta["command"] = command

    return True, "", meta
