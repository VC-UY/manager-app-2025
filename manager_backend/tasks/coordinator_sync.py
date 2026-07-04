"""Synchronise tâches et statuts workflow vers le Coordinateur (Redis)."""

from __future__ import annotations

import logging
import uuid
from typing import Iterable, Optional

from redis_communication.proxy_rpc import proxy_publish
from redis_communication.utils import get_manager_login_token

logger = logging.getLogger(__name__)


def publish_tasks_created(workflow, tasks: Iterable) -> int:
    """Publie task/created pour chaque tâche afin que le Coordinateur les affiche."""
    token = get_manager_login_token(getattr(workflow, "owner", None))
    sender = str(getattr(getattr(workflow, "owner", None), "remote_id", None) or "manager")
    count = 0
    for task in tasks:
        try:
            proxy_publish(
                "task/created",
                {
                    "task_id": str(task.id),
                    "workflow_id": str(workflow.id),
                    "name": task.name,
                    "status": task.status,
                    "command": task.command or "",
                    "description": task.description or "",
                    "required_resources": task.required_resources or {},
                    "input_files": task.input_files or [],
                    "output_files": task.output_files or [],
                    "progress": float(task.progress or 0),
                },
                token=token,
                sender_id=sender,
                request_id=str(uuid.uuid4()),
                to_volunteers=False,
            )
            count += 1
        except Exception as exc:
            logger.warning("Publication task/created échouée pour %s: %s", task.id, exc)
    logger.info("Publié %s tâche(s) vers le coordinateur (workflow %s)", count, workflow.id)
    return count


def publish_workflow_status(workflow, message: str = "") -> None:
    """Met à jour le statut du workflow côté Coordinateur."""
    token = get_manager_login_token(getattr(workflow, "owner", None))
    sender = str(getattr(getattr(workflow, "owner", None), "remote_id", None) or "manager")
    try:
        proxy_publish(
            "workflow/status_changed",
            {
                "workflow_id": str(workflow.id),
                "status": workflow.status,
                "message": message or f"Statut {workflow.status}",
                "name": workflow.name,
                "workflow_type": workflow.workflow_type,
            },
            token=token,
            sender_id=sender,
            request_id=str(uuid.uuid4()),
            to_volunteers=False,
        )
    except Exception as exc:
        logger.warning("Publication workflow/status_changed échouée: %s", exc)


def publish_task_status(
    workflow,
    task,
    *,
    volunteer_id: Optional[str] = None,
    message: str = "",
    clear_assignment: bool = False,
) -> None:
    """Synchronise le statut d'une tâche vers le Coordinateur."""
    token = get_manager_login_token(getattr(workflow, "owner", None))
    sender = str(getattr(getattr(workflow, "owner", None), "remote_id", None) or "manager")
    try:
        proxy_publish(
            "task/status_sync",
            {
                "task_id": str(task.id),
                "workflow_id": str(workflow.id),
                "status": task.status,
                "progress": float(task.progress or 0),
                "name": task.name,
                "volunteer_id": volunteer_id,
                "clear_assignment": clear_assignment,
                "message": message or f"Statut tâche {task.status}",
            },
            token=token,
            sender_id=sender,
            request_id=str(uuid.uuid4()),
            to_volunteers=False,
        )
    except Exception as exc:
        logger.warning("Publication task/status_sync échouée pour %s: %s", task.id, exc)
