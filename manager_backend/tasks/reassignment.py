"""
Réassignation des tâches échouées à des volontaires réellement en ligne.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from tasks.models import TaskStatus
from tasks.recovery import prepare_failed_tasks_for_retry, recover_pending_and_failed_work
from volunteers.presence import get_online_volunteers_data
from workflows.models import Workflow, WorkflowStatus

logger = logging.getLogger(__name__)

MAX_TASK_RETRIES = 3


def reassign_failed_tasks(
    workflow: Workflow,
    volunteers_data: Optional[List[Dict[str, Any]]] = None,
) -> dict:
    """
    Réassigne les tâches échouées (dans la limite de retry_count).
    Utilise uniquement les volontaires en ligne (heartbeat).
    """
    online = volunteers_data or get_online_volunteers_data()
    failed_count = workflow.tasks.filter(status=TaskStatus.FAILED).count()

    if failed_count == 0:
        return {"reassigned": 0, "skipped": 0}

    if not online:
        logger.warning(
            "Aucun volontaire en ligne pour réassigner le workflow %s — reste en PARTIAL_FAILURE",
            workflow.id,
        )
        if workflow.status not in (WorkflowStatus.FAILED, WorkflowStatus.PARTIAL_FAILURE):
            workflow.status = WorkflowStatus.PARTIAL_FAILURE
            workflow.save(update_fields=["status", "updated_at"])
        return {"reassigned": 0, "skipped": failed_count, "error": "no_online_volunteers"}

    prepared = prepare_failed_tasks_for_retry(workflow)
    if prepared == 0:
        return {"reassigned": 0, "skipped": failed_count}

    from tasks.assignment import assign_and_publish

    workflow.status = WorkflowStatus.REASSIGNING
    workflow.save(update_fields=["status", "updated_at"])

    result = assign_and_publish(workflow, online)
    assigned = int(result.get("assigned") or 0)

    try:
        from websocket_service.client import notify_event

        notify_event(
            "workflow_status_change",
            {
                "workflow_id": str(workflow.id),
                "status": workflow.status,
                "message": (
                    f"{assigned} tâche(s) réassignée(s)"
                    if assigned
                    else "Échecs préparés, en attente de volontaires en ligne"
                ),
            },
        )
    except Exception:
        pass

    return {
        "reassigned": assigned,
        "prepared": prepared,
        "skipped": max(0, failed_count - prepared),
        "assignments": result.get("assignment"),
    }


def retry_all_recoverable_work(
    volunteers_data: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Relance globale (appelée quand un volontaire revient en ligne)."""
    return recover_pending_and_failed_work(volunteers_data)
