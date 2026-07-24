"""
Reprise automatique quand un volontaire revient en ligne :
- tâches CREATED en attente
- tâches ASSIGNED expirées
- tâches FAILED encore dans la limite de retry
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from django.utils import timezone

from tasks.models import TaskStatus
from volunteers.models import VolunteerTask
from workflows.models import Workflow, WorkflowStatus

logger = logging.getLogger(__name__)


def prepare_failed_tasks_for_retry(workflow: Workflow) -> int:
    """Remet les tâches FAILED réessayables en CREATED."""
    from tasks.workflow_utils import dependencies_satisfied

    max_retries = workflow.retry_count or 3
    prepared = 0

    for task in workflow.tasks.filter(status=TaskStatus.FAILED):
        if (task.attempts or 0) >= max_retries:
            continue
        if not dependencies_satisfied(task):
            continue

        VolunteerTask.objects.filter(task=task).exclude(status="FAILED").update(
            status="FAILED"
        )
        task.status = TaskStatus.CREATED
        task.progress = 0
        task.end_time = None
        details = dict(task.error_details or {})
        details.pop("attempts_counted", None)
        details["retry_prepared_at"] = timezone.now().isoformat()
        task.error_details = details
        task.save()
        prepared += 1
        logger.info(
            "Tâche %s préparée pour retry (attempts=%s/%s)",
            task.id,
            task.attempts,
            max_retries,
        )

    return prepared


def recover_pending_and_failed_work(
    volunteers_data: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Point d'entrée unique : présence en ligne → libère les assignations mortes,
    prépare les FAILED, republie la file vers le Coordinateur (qui assigne).

    Important: utiliser task/created (pas seulement task/status_sync).
    status_sync refuse FAILED/COMPLETED → PENDING et ignore les tâches
    absentes du Mongo coordinateur — d'où une file CD vide alors que le
    Manager a encore des CREATED/PENDING.
    """
    from tasks.coordinator_sync import (
        publish_assign_request,
        publish_tasks_created,
        publish_workflow_tasks_ready,
    )
    from tasks.models import TaskStatus
    from volunteers.presence import (
        get_online_volunteers_data,
        release_stale_assignments,
        sweep_stale_volunteers,
    )

    sweep_stale_volunteers()
    released = release_stale_assignments()

    online = volunteers_data or get_online_volunteers_data()

    statuses = [
        WorkflowStatus.PENDING,
        WorkflowStatus.SUBMITTED,
        WorkflowStatus.ASSIGNING,
        WorkflowStatus.RUNNING,
        WorkflowStatus.PARTIAL_FAILURE,
        WorkflowStatus.REASSIGNING,
        WorkflowStatus.FAILED,
    ]
    workflows = Workflow.objects.filter(status__in=statuses)

    prepared_failed = 0
    for workflow in workflows:
        prepared_failed += prepare_failed_tasks_for_retry(workflow)

    if not online:
        return {
            "online": 0,
            "released": released,
            "prepared_failed": prepared_failed,
            "assigned": 0,
            "message": "Aucun volontaire en ligne — tâches conservées en file d'attente.",
        }

    queue_statuses = (TaskStatus.CREATED, TaskStatus.PENDING, TaskStatus.RETRYING)
    synced = 0
    for workflow in workflows:
        queued = list(workflow.tasks.filter(status__in=queue_statuses))
        if not queued:
            continue
        synced += publish_tasks_created(workflow, queued)
        publish_workflow_tasks_ready(
            workflow,
            message="Recovery manager — tâches republiquées pour assignation",
        )

    publish_assign_request(message="Recovery manager — assignation demandée au coordinateur")
    logger.info(
        "Recovery: %s tâche(s) republiee(s) via task/created, assignation déléguée au coordinateur",
        synced,
    )

    return {
        "online": len(online),
        "released": released,
        "prepared_failed": prepared_failed,
        "synced": synced,
        "assigned": 0,
        "message": (
            f"{synced} tâche(s) en file — le coordinateur assignera les volontaires"
            if synced
            else "Recovery terminée — assignation par le coordinateur"
        ),
    }
