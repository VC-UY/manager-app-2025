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
    prépare les FAILED, assigne et publie.
    """
    from tasks.assignment import assign_and_publish, assign_workflow_to_volunteers, publish_assignments
    from volunteers.presence import (
        get_online_volunteers_data,
        release_stale_assignments,
        sweep_stale_volunteers,
    )

    sweep_stale_volunteers()
    released = release_stale_assignments()

    online = volunteers_data or get_online_volunteers_data()
    if not online:
        return {
            "online": 0,
            "released": released,
            "prepared_failed": 0,
            "assigned": 0,
        }

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
    assigned = 0

    for workflow in workflows:
        # Ne pas relancer un workflow totalement abandonné (toutes tâches max retries)
        prepared_failed += prepare_failed_tasks_for_retry(workflow)

        has_created = workflow.tasks.filter(status=TaskStatus.CREATED).exists()
        has_assigned = workflow.tasks.filter(status=TaskStatus.ASSIGNED).exists()

        if has_created:
            result = assign_and_publish(workflow, online)
            assigned += int(result.get("assigned") or 0)
            logger.info(
                "Recovery workflow %s (CREATED): %s",
                workflow.id,
                result.get("message"),
            )
        elif has_assigned:
            assignment = assign_workflow_to_volunteers(workflow, online)
            if assignment:
                count = publish_assignments(workflow, assignment)
                assigned += count
                if count:
                    workflow.status = WorkflowStatus.RUNNING
                    workflow.save(update_fields=["status", "updated_at"])
                    logger.info(
                        "Recovery workflow %s: republication %s tâche(s)",
                        workflow.id,
                        count,
                    )

    return {
        "online": len(online),
        "released": released,
        "prepared_failed": prepared_failed,
        "assigned": assigned,
    }
