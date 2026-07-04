"""
Assignation simple et fiable des taches aux volontaires (sans PyTorch).
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional

from django.utils import timezone

from redis_communication.proxy_rpc import proxy_publish
from redis_communication.utils import (
    build_task_file_transfer_info,
    get_manager_login_token,
)
from tasks.models import Task, TaskStatus
from tasks.workflow_utils import get_assignable_tasks
from volunteers.models import Volunteer, VolunteerTask
from workflows.models import Workflow, WorkflowStatus

logger = logging.getLogger(__name__)


def assign_workflow_to_volunteers(
    workflow: Workflow,
    volunteers_data: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, str]]]:
    """Assigne les taches CREATED en round-robin aux volontaires fournis."""
    if not volunteers_data:
        logger.info("Aucun volontaire pour le workflow %s", workflow.id)
        return {}

    volunteer_objs = []
    for vdata in volunteers_data:
        volunteer_id = vdata.get("volunteer_id")
        if not volunteer_id:
            continue
        resources = vdata.get("resources") or {}
        volunteer, _ = Volunteer.objects.update_or_create(
            coordinator_volunteer_id=str(volunteer_id),
            defaults={
                "name": vdata.get("username", f"Volontaire {volunteer_id}"),
                "cpu_cores": resources.get("cpu_cores", 1),
                "ram_mb": resources.get("memory_mb", 1024),
                "disk_gb": int(resources.get("disk_space_mb", 10240) / 1024),
                "status": "available",
                "gpu": resources.get("gpu", False),
                "ip_address": resources.get("ip_address", "0.0.0.0"),
            },
        )
        volunteer_objs.append((str(volunteer_id), volunteer))

    if not volunteer_objs:
        return {}

    assignments: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    assignable = get_assignable_tasks(workflow)

    for index, task in enumerate(assignable):
        volunteer_id, volunteer = volunteer_objs[index % len(volunteer_objs)]
        VolunteerTask.objects.update_or_create(
            volunteer=volunteer,
            task=task,
            defaults={
                "assigned_at": timezone.now(),
                "status": TaskStatus.ASSIGNED,
            },
        )
        task.status = TaskStatus.ASSIGNED
        task.save(update_fields=["status"])
        assignments[volunteer_id].append(
            {"task_id": str(task.id), "task_name": task.name}
        )

    # Inclure aussi les taches deja ASSIGNED non terminees (republication)
    for task in workflow.tasks.filter(status=TaskStatus.ASSIGNED):
        link = VolunteerTask.objects.filter(task=task).select_related("volunteer").first()
        if not link or not link.volunteer.coordinator_volunteer_id:
            # Reassigner a un volontaire disponible
            volunteer_id, volunteer = volunteer_objs[0]
            VolunteerTask.objects.update_or_create(
                volunteer=volunteer,
                task=task,
                defaults={
                    "assigned_at": timezone.now(),
                    "status": TaskStatus.ASSIGNED,
                },
            )
            vid = volunteer_id
        else:
            vid = str(link.volunteer.coordinator_volunteer_id)
        entry = {"task_id": str(task.id), "task_name": task.name}
        if entry not in assignments.get(vid, []):
            assignments[vid].append(entry)

    if not assignments:
        logger.info("Aucune tache assignable pour %s", workflow.id)
        return {}

    if assignments:
        workflow.status = WorkflowStatus.PENDING
        workflow.save(update_fields=["status", "updated_at"])

    logger.info(
        "Assignation workflow %s: %s volontaires, %s taches",
        workflow.id,
        len(assignments),
        sum(len(v) for v in assignments.values()),
    )
    return dict(assignments)


def publish_assignments(
    workflow: Workflow,
    assignment_result: Dict[str, List[Dict[str, str]]],
    file_server_port: Optional[int] = None,
) -> int:
    """Publie les assignations sur Redis (bus direct manager/volontaire)."""
    if not assignment_result:
        return 0

    token = get_manager_login_token(workflow.owner)
    sender = str(getattr(workflow.owner, "remote_id", None) or "manager")
    published = 0

    for volunteer_id, task_list in assignment_result.items():
        enriched = []
        for info in task_list:
            try:
                task = Task.objects.get(id=info["task_id"])
            except Task.DoesNotExist:
                continue
            transfer = build_task_file_transfer_info(workflow, task, file_server_port)
            enriched.append(
                {
                    "task_id": str(task.id),
                    "name": task.name,
                    "description": task.description,
                    "command": task.command,
                    "dependencies": task.dependencies,
                    "is_subtask": task.is_subtask,
                    "status": task.status,
                    "required_resources": task.required_resources,
                    "attempts": task.attempts,
                    "workflow_id": str(workflow.id),
                    "parameters": task.parameters,
                    "estimated_execution_time": task.estimated_max_time,
                    "input_data": transfer,
                    "input_data_size": task.input_size,
                    "docker_information": task.docker_info or {},
                }
            )
        if not enriched:
            continue
        proxy_publish(
            "task/assignment",
            {
                "workflow_id": str(workflow.id),
                "assignments": {volunteer_id: enriched},
            },
            token=token,
            sender_id=sender,
            request_id=str(uuid.uuid4()),
            to_volunteers=False,
        )
        published += len(enriched)
        logger.info(
            "Assigne %s taches du workflow %s au volontaire %s",
            len(enriched),
            workflow.id,
            volunteer_id,
        )
    return published


def _filter_online_volunteers(
    volunteers_data: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Ne garde que les volontaires réellement en ligne (heartbeat récent)."""
    from volunteers.presence import get_online_volunteers_data

    online = get_online_volunteers_data()
    if not volunteers_data:
        return online

    online_ids = {v["volunteer_id"] for v in online}
    # Intersection: listés par le coordinateur ET encore en ligne (heartbeat)
    filtered = [
        v for v in volunteers_data if str(v.get("volunteer_id") or "") in online_ids
    ]
    # Si le coordinateur a une liste périmée, utiliser uniquement les en-ligne
    return filtered if filtered else online


def assign_and_publish(
    workflow: Workflow,
    volunteers_data: Optional[List[Dict[str, Any]]],
    file_server_port: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Assigne et publie si des volontaires sont la.
    Sinon laisse le workflow en PENDING (attente volontaire).
    Ne leve jamais d'exception fatale pour "pas de volontaire".
    """
    volunteers_data = _filter_online_volunteers(volunteers_data)
    if not volunteers_data:
        workflow.status = WorkflowStatus.PENDING
        workflow.save(update_fields=["status", "updated_at"])
        return {
            "status": "waiting",
            "message": "Soumission OK. En attente de volontaires en ligne.",
            "assigned": 0,
        }

    workflow.status = WorkflowStatus.ASSIGNING
    workflow.save(update_fields=["status", "updated_at"])

    assignment = assign_workflow_to_volunteers(workflow, volunteers_data)
    if not assignment:
        workflow.status = WorkflowStatus.PENDING
        workflow.save(update_fields=["status", "updated_at"])
        return {
            "status": "waiting",
            "message": "Soumission OK. Aucune tache assignable pour le moment.",
            "assigned": 0,
        }

    count = publish_assignments(workflow, assignment, file_server_port)
    workflow.status = WorkflowStatus.RUNNING
    workflow.save(update_fields=["status", "updated_at"])
    return {
        "status": "running",
        "message": f"{count} tache(s) assignee(s) aux volontaires.",
        "assigned": count,
        "assignment": assignment,
    }


def try_assign_pending_workflows(volunteers_data: Optional[List[Dict[str, Any]]] = None) -> int:
    """Tente d'assigner tous les workflows en attente (PENDING/SPLITTING avec taches CREATED)."""
    if not volunteers_data:
        return 0

    pending = Workflow.objects.filter(
        status__in=[
            WorkflowStatus.PENDING,
            WorkflowStatus.SUBMITTED,
            WorkflowStatus.SPLITTING,
            WorkflowStatus.ASSIGNING,
        ]
    )
    assigned_workflows = 0
    for workflow in pending:
        if not workflow.tasks.filter(status=TaskStatus.CREATED).exists():
            continue
        result = assign_and_publish(workflow, volunteers_data)
        if result.get("assigned", 0) > 0:
            assigned_workflows += 1
    return assigned_workflows
