"""
File d'attente et assignation progressive des tâches aux volontaires.

Principes :
- La soumission crée des tâches en file (CREATED) ; elle ne dépend pas des volontaires.
- Chaque volontaire reçoit des tâches tant que la somme des durées estimées
  reste ≤ sa capacité (ex. 40 min) ; le reste attend.
- Priorité workflow : les workflows urgents passent avant.
- Quand un volontaire termine ou qu'un nouveau arrive, on réassigne depuis la file.
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

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

# Workflows dont les tâches CREATED peuvent être prises dans la file globale
QUEUE_WORKFLOW_STATUSES = [
    WorkflowStatus.PENDING,
    WorkflowStatus.SUBMITTED,
    WorkflowStatus.SPLITTING,
    WorkflowStatus.ASSIGNING,
    WorkflowStatus.RUNNING,
    WorkflowStatus.PARTIAL_FAILURE,
    WorkflowStatus.REASSIGNING,
]


def _upsert_volunteers(
    volunteers_data: List[Dict[str, Any]],
) -> List[Tuple[str, Volunteer]]:
    volunteer_objs: List[Tuple[str, Volunteer]] = []
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
        # Préférences éventuelles dans le payload présence
        prefs = vdata.get("preferences")
        if prefs and isinstance(prefs, dict):
            meta = dict(volunteer.meta_info or {})
            meta["preferences"] = {**(meta.get("preferences") or {}), **prefs}
            volunteer.meta_info = meta
            volunteer.save(update_fields=["meta_info"])
        volunteer_objs.append((str(volunteer_id), volunteer))
    return volunteer_objs


def _queued_tasks_for_workflow(workflow: Workflow) -> List[Task]:
    return get_assignable_tasks(workflow)


def _global_task_queue(workflow: Optional[Workflow] = None) -> List[Task]:
    """
    File globale : priorité workflow décroissante, puis date de soumission.
    Si workflow est fourni, ne renvoie que ses tâches (en respectant le même ordre).
    """
    qs = Workflow.objects.filter(status__in=QUEUE_WORKFLOW_STATUSES)
    if workflow is not None:
        qs = qs.filter(id=workflow.id)
    # Priorité haute d'abord, puis plus ancien soumis en premier (FIFO à priorité égale)
    workflows = qs.order_by("-priority", "submitted_at", "created_at")
    tasks: List[Task] = []
    for wf in workflows:
        for task in _queued_tasks_for_workflow(wf):
            # Précharger workflow pour matching priorité / type
            task.workflow = wf
            tasks.append(task)
    return tasks


def assign_workflow_to_volunteers(
    workflow: Workflow,
    volunteers_data: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, str]]]:
    """Assigne les tâches CREATED du workflow selon capacité + préférences."""
    by_wf = assign_queued_tasks(volunteers_data, workflow=workflow)
    return by_wf.get(str(workflow.id), {})


def assign_queued_tasks(
    volunteers_data: List[Dict[str, Any]],
    workflow: Optional[Workflow] = None,
) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
    """
    Assigne depuis la file d'attente (globale ou un workflow).

    Retour : { workflow_id: { volunteer_id: [ {task_id, task_name, ...} ] } }
    """
    from volunteers.matching import (
        task_estimated_seconds,
        volunteer_can_run_task,
        volunteer_remaining_capacity_seconds,
    )

    if not volunteers_data:
        return {}

    volunteer_objs = _upsert_volunteers(volunteers_data)
    if not volunteer_objs:
        return {}

    # Budget temps restant par volontaire (None = illimité)
    remaining: Dict[str, Optional[float]] = {}
    for vid, vol in volunteer_objs:
        remaining[vid] = volunteer_remaining_capacity_seconds(vol)

    queue = _global_task_queue(workflow=workflow)
    if not queue:
        return {}

    # workflow_id -> volunteer_id -> tasks
    result: Dict[str, Dict[str, List[Dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    assigned_count = 0
    skipped_no_capacity = 0

    for task in queue:
        eligible: List[Tuple[str, Volunteer]] = []
        for vid, vol in volunteer_objs:
            budget = remaining[vid]
            if not volunteer_can_run_task(vol, task, remaining_seconds=budget):
                continue
            eligible.append((vid, vol))

        if not eligible:
            skipped_no_capacity += 1
            continue

        # Préférer le volontaire avec le plus de capacité restante (ou le moins chargé)
        def _sort_key(item: Tuple[str, Volunteer]):
            vid, _ = item
            budget = remaining[vid]
            # Illimité en dernier pour laisser de la place aux budgets serrés? Non :
            # on préfère celui qui a le plus de marge pour équilibrer.
            if budget is None:
                return (1, 0.0)
            return (0, -budget)

        eligible.sort(key=_sort_key)
        volunteer_id, volunteer = eligible[0]
        est = task_estimated_seconds(task)

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

        if remaining[volunteer_id] is not None:
            remaining[volunteer_id] = max(0.0, float(remaining[volunteer_id]) - est)

        wf_id = str(task.workflow_id)
        result[wf_id][volunteer_id].append(
            {
                "task_id": str(task.id),
                "task_name": task.name,
                "workflow_type": getattr(task.workflow, "workflow_type", ""),
            }
        )
        assigned_count += 1

    # Republication des tâches déjà ASSIGNED (même volontaire) pour le workflow ciblé
    if workflow is not None:
        wf_id = str(workflow.id)
        for task in workflow.tasks.filter(status=TaskStatus.ASSIGNED):
            link = (
                VolunteerTask.objects.filter(task=task)
                .select_related("volunteer")
                .first()
            )
            if not link or not link.volunteer.coordinator_volunteer_id:
                continue
            vid = str(link.volunteer.coordinator_volunteer_id)
            online_ids = {v for v, _ in volunteer_objs}
            if vid not in online_ids:
                continue
            entry = {
                "task_id": str(task.id),
                "task_name": task.name,
                "workflow_type": getattr(workflow, "workflow_type", ""),
            }
            existing_ids = {e["task_id"] for e in result[wf_id].get(vid, [])}
            if entry["task_id"] not in existing_ids:
                result[wf_id][vid].append(entry)

    logger.info(
        "File d'attente: %s assignée(s), %s encore en attente (capacité/préférences)",
        assigned_count,
        skipped_no_capacity,
    )
    # Convertir defaultdicts
    return {
        wf: {vid: tasks for vid, tasks in vols.items()}
        for wf, vols in result.items()
    }


def publish_assignments(
    workflow: Workflow,
    assignment_result: Dict[str, List[Dict[str, str]]],
    file_server_port: Optional[int] = None,
) -> int:
    """Publie les assignations sur Redis (bus manager → volontaires via coordinateur)."""
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
                    "workflow_type": getattr(workflow, "workflow_type", ""),
                    "parameters": task.parameters,
                    "estimated_execution_time": task.estimated_max_time,
                    "input_data": transfer,
                    "input_data_size": task.input_size,
                    "docker_information": task.docker_info or {},
                }
            )
        if not enriched:
            continue
        payload = {
            "workflow_id": str(workflow.id),
            "assignments": {volunteer_id: enriched},
        }
        # Coordinateur (Redis interne) + volontaires (proxy externe)
        for to_volunteers in (False, True):
            try:
                proxy_publish(
                    "task/assignment",
                    payload,
                    token=token,
                    sender_id=sender,
                    request_id=str(uuid.uuid4()),
                    to_volunteers=to_volunteers,
                )
            except Exception as exc:
                logger.warning(
                    "Publication task/assignment (volunteers=%s) échouée: %s",
                    to_volunteers,
                    exc,
                )
        # Aligner le statut côté coordinateur (ASSIGNED, pas PENDING)
        try:
            from tasks.coordinator_sync import publish_task_status

            for info in enriched:
                try:
                    task = Task.objects.get(id=info["task_id"])
                except Task.DoesNotExist:
                    continue
                publish_task_status(
                    workflow,
                    task,
                    volunteer_id=volunteer_id,
                    message="Tâche assignée au volontaire",
                )
        except Exception as sync_err:
            logger.warning("Sync statut assignation: %s", sync_err)

        published += len(enriched)
        logger.info(
            "Assigné %s tâche(s) du workflow %s au volontaire %s",
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
    filtered = [
        v for v in volunteers_data if str(v.get("volunteer_id") or "") in online_ids
    ]
    return filtered if filtered else online


def _queue_status_message(workflow: Workflow, assigned: int) -> str:
    pending = workflow.tasks.filter(status=TaskStatus.CREATED).count()
    total = workflow.tasks.count()
    if assigned > 0 and pending > 0:
        return (
            f"{assigned} tâche(s) assignée(s), {pending} encore en file d'attente "
            f"(capacité des volontaires ou priorité). Total : {total}."
        )
    if assigned > 0:
        return f"{assigned} tâche(s) assignée(s) aux volontaires. Total : {total}."
    if pending > 0:
        return (
            f"Workflow soumis : {pending} tâche(s) en file d'attente. "
            "Elles seront assignées dès qu'un volontaire compatible est disponible."
        )
    return "Aucune tâche à assigner pour le moment."


def assign_and_publish(
    workflow: Workflow,
    volunteers_data: Optional[List[Dict[str, Any]]],
    file_server_port: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Assigne ce qui rentre dans la capacité des volontaires en ligne.
    Le reste reste CREATED (file d'attente). Jamais d'échec pour « pas de volontaire ».
    """
    volunteers_data = _filter_online_volunteers(volunteers_data)
    pending_before = workflow.tasks.filter(status=TaskStatus.CREATED).count()

    if not volunteers_data:
        if workflow.tasks.filter(status=TaskStatus.FAILED).exists() and pending_before == 0:
            workflow.status = WorkflowStatus.PARTIAL_FAILURE
        elif workflow.status not in (
            WorkflowStatus.RUNNING,
            WorkflowStatus.PARTIAL_FAILURE,
            WorkflowStatus.FAILED,
        ):
            workflow.status = WorkflowStatus.PENDING
        workflow.save(update_fields=["status", "updated_at"])
        msg = (
            f"Workflow en file d'attente : {pending_before} tâche(s) prêtes. "
            "Aucun volontaire en ligne pour le moment — assignation dès qu'un volontaire se connecte."
        )
        return {"status": "waiting", "message": msg, "assigned": 0, "queued": pending_before}

    prev_status = workflow.status
    workflow.status = WorkflowStatus.ASSIGNING
    workflow.save(update_fields=["status", "updated_at"])

    by_wf = assign_queued_tasks(volunteers_data, workflow=workflow)
    assignment = by_wf.get(str(workflow.id), {})

    if not assignment:
        if workflow.tasks.filter(status=TaskStatus.FAILED).exists() and pending_before == 0:
            workflow.status = WorkflowStatus.PARTIAL_FAILURE
        elif prev_status in (WorkflowStatus.RUNNING, WorkflowStatus.PARTIAL_FAILURE):
            workflow.status = prev_status
        else:
            workflow.status = WorkflowStatus.PENDING
        workflow.save(update_fields=["status", "updated_at"])
        pending = workflow.tasks.filter(status=TaskStatus.CREATED).count()
        msg = (
            f"{pending} tâche(s) en file d'attente. "
            "Aucun volontaire n'a assez de capacité ou de ressources pour l'instant "
            "(ex. durée max préférée trop courte pour une tâche). "
            "Les tâches resteront en attente jusqu'à un volontaire compatible."
        )
        return {"status": "waiting", "message": msg, "assigned": 0, "queued": pending}

    count = publish_assignments(workflow, assignment, file_server_port)
    pending = workflow.tasks.filter(status=TaskStatus.CREATED).count()
    # RUNNING dès qu'au moins une tâche part ; le reste peut rester en file
    workflow.status = WorkflowStatus.RUNNING if count > 0 else WorkflowStatus.PENDING
    workflow.save(update_fields=["status", "updated_at"])
    return {
        "status": "running" if count > 0 else "waiting",
        "message": _queue_status_message(workflow, count),
        "assigned": count,
        "queued": pending,
        "assignment": assignment,
    }


def assign_all_queued_work(
    volunteers_data: Optional[List[Dict[str, Any]]] = None,
    file_server_port: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Passe la file globale (tous workflows, par priorité) et assigne selon capacité.
    Appelé à la connexion d'un volontaire / recovery.
    """
    volunteers_data = _filter_online_volunteers(volunteers_data)
    if not volunteers_data:
        return {"assigned": 0, "workflows": 0, "message": "Aucun volontaire en ligne."}

    by_wf = assign_queued_tasks(volunteers_data, workflow=None)
    total = 0
    workflows_touched = 0

    for wf_id, assignment in by_wf.items():
        try:
            wf = Workflow.objects.get(id=wf_id)
        except Workflow.DoesNotExist:
            continue
        count = publish_assignments(wf, assignment, file_server_port)
        if count:
            total += count
            workflows_touched += 1
            wf.status = WorkflowStatus.RUNNING
            wf.save(update_fields=["status", "updated_at"])
        else:
            pending = wf.tasks.filter(status=TaskStatus.CREATED).count()
            if pending and wf.status not in (
                WorkflowStatus.RUNNING,
                WorkflowStatus.PARTIAL_FAILURE,
            ):
                wf.status = WorkflowStatus.PENDING
                wf.save(update_fields=["status", "updated_at"])

    return {
        "assigned": total,
        "workflows": workflows_touched,
        "message": (
            f"{total} tâche(s) assignée(s) sur {workflows_touched} workflow(s) "
            "(file d'attente par priorité)."
            if total
            else "Aucune nouvelle assignation (file vide ou capacité insuffisante)."
        ),
    }


def try_assign_pending_workflows(volunteers_data: Optional[List[Dict[str, Any]]] = None) -> int:
    """Tente d'assigner toute la file (priorité respectée)."""
    result = assign_all_queued_work(volunteers_data)
    return int(result.get("workflows") or 0)
