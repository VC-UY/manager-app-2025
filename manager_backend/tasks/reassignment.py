"""
Réassignation des tâches échouées à de nouveaux volontaires.
"""

import logging
import uuid

from django.utils import timezone

from redis_communication.client import RedisClient
from redis_communication.utils import get_manager_login_token, build_task_file_transfer_info
from tasks.models import Task, TaskStatus
from tasks.scheduller import assign_workflow_to_volunteers
from tasks.workflow_utils import dependencies_satisfied
from volunteers.models import Volunteer, VolunteerTask
from workflows.models import Workflow, WorkflowStatus

logger = logging.getLogger(__name__)

MAX_TASK_RETRIES = 3


def _get_available_volunteers_from_coordinator(workflow: Workflow) -> list:
    """Récupère la liste des volontaires via le coordinateur."""
    from workflows.handlers import submit_workflow_handler

    success, response = submit_workflow_handler(str(workflow.id))
    if success and isinstance(response, dict):
        return response.get('volunteers', [])
    return []


def _build_task_payload(task: Task, server_port: int = None) -> dict:
    transfer = build_task_file_transfer_info(task.workflow, task, server_port)
    return {
        'task_id': str(task.id),
        'name': task.name,
        'description': task.description,
        'command': task.command,
        'dependencies': task.dependencies,
        'is_subtask': task.is_subtask,
        'status': task.status,
        'required_resources': task.required_resources,
        'attempts': task.attempts,
        'workflow_id': str(task.workflow_id),
        'parameters': task.parameters,
        'estimated_execution_time': task.estimated_max_time,
        'input_data': transfer,
        'input_data_size': task.input_size,
        'docker_information': task.docker_info or {},
    }


def _publish_assignments(workflow: Workflow, assignment_result: dict, server_port: int):
    redis_client = RedisClient.get_instance()
    token = get_manager_login_token()

    for volunteer_id, task_list in assignment_result.items():
        enriched_tasks = []
        for task_info in task_list:
            try:
                task = Task.objects.get(id=task_info['task_id'])
                enriched_tasks.append(_build_task_payload(task, server_port))
            except Task.DoesNotExist:
                logger.error("Tâche %s introuvable lors de la réassignation", task_info.get('task_id'))

        if enriched_tasks:
            redis_client.publish(
                'task/assignment',
                {
                    'workflow_id': str(workflow.id),
                    'assignments': {volunteer_id: enriched_tasks},
                },
                str(uuid.uuid4()),
                token,
                'request',
            )
            logger.info(
                "Réassignation publiée: %d tâche(s) → volontaire %s",
                len(enriched_tasks),
                volunteer_id,
            )


def reassign_failed_tasks(workflow: Workflow, volunteers_data: list | None = None) -> dict:
    """
    Réassigne les tâches échouées (dans la limite de retry_count) à de nouveaux volontaires.
    Ne réassigne que les tâches dont les dépendances sont satisfaites.
    """
    max_retries = workflow.retry_count or MAX_TASK_RETRIES
    failed_tasks = workflow.tasks.filter(status=TaskStatus.FAILED)

    if not failed_tasks.exists():
        return {'reassigned': 0, 'skipped': 0}

    if volunteers_data is None:
        volunteers_data = _get_available_volunteers_from_coordinator(workflow)

    if not volunteers_data:
        logger.warning("Aucun volontaire disponible pour la réassignation du workflow %s", workflow.id)
        workflow.status = WorkflowStatus.PARTIAL_FAILURE
        workflow.save(update_fields=['status', 'updated_at'])
        return {'reassigned': 0, 'skipped': failed_tasks.count(), 'error': 'no_volunteers'}

    workflow.status = WorkflowStatus.REASSIGNING
    workflow.save(update_fields=['status', 'updated_at'])

    reassigned = 0
    skipped = 0

    for task in failed_tasks:
        if task.attempts >= max_retries:
            logger.warning(
                "Tâche %s abandonnée après %d tentatives", task.id, task.attempts
            )
            skipped += 1
            continue

        if not dependencies_satisfied(task):
            logger.info("Tâche %s en attente de dépendances avant réassignation", task.id)
            skipped += 1
            continue

        task.increment_attempts()
        task.status = TaskStatus.RETRYING
        task.end_time = None
        task.error_details = {}
        task.save()

        VolunteerTask.objects.filter(task=task).update(status='FAILED')

        task.status = TaskStatus.CREATED
        task.save()
        reassigned += 1

    if reassigned == 0:
        return {'reassigned': 0, 'skipped': skipped}

    assignment_result = assign_workflow_to_volunteers(
        workflow,
        volunteers_data,
        algorithm='round_robin',
    )

    from tasks.file_server import start_file_server

    server_port = start_file_server(workflow)
    if server_port:
        _publish_assignments(workflow, assignment_result, server_port)

    workflow.status = WorkflowStatus.RUNNING
    workflow.save(update_fields=['status', 'updated_at'])

    from websocket_service.client import notify_event

    notify_event('workflow_status_change', {
        'workflow_id': str(workflow.id),
        'status': WorkflowStatus.REASSIGNING,
        'message': f"{reassigned} tâche(s) en cours de réassignation",
    })

    return {'reassigned': reassigned, 'skipped': skipped, 'assignments': assignment_result}
