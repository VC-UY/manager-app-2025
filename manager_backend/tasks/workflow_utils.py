"""
Utilitaires partagés : dépendances entre tâches, agrégation et finalisation de workflow.
"""

import logging
import os
import shutil

from django.utils import timezone

from tasks.models import Task, TaskStatus
from workflows.models import Workflow, WorkflowStatus, WorkflowType

logger = logging.getLogger(__name__)


def dependencies_satisfied(task: Task) -> bool:
    """Vérifie que toutes les tâches parentes/dépendances sont terminées."""
    dependency_ids = task.dependencies or []
    if not dependency_ids:
        return True

    completed = set(
        str(task_id)
        for task_id in Task.objects.filter(
            workflow=task.workflow,
            status=TaskStatus.COMPLETED,
        ).values_list('id', flat=True)
    )

    for dep_id in dependency_ids:
        if str(dep_id) not in completed:
            return False
    return True


def get_assignable_tasks(workflow: Workflow):
    """Retourne les tâches CREATED dont les dépendances sont satisfaites."""
    candidates = workflow.tasks.filter(status=TaskStatus.CREATED).order_by('created_at')
    return [task for task in candidates if dependencies_satisfied(task)]


def workflow_task_counts(workflow: Workflow) -> dict:
    """Compte les tâches par statut pour un workflow."""
    tasks = workflow.tasks.all()
    return {
        'total': tasks.count(),
        'completed': tasks.filter(status=TaskStatus.COMPLETED).count(),
        'failed': tasks.filter(status=TaskStatus.FAILED).count(),
        'running': tasks.filter(status__in=[TaskStatus.RUNNING, TaskStatus.ASSIGNED]).count(),
        'pending': tasks.filter(
            status__in=[TaskStatus.CREATED, TaskStatus.PENDING, TaskStatus.RETRYING]
        ).count(),
    }


def _aggregate_matrix(workflow: Workflow) -> bool:
    """Fusionne les résultats matriciels par blocs."""
    try:
        import pickle
        import numpy as np

        if not workflow.output_path:
            workflow.output_path = os.path.join(
                workflow.executable_path or '/tmp', 'output'
            )
        aggregated_dir = os.path.join(workflow.output_path, 'aggregated')
        os.makedirs(aggregated_dir, exist_ok=True)

        results = []
        for task in workflow.tasks.filter(status=TaskStatus.COMPLETED):
            for output_file in task.output_files or []:
                if os.path.isfile(output_file) and output_file.endswith('.pkl'):
                    with open(output_file, 'rb') as f:
                        results.append(pickle.load(f))

        if results:
            combined_path = os.path.join(aggregated_dir, 'matrix_result.pkl')
            with open(combined_path, 'wb') as f:
                pickle.dump({'blocks': len(results), 'results': results}, f)
        return True
    except Exception as exc:
        logger.error("Échec agrégation matrice pour %s: %s", workflow.id, exc)
        return False


def aggregate_workflow_results(workflow: Workflow) -> bool:
    """
    Agrège les résultats des tâches terminées selon le type de workflow.
    Retourne True si l'agrégation a réussi (ou n'était pas nécessaire).
    """
    if workflow.workflow_type == WorkflowType.ML_TRAINING:
        return _aggregate_ml_training(workflow)
    if workflow.workflow_type == WorkflowType.OPEN_MALARIA:
        return _aggregate_openmalaria(workflow)
    if workflow.workflow_type in (WorkflowType.MATRIX_ADDITION, WorkflowType.MATRIX_MULTIPLICATION):
        return _aggregate_matrix(workflow)
    logger.info(
        "Aucune agrégation spécifique pour le workflow %s (type %s)",
        workflow.id,
        workflow.workflow_type,
    )
    return True


def _aggregate_ml_training(workflow: Workflow) -> bool:
    try:
        from workflows.examples.distributed_training_demo.merge_models import merge_models

        if not workflow.output_path or not os.path.isdir(workflow.output_path):
            logger.warning("Chemin de sortie ML introuvable pour le workflow %s", workflow.id)
            return False

        output_path = os.path.join(workflow.output_path, 'merged_model.pt')
        merge_models(workflow.output_path, output_path)
        logger.info("Modèles ML fusionnés pour le workflow %s", workflow.id)

        for item in os.listdir(workflow.output_path):
            item_path = os.path.join(workflow.output_path, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
        return True
    except Exception as exc:
        logger.error("Échec de l'agrégation ML pour %s: %s", workflow.id, exc)
        return False


def _aggregate_openmalaria(workflow: Workflow) -> bool:
    """Consolide les fichiers de sortie OpenMalaria dans un répertoire unique."""
    try:
        if not workflow.output_path:
            return True

        aggregated_dir = os.path.join(workflow.output_path, 'aggregated')
        os.makedirs(aggregated_dir, exist_ok=True)

        for task in workflow.tasks.filter(status=TaskStatus.COMPLETED):
            for output_file in task.output_files or []:
                if os.path.isfile(output_file):
                    dest = os.path.join(aggregated_dir, os.path.basename(output_file))
                    if not os.path.exists(dest):
                        shutil.copy2(output_file, dest)
        return True
    except Exception as exc:
        logger.error("Échec de l'agrégation OpenMalaria pour %s: %s", workflow.id, exc)
        return False


def check_and_finalize_workflow(workflow: Workflow) -> str:
    """
    Vérifie l'état global du workflow et déclenche agrégation / finalisation si nécessaire.
    Retourne le nouveau statut du workflow.
    """
    counts = workflow_task_counts(workflow)

    if counts['running'] > 0 or counts['pending'] > 0:
        if workflow.status != WorkflowStatus.RUNNING:
            workflow.status = WorkflowStatus.RUNNING
            workflow.save(update_fields=['status', 'updated_at'])
        return workflow.status

    if counts['failed'] > 0:
        workflow.status = WorkflowStatus.PARTIAL_FAILURE
        workflow.save(update_fields=['status', 'updated_at'])
        return workflow.status

    if counts['completed'] == counts['total'] and counts['total'] > 0:
        workflow.status = WorkflowStatus.AGGREGATING
        workflow.save(update_fields=['status', 'updated_at'])

        from websocket_service.client import notify_event

        notify_event('workflow_status_change', {
            'workflow_id': str(workflow.id),
            'status': WorkflowStatus.AGGREGATING,
            'message': f"Agrégation des résultats du workflow {workflow.name}...",
        })

        if aggregate_workflow_results(workflow):
            workflow.status = WorkflowStatus.COMPLETED
            workflow.completed_at = timezone.now()
            workflow.save(update_fields=['status', 'completed_at', 'updated_at'])
            notify_event('workflow_status_change', {
                'workflow_id': str(workflow.id),
                'status': WorkflowStatus.COMPLETED,
                'message': f"Workflow {workflow.name} terminé avec succès",
            })
        else:
            workflow.status = WorkflowStatus.FAILED
            workflow.save(update_fields=['status', 'updated_at'])
            notify_event('workflow_status_change', {
                'workflow_id': str(workflow.id),
                'status': WorkflowStatus.FAILED,
                'message': f"Échec de l'agrégation du workflow {workflow.name}",
            })

    return workflow.status
