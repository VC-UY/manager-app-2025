"""
Utilitaires partages : dependances entre taches, aggregation et finalisation de workflow.
"""

import json
import logging
import os
import shutil
from glob import glob

from django.utils import timezone

from tasks.models import Task, TaskStatus
from workflows.models import Workflow, WorkflowStatus, WorkflowType

logger = logging.getLogger(__name__)


def dependencies_satisfied(task: Task) -> bool:
    """Verifie que toutes les taches parentes/dependances sont terminees."""
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
    """Retourne les taches CREATED dont les dependances sont satisfaites."""
    candidates = workflow.tasks.filter(status=TaskStatus.CREATED).order_by('created_at')
    return [task for task in candidates if dependencies_satisfied(task)]


def workflow_task_counts(workflow: Workflow) -> dict:
    """Compte les taches par statut pour un workflow."""
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


def _ensure_output_dir(workflow: Workflow) -> str:
    if not workflow.output_path:
        workflow.output_path = os.path.join(workflow.executable_path or '/tmp', 'output')
        workflow.save(update_fields=['output_path', 'updated_at'])
    os.makedirs(workflow.output_path, exist_ok=True)
    return workflow.output_path


def _collect_task_files(workflow: Workflow, suffixes=None):
    """Collecte les fichiers de sortie telecharges pour les taches completees."""
    files = []
    for task in workflow.tasks.filter(status=TaskStatus.COMPLETED):
        for output_file in task.output_files or []:
            if os.path.isfile(output_file):
                if not suffixes or any(output_file.endswith(suffix) for suffix in suffixes):
                    files.append(output_file)
    # Fallback: chercher sous output_path (ignorer le staging de merge)
    output_root = _ensure_output_dir(workflow)
    if suffixes and not files:
        for suffix in suffixes:
            for path in glob(os.path.join(output_root, f'**/*{suffix}'), recursive=True):
                if '_merge_staging' in path:
                    continue
                files.append(path)
    return sorted(set(files))


def _aggregate_matrix(workflow: Workflow) -> bool:
    """Fusionne les resultats matriciels par blocs."""
    try:
        import pickle

        output_root = _ensure_output_dir(workflow)
        aggregated_dir = os.path.join(output_root, 'aggregated')
        os.makedirs(aggregated_dir, exist_ok=True)

        results = []
        for path in _collect_task_files(workflow, suffixes=['.pkl', 'result.pkl']):
            with open(path, 'rb') as handle:
                results.append(pickle.load(handle))

        if not results:
            logger.warning("Aucun resultat matriciel a agreger pour %s", workflow.id)
            return False

        combined_path = os.path.join(aggregated_dir, 'matrix_result.pkl')
        with open(combined_path, 'wb') as handle:
            pickle.dump({'blocks': len(results), 'results': results}, handle)
        logger.info("Aggregation matrice: %s blocs -> %s", len(results), combined_path)
        return True
    except Exception as exc:
        logger.error("Echec aggregation matrice pour %s: %s", workflow.id, exc)
        return False


def aggregate_workflow_results(workflow: Workflow) -> bool:
    """Agrege les resultats des taches terminees selon le type de workflow."""
    if workflow.workflow_type == WorkflowType.ML_TRAINING:
        return _aggregate_ml_training(workflow)
    if workflow.workflow_type == WorkflowType.OPEN_MALARIA:
        return _aggregate_openmalaria(workflow)
    if workflow.workflow_type in (WorkflowType.MATRIX_ADDITION, WorkflowType.MATRIX_MULTIPLICATION):
        return _aggregate_matrix(workflow)
    logger.info(
        "Aucune aggregation specifique pour le workflow %s (type %s)",
        workflow.id,
        workflow.workflow_type,
    )
    return True


def _aggregate_ml_training(workflow: Workflow) -> bool:
    try:
        from workflows.examples.distributed_training_demo.merge_models import merge_models

        output_root = _ensure_output_dir(workflow)
        model_files = _collect_task_files(workflow, suffixes=['model.pt'])
        if not model_files:
            logger.warning("Aucun model.pt a fusionner pour le workflow %s", workflow.id)
            return False

        # Copier les modeles dans des sous-dossiers pour merge_models
        staging_dir = os.path.join(output_root, '_merge_staging')
        if os.path.isdir(staging_dir):
            shutil.rmtree(staging_dir)
        os.makedirs(staging_dir, exist_ok=True)

        for index, model_file in enumerate(model_files):
            shard_dir = os.path.join(staging_dir, f'shard_{index}')
            os.makedirs(shard_dir, exist_ok=True)
            shutil.copy2(model_file, os.path.join(shard_dir, 'model.pt'))

        aggregated_dir = os.path.join(output_root, 'aggregated')
        os.makedirs(aggregated_dir, exist_ok=True)
        output_path = os.path.join(aggregated_dir, 'merged_model.pt')
        merge_info = merge_models(staging_dir, output_path)

        # Consolider les metriques
        metrics_files = _collect_task_files(workflow, suffixes=['metrics.json'])
        metrics = []
        for metrics_file in metrics_files:
            try:
                with open(metrics_file, 'r', encoding='utf-8') as handle:
                    metrics.append(json.load(handle))
            except Exception:
                continue
        summary = {
            'models_merged': merge_info.get('models_merged', len(model_files)),
            'merged_model': output_path,
            'shard_metrics': metrics,
        }
        if metrics:
            summary['mean_accuracy'] = sum(m.get('accuracy', 0) for m in metrics) / len(metrics)
        with open(os.path.join(aggregated_dir, 'summary.json'), 'w', encoding='utf-8') as handle:
            json.dump(summary, handle, indent=2)

        shutil.rmtree(staging_dir, ignore_errors=True)
        logger.info("Modeles ML fusionnes pour le workflow %s -> %s", workflow.id, output_path)
        return True
    except Exception as exc:
        logger.error("Echec de l'aggregation ML pour %s: %s", workflow.id, exc)
        return False


def _aggregate_openmalaria(workflow: Workflow) -> bool:
    """Consolide les sorties OpenMalaria et calcule un resume global."""
    try:
        output_root = _ensure_output_dir(workflow)
        aggregated_dir = os.path.join(output_root, 'aggregated')
        os.makedirs(aggregated_dir, exist_ok=True)

        output_files = _collect_task_files(workflow, suffixes=['output.txt'])
        if not output_files:
            logger.warning("Aucun output.txt OpenMalaria pour %s", workflow.id)
            return False

        summary_rows = []
        for index, output_file in enumerate(output_files):
            dest = os.path.join(aggregated_dir, f'shard_{index}_output.txt')
            shutil.copy2(output_file, dest)
            stats = {}
            with open(output_file, 'r', encoding='utf-8') as handle:
                for line in handle:
                    if '=' in line and not line.startswith('day,'):
                        key, value = line.strip().split('=', 1)
                        stats[key] = value
            summary_rows.append(stats)

        populations = [float(row.get('population', 0)) for row in summary_rows]
        total_cases = [float(row.get('total_cases', 0)) for row in summary_rows]
        prevalences = [float(row.get('prevalence', 0)) for row in summary_rows]
        summary = {
            'shards': len(summary_rows),
            'total_population': sum(populations),
            'total_cases': sum(total_cases),
            'mean_prevalence': (sum(prevalences) / len(prevalences)) if prevalences else 0,
            'shard_summaries': summary_rows,
        }
        with open(os.path.join(aggregated_dir, 'summary.json'), 'w', encoding='utf-8') as handle:
            json.dump(summary, handle, indent=2)
        logger.info("Aggregation OpenMalaria pour %s: %s shards", workflow.id, len(summary_rows))
        return True
    except Exception as exc:
        logger.error("Echec de l'aggregation OpenMalaria pour %s: %s", workflow.id, exc)
        return False


def check_and_finalize_workflow(workflow: Workflow) -> str:
    """
    Verifie l'etat global du workflow et declenche aggregation / finalisation si necessaire.
    Retourne le nouveau statut du workflow.
    """
    counts = workflow_task_counts(workflow)

    if counts['running'] > 0 or counts['pending'] > 0:
        if workflow.status != WorkflowStatus.RUNNING:
            workflow.status = WorkflowStatus.RUNNING
            workflow.save(update_fields=['status', 'updated_at'])
        return workflow.status

    if counts['failed'] > 0 and counts['completed'] < counts['total']:
        if counts['completed'] == 0:
            workflow.status = WorkflowStatus.FAILED
        else:
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
            'message': f"Aggregation des resultats du workflow {workflow.name}...",
        })

        if aggregate_workflow_results(workflow):
            workflow.status = WorkflowStatus.COMPLETED
            workflow.completed_at = timezone.now()
            workflow.save(update_fields=['status', 'completed_at', 'updated_at'])
            notify_event('workflow_status_change', {
                'workflow_id': str(workflow.id),
                'status': WorkflowStatus.COMPLETED,
                'message': f"Workflow {workflow.name} termine avec succes",
            })
        else:
            workflow.status = WorkflowStatus.FAILED
            workflow.save(update_fields=['status', 'updated_at'])
            notify_event('workflow_status_change', {
                'workflow_id': str(workflow.id),
                'status': WorkflowStatus.FAILED,
                'message': f"Echec de l'aggregation du workflow {workflow.name}",
            })

    return workflow.status
