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
    try:
        if workflow.workflow_type == WorkflowType.DISTRIBUTED_LEARNING:
            from workflows.distributed_learning_service import stop_for_workflow
            stop_for_workflow(str(workflow.id))
    except Exception as exc:
        logger.warning("Arrêt service DL: %s", exc)

    if workflow.workflow_type == WorkflowType.ML_TRAINING:
        return _aggregate_ml_training(workflow)
    if workflow.workflow_type == WorkflowType.DISTRIBUTED_LEARNING:
        return _aggregate_distributed_learning(workflow)
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
        meta = workflow.metadata or {}
        summary = {
            'paradigm': 'partition_train_aggregate',
            'methodology': (
                "Jeu de données global partitionné en shards disjoints. "
                "Entraînement local sur chaque partition puis agrégation des "
                "poids (moyenne fédérée) pour reconstruire le modèle global."
            ),
            'total_samples': meta.get('total_samples'),
            'num_partitions': meta.get('num_tasks'),
            'epochs': meta.get('epochs'),
            'models_merged': merge_info.get('models_merged', len(model_files)),
            'merged_model': output_path,
            'shard_metrics': metrics,
        }
        if metrics:
            summary['mean_accuracy'] = sum(m.get('accuracy', 0) for m in metrics) / len(metrics)
            total_samples = sum(m.get('samples', 0) for m in metrics) or 1
            summary['weighted_accuracy'] = (
                sum(m.get('accuracy', 0) * m.get('samples', 0) for m in metrics) / total_samples
            )
        with open(os.path.join(aggregated_dir, 'summary.json'), 'w', encoding='utf-8') as handle:
            json.dump(summary, handle, indent=2)

        shutil.rmtree(staging_dir, ignore_errors=True)
        logger.info("Modeles ML fusionnes pour le workflow %s -> %s", workflow.id, output_path)
        return True
    except Exception as exc:
        logger.error("Echec de l'aggregation ML pour %s: %s", workflow.id, exc)
        return False


def _aggregate_distributed_learning(workflow: Workflow) -> bool:
    """Consolide les statistiques gossip de chaque volontaire."""
    try:
        output_root = _ensure_output_dir(workflow)
        aggregated_dir = os.path.join(output_root, "aggregated")
        os.makedirs(aggregated_dir, exist_ok=True)

        summaries = _collect_task_files(workflow, suffixes=["dl_summary.json"])
        round_stats = []
        for path in summaries:
            try:
                with open(path, encoding="utf-8") as handle:
                    round_stats.append(json.load(handle))
            except Exception:
                continue

        meta = workflow.metadata or {}
        global_summary = {
            "paradigm": "gossip_distributed_learning",
            "methodology": (
                "Apprentissage distribué gossip (AD-PSGD) sur volontaires VC-UY. "
                "Chaque volontaire exécute des rounds locaux + échange de modèles "
                "via le manager DL TCP; agrégation des métriques finales."
            ),
            "n_volunteers": meta.get("n_volunteers"),
            "max_rounds": meta.get("max_rounds"),
            "model": meta.get("model"),
            "dataset": meta.get("dataset"),
            "compression": meta.get("compression"),
            "dl_manager_host": meta.get("dl_manager_host"),
            "dl_manager_port": meta.get("dl_manager_port"),
            "volunteer_summaries": round_stats,
            "runtime": "vc-uyr",
        }
        out_path = os.path.join(aggregated_dir, "global_stats.json")
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(global_summary, handle, indent=2)

        logger.info(
            "Aggregation DL workflow %s: %s résumés -> %s",
            workflow.id,
            len(round_stats),
            out_path,
        )
        return True
    except Exception as exc:
        logger.error("Echec aggregation DL pour %s: %s", workflow.id, exc)
        return False


def _aggregate_openmalaria(workflow: Workflow) -> bool:
    """
    Agrège les partitions d'une étude épidémiologique globale.

    Prévalence globale = moyenne pondérée par la taille de chaque sous-population.
    Cas totaux = somme des cas des partitions.
    """
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
            dest = os.path.join(aggregated_dir, f'partition_{index}_output.txt')
            shutil.copy2(output_file, dest)
            stats = {}
            with open(output_file, 'r', encoding='utf-8') as handle:
                for line in handle:
                    if '=' in line and not line.startswith('day,'):
                        key, value = line.strip().split('=', 1)
                        stats[key] = value
            summary_rows.append(stats)

        # Copier aussi les métriques JSON de partition si présentes
        for index, metrics_file in enumerate(
            _collect_task_files(workflow, suffixes=['partition_metrics.json'])
        ):
            shutil.copy2(
                metrics_file,
                os.path.join(aggregated_dir, f'partition_{index}_metrics.json'),
            )

        populations = [float(row.get('population', 0) or 0) for row in summary_rows]
        total_cases = [float(row.get('total_cases', 0) or 0) for row in summary_rows]
        prevalences = [float(row.get('prevalence', 0) or 0) for row in summary_rows]
        eirs = [float(row.get('eir_annual', 0) or 0) for row in summary_rows]
        total_pop = sum(populations) or 1.0

        # Moyennes pondérées par population (agrégation scientifiquement correcte)
        weighted_prevalence = sum(
            p * prev for p, prev in zip(populations, prevalences)
        ) / total_pop
        weighted_eir = sum(p * e for p, e in zip(populations, eirs)) / total_pop

        meta = workflow.metadata or {}
        summary = {
            'paradigm': 'partition_simulate_aggregate',
            'study_id': str(workflow.id),
            'study_name': meta.get('study_name'),
            'methodology': (
                "Étude globale partitionnée en sous-populations disjointes. "
                "Chaque partition est simulée indépendamment avec les mêmes "
                "paramètres épidémiologiques; l'agrégation reconstruit les "
                "indicateurs globaux par pondération selon la taille de population."
            ),
            'num_partitions': len(summary_rows),
            'total_population': sum(populations),
            'total_cases': sum(total_cases),
            'global_prevalence': weighted_prevalence,
            'global_eir_annual': weighted_eir,
            'simulation_days': meta.get('simulation_days'),
            'monte_carlo_runs': meta.get('monte_carlo_runs'),
            'partition_summaries': summary_rows,
        }
        with open(os.path.join(aggregated_dir, 'summary.json'), 'w', encoding='utf-8') as handle:
            json.dump(summary, handle, indent=2)

        # Manifeste d'étude pour la défense scientifique
        study_doc = {
            'title': meta.get('study_name') or f'OpenMalaria study {workflow.id}',
            'workflow_id': str(workflow.id),
            'paradigm': summary['paradigm'],
            'methodology': summary['methodology'],
            'global_results': {
                'total_population': summary['total_population'],
                'total_cases': summary['total_cases'],
                'global_prevalence': summary['global_prevalence'],
                'global_eir_annual': summary['global_eir_annual'],
            },
        }
        with open(os.path.join(aggregated_dir, 'global_study_report.json'), 'w', encoding='utf-8') as handle:
            json.dump(study_doc, handle, indent=2)

        logger.info(
            "Aggregation OpenMalaria %s: %s partitions, prev_globale=%.4f",
            workflow.id,
            len(summary_rows),
            weighted_prevalence,
        )
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
        if workflow.workflow_type == WorkflowType.DISTRIBUTED_LEARNING:
            try:
                from workflows.distributed_learning_service import stop_for_workflow
                stop_for_workflow(str(workflow.id))
            except Exception as exc:
                logger.warning("Arrêt service DL (échec): %s", exc)
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
