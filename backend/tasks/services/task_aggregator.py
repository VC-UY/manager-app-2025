#backend/tasks/services/task_aggregator.py

import logging
import numpy as np
from datetime import datetime
from django.utils import timezone
from django.db import transaction

from tasks.models import Task, TaskStatus
from workflows.models import WorkflowStatus

logger = logging.getLogger(__name__)

def aggregate_matrix_results(main_task):
    """
    Agrège les résultats des sous-tâches pour une tâche principale matricielle.
    
    Args:
        main_task (Task): La tâche principale dont les sous-tâches doivent être agrégées
        
    Returns:
        dict: Résultat agrégé et méta-informations
    """
    logger.info(f"Début d'agrégation des résultats pour la tâche {main_task.id}")
    
    # Vérifier le type de workflow
    workflow_type = main_task.workflow.workflow_type
    if workflow_type in ['MATRIX_ADDITION', 'Addition de matrices']:
        return _aggregate_addition_results(main_task)
    elif workflow_type in ['MATRIX_MULTIPLICATION', 'Multiplication de matrices']:
        return _aggregate_multiplication_results(main_task)
    else:
        error_msg = f"Type de workflow non supporté pour l'agrégation: {workflow_type}"
        logger.error(error_msg)
        raise ValueError(error_msg)

@transaction.atomic
def _aggregate_addition_results(main_task):
    """
    Agrège les résultats des sous-tâches pour une addition matricielle.
    Correspond à l'Algorithm 3 du document.
    
    Args:
        main_task (Task): La tâche principale d'addition
        
    Returns:
        dict: Matrice résultante et méta-informations
    """
    # Récupérer les sous-tâches
    subtasks = Task.objects.filter(parent_task=main_task)
    
    # Compteurs pour les statistiques
    total_subtasks = subtasks.count()
    completed_subtasks = subtasks.filter(status=TaskStatus.COMPLETED).count()
    
    # Vérifier si toutes les sous-tâches sont terminées
    if completed_subtasks < total_subtasks:
        logger.warning(f"Toutes les sous-tâches ne sont pas terminées: {completed_subtasks}/{total_subtasks}")
    
    # Récupérer les dimensions de la matrice résultante
    matrix_dimensions = None
    for param in main_task.parameters:
        if param.get('name') == 'matrix_a_dimensions':
            matrix_dimensions = param.get('value')
            break
    
    if not matrix_dimensions or len(matrix_dimensions) != 2:
        error_msg = "Dimensions de la matrice non valides"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    n, m = matrix_dimensions  # Lignes, colonnes
    
    # Initialiser la matrice résultante avec des NaN
    result_matrix = np.full((n, m), np.nan)
    
    # Parcourir les sous-tâches terminées
    for subtask in subtasks.filter(status=TaskStatus.COMPLETED):
        # Extraire les coordonnées du bloc
        row_start = subtask.matrix_block_row_start
        row_end = subtask.matrix_block_row_end
        col_start = subtask.matrix_block_col_start
        col_end = subtask.matrix_block_col_end
        
        # Vérifier la validité des coordonnées
        if None in [row_start, row_end, col_start, col_end]:
            logger.warning(f"Coordonnées de bloc invalides pour la sous-tâche {subtask.id}")
            continue
        
        # Extraire le bloc de résultat
        block_result = subtask.results.get('block_data', [])
        
        if not block_result:
            logger.warning(f"Données de bloc vides pour la sous-tâche {subtask.id}")
            continue
        
        # Convertir en tableau numpy si nécessaire
        if isinstance(block_result, list):
            block_result = np.array(block_result)
        
        # Vérifier les dimensions du bloc
        block_rows, block_cols = block_result.shape
        expected_rows = row_end - row_start + 1
        expected_cols = col_end - col_start + 1
        
        if block_rows != expected_rows or block_cols != expected_cols:
            logger.warning(f"Dimensions du bloc incorrectes pour la sous-tâche {subtask.id}: "
                          f"attendu ({expected_rows}, {expected_cols}), "
                          f"reçu ({block_rows}, {block_cols})")
            continue
        
        # Intégrer le bloc dans la matrice résultante
        result_matrix[row_start:row_end+1, col_start:col_end+1] = block_result
    
    # Vérifier s'il reste des valeurs NaN (blocs manquants)
    missing_values = np.isnan(result_matrix).sum()
    if missing_values > 0:
        logger.warning(f"{missing_values} éléments manquants dans la matrice résultante")
    
    # Mettre à jour la tâche principale avec le résultat
    main_task.results = {
        'matrix_data': result_matrix.tolist(),
        'dimensions': matrix_dimensions,
        'completion_time': datetime.now().isoformat(),
        'missing_values': int(missing_values)
    }
    
    # Calculer le taux de succès
    success_rate = (completed_subtasks / total_subtasks) * 100 if total_subtasks > 0 else 0
    
    # Mettre à jour le statut de la tâche principale
    if missing_values == 0 and completed_subtasks == total_subtasks:
        main_task.set_status(TaskStatus.COMPLETED)
        main_task.progress = 100
    else:
        # Si certaines sous-tâches ne sont pas terminées, mais que toutes les valeurs sont remplies
        if missing_values == 0:
            main_task.set_status(TaskStatus.COMPLETED)
            main_task.progress = 100
        else:
            # Calcul de la progression en fonction des valeurs remplies
            filled_values = n * m - missing_values
            progress_pct = (filled_values / (n * m)) * 100
            main_task.progress = progress_pct
            
            if progress_pct >= 100:
                main_task.set_status(TaskStatus.COMPLETED)
            else:
                main_task.set_status(TaskStatus.RUNNING)
    
    main_task.end_time = timezone.now()
    main_task.save()
    
    # Vérifier si le workflow peut être marqué comme terminé
    _check_workflow_completion(main_task.workflow)
    
    return {
        'matrix': result_matrix.tolist(),
        'dimensions': matrix_dimensions,
        'missing_values': int(missing_values),
        'completed_subtasks': completed_subtasks,
        'total_subtasks': total_subtasks,
        'success_rate': success_rate
    }

@transaction.atomic
def _aggregate_multiplication_results(main_task):
    """
    Agrège les résultats des sous-tâches pour une multiplication matricielle.
    Correspond à l'Algorithm 6 du document.
    
    Args:
        main_task (Task): La tâche principale de multiplication
        
    Returns:
        dict: Matrice résultante et méta-informations
    """
    # Récupérer les sous-tâches
    subtasks = Task.objects.filter(parent_task=main_task)
    
    # Compteurs pour les statistiques
    total_subtasks = subtasks.count()
    completed_subtasks = subtasks.filter(status=TaskStatus.COMPLETED).count()
    
    # Vérifier si toutes les sous-tâches sont terminées
    if completed_subtasks < total_subtasks:
        logger.warning(f"Toutes les sous-tâches ne sont pas terminées: {completed_subtasks}/{total_subtasks}")
    
    # Récupérer les dimensions des matrices d'entrée
    matrix_a_dimensions = None
    matrix_b_dimensions = None
    for param in main_task.parameters:
        if param.get('name') == 'matrix_a_dimensions':
            matrix_a_dimensions = param.get('value')
        elif param.get('name') == 'matrix_b_dimensions':
            matrix_b_dimensions = param.get('value')
    
    if not matrix_a_dimensions or not matrix_b_dimensions:
        error_msg = "Dimensions des matrices d'entrée non valides"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Dimensions de la matrice résultante (n×m = n×k * k×m)
    n = matrix_a_dimensions[0]  # Lignes de A
    k = matrix_a_dimensions[1]  # Colonnes de A = Lignes de B
    m = matrix_b_dimensions[1]  # Colonnes de B
    
    # Initialiser la matrice résultante avec des NaN
    result_matrix = np.full((n, m), np.nan)
    
    # Parcourir les sous-tâches terminées
    for subtask in subtasks.filter(status=TaskStatus.COMPLETED):
        # Extraire les coordonnées du bloc
        row_start = subtask.matrix_block_row_start
        row_end = subtask.matrix_block_row_end
        col_start = subtask.matrix_block_col_start
        col_end = subtask.matrix_block_col_end
        
        # Vérifier la validité des coordonnées
        if None in [row_start, row_end, col_start, col_end]:
            logger.warning(f"Coordonnées de bloc invalides pour la sous-tâche {subtask.id}")
            continue
        
        # Extraire le bloc de résultat
        block_result = subtask.results.get('block_data', [])
        
        if not block_result:
            logger.warning(f"Données de bloc vides pour la sous-tâche {subtask.id}")
            continue
        
        # Convertir en tableau numpy si nécessaire
        if isinstance(block_result, list):
            block_result = np.array(block_result)
        
        # Vérifier les dimensions du bloc
        block_rows, block_cols = block_result.shape
        expected_rows = row_end - row_start + 1
        expected_cols = col_end - col_start + 1
        
        if block_rows != expected_rows or block_cols != expected_cols:
            logger.warning(f"Dimensions du bloc incorrectes pour la sous-tâche {subtask.id}: "
                          f"attendu ({expected_rows}, {expected_cols}), "
                          f"reçu ({block_rows}, {block_cols})")
            continue
        
        # Intégrer le bloc dans la matrice résultante
        result_matrix[row_start:row_end+1, col_start:col_end+1] = block_result
    
    # Vérifier s'il reste des valeurs NaN (blocs manquants)
    missing_values = np.isnan(result_matrix).sum()
    if missing_values > 0:
        logger.warning(f"{missing_values} éléments manquants dans la matrice résultante")
    
    # Mettre à jour la tâche principale avec le résultat
    result_dimensions = [n, m]
    main_task.results = {
        'matrix_data': result_matrix.tolist(),
        'dimensions': result_dimensions,
        'completion_time': datetime.now().isoformat(),
        'missing_values': int(missing_values),
        'algorithm': next((p['value'] for p in main_task.parameters if p['name'] == 'algorithm'), 'standard')
    }
    
    # Calculer le taux de succès
    success_rate = (completed_subtasks / total_subtasks) * 100 if total_subtasks > 0 else 0
    
    # Mettre à jour le statut de la tâche principale
    if missing_values == 0 and completed_subtasks == total_subtasks:
        main_task.set_status(TaskStatus.COMPLETED)
        main_task.progress = 100
    else:
        # Si certaines sous-tâches ne sont pas terminées, mais que toutes les valeurs sont remplies
        if missing_values == 0:
            main_task.set_status(TaskStatus.COMPLETED)
            main_task.progress = 100
        else:
            # Calcul de la progression en fonction des valeurs remplies
            filled_values = n * m - missing_values
            progress_pct = (filled_values / (n * m)) * 100
            main_task.progress = progress_pct
            
            if progress_pct >= 100:
                main_task.set_status(TaskStatus.COMPLETED)
            else:
                main_task.set_status(TaskStatus.RUNNING)
    
    main_task.end_time = timezone.now()
    main_task.save()
    
    # Vérifier si le workflow peut être marqué comme terminé
    _check_workflow_completion(main_task.workflow)
    
    return {
        'matrix': result_matrix.tolist(),
        'dimensions': result_dimensions,
        'missing_values': int(missing_values),
        'completed_subtasks': completed_subtasks,
        'total_subtasks': total_subtasks,
        'success_rate': success_rate
    }

def _check_workflow_completion(workflow):
    """
    Vérifie si toutes les tâches d'un workflow sont terminées et met à jour son statut
    
    Args:
        workflow: Le workflow à vérifier
    """
    # Récupérer toutes les tâches principales du workflow
    main_tasks = Task.objects.filter(workflow=workflow, parent_task=None)
    
    # Vérifier si toutes les tâches principales sont terminées
    all_completed = all(task.status == TaskStatus.COMPLETED for task in main_tasks)
    
    if all_completed:
        workflow.status = WorkflowStatus.COMPLETED
        workflow.save()
        logger.info(f"Workflow {workflow.id} marqué comme terminé")
    else:
        # Vérifier s'il y a des tâches échouées
        has_failed = any(task.status == TaskStatus.FAILED for task in main_tasks)
        
        if has_failed:
            workflow.status = WorkflowStatus.PARTIAL_FAILURE
            workflow.save()
            logger.warning(f"Workflow {workflow.id} marqué comme partiellement échoué")
        else:
            # Si des tâches sont encore en cours d'exécution, mettre à jour la progression
            running_tasks = [task for task in main_tasks if task.status in [TaskStatus.RUNNING, TaskStatus.PENDING]]
            if running_tasks:
                avg_progress = sum(task.progress for task in main_tasks) / len(main_tasks)
                workflow.progress = avg_progress
                workflow.save()
                logger.info(f"Progression du workflow {workflow.id} mise à jour: {avg_progress}%")