#backend/tasks/task_splitter.py

import math
import logging
import uuid
from utils.mongodb import get_collection
from workflows.models import WorkflowStatus, TaskStatus
from tasks.models import Task
from docker.services.DockerService import DockerService

from django.db import transaction
from bson.objectid import ObjectId
from datetime import datetime

logger = logging.getLogger(__name__)

class TaskSplitter:
    """Service avancé pour découper les tâches en sous-tâches pour les opérations matricielles"""
    
    @staticmethod
    @transaction.atomic
    def split_workflow(workflow):
        """
        Découpe un workflow en tâches individuelles selon son type et les conteneurise.
        Spécialisé pour les opérations matricielles (addition et multiplication).
        """
        logger.info(f"Découpage du workflow {workflow.id} de type {workflow.workflow_type}")
        
        workflow_type = TaskSplitter._map_workflow_type(workflow.workflow_type)
        logger.info(f"Original workflow type: {workflow.workflow_type}")
        logger.info(f"Mapped workflow type: {workflow_type}")
        
        try:
            tasks = TaskSplitter._process_workflow_by_type(workflow, workflow_type)
            TaskSplitter._update_workflow_status(workflow, WorkflowStatus.SPLITTING)
            TaskSplitter._containerize_tasks(tasks)
            TaskSplitter._update_workflow_status(workflow, WorkflowStatus.ASSIGNING)
            
            logger.info(f"Workflow {workflow.id} découpé en {len(tasks)} tâches et conteneurisé")
            return tasks
        except Exception as e:
            TaskSplitter._handle_workflow_error(workflow, e)
            raise

    @staticmethod
    def _map_workflow_type(workflow_type):
        """Mappe le type de workflow affiché à sa valeur technique."""
        workflow_type_mapping = {
            'MATRIX_ADDITION': 'MATRIX_ADDITION',
            'MATRIX_MULTIPLICATION': 'MATRIX_MULTIPLICATION',
            'Addition de matrices': 'MATRIX_ADDITION',
            'Multiplication de matrices': 'MATRIX_MULTIPLICATION'
        }
        return workflow_type_mapping.get(workflow_type, workflow_type)

    @staticmethod
    def _process_workflow_by_type(workflow, workflow_type):
        """Traite le workflow en fonction de son type."""
        if workflow_type == 'MATRIX_ADDITION':
            return TaskSplitter._split_matrix_addition_workflow(workflow)
        elif workflow_type == 'MATRIX_MULTIPLICATION':
            return TaskSplitter._split_matrix_multiplication_workflow(workflow)
        else:
            error_msg = f"Type de workflow non supporté: {workflow_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    @staticmethod
    def _update_workflow_status(workflow, status):
        """Met à jour le statut du workflow."""
        workflow.status = status
        workflow.save()

    @staticmethod
    def _handle_workflow_error(workflow, exception):
        """Gère les erreurs survenues lors du traitement du workflow."""
        logger.error(f"Erreur lors du découpage du workflow {workflow.id}: {str(exception)}")
        workflow.status = WorkflowStatus.FAILED
        workflow.save()
    
    @staticmethod
    def _containerize_tasks(tasks):
        """
        Conteneurise toutes les tâches créées en construisant des images Docker
        et les pousse sur Docker Hub ou le registre local.
        """
        logger.info(f"Début de la conteneurisation pour {len(tasks)} tâches")
        docker_service = DockerService()
        
        # Ne conteneuriser que les tâches feuilles (sous-tâches sans enfants)
        leaf_tasks = []
        for task in tasks:
            is_leaf = True
            for t in tasks:
                if hasattr(t, 'parent_task') and t.parent_task and str(t.parent_task.id) == str(task.id):
                    is_leaf = False
                    break
            if is_leaf:
                leaf_tasks.append(task)
        
        logger.info(f"Conteneurisation de {len(leaf_tasks)} tâches feuilles")
        
        for task in leaf_tasks:
            try:
                # Adapter la commande de la tâche selon son type de workflow
                if task.workflow.workflow_type == 'MATRIX_ADDITION':
                    task.command = f"matrix_add_{task.command.split('_')[-1]}" if '_' in task.command else "matrix_add"
                elif task.workflow.workflow_type == 'MATRIX_MULTIPLICATION':
                    task.command = f"matrix_multiply_{task.command.split('_')[-1]}" if '_' in task.command else "matrix_multiply"
                
                task.save(update_fields=['command'])
                
                logger.info(f"Conteneurisation de la tâche {task.id}, commande: {task.command}")
                
                # Construire l'image Docker pour cette tâche
                image_name = docker_service.build_image_for_task(task)
                
                # Mettre à jour la tâche avec le nom de l'image
                task.docker_image = image_name
                task.save(update_fields=['docker_image'])
                
                logger.info(f"Tâche {task.id} conteneurisée avec succès: {image_name}")
                
            except Exception as e:
                logger.error(f"Erreur lors de la conteneurisation de la tâche {task.id}: {str(e)}")
                # Ne pas faire échouer tout le processus, mais marquer la tâche comme échouée
                task.status = TaskStatus.FAILED
                task.error_details = {
                    "error": str(e),
                    "stage": "containerization"
                }
                task.save(update_fields=['status', 'error_details'])
        
        logger.info(f"Conteneurisation terminée pour {len(leaf_tasks)} tâches feuilles")
    
    @staticmethod
    def _split_matrix_addition_workflow(workflow):
        """
        Algorithme de découpage pour l'addition matricielle.
        Voir Algorithm 1 du document.
        
        Args:
            workflow (Workflow): Workflow d'addition matricielle
            
        Returns:
            list: Liste des tâches créées
        """
        metadata = workflow.metadata or {}
        input_data = metadata.get('input_data', {})
        matrix_a = input_data.get('matrix_a', {})
        matrix_b = input_data.get('matrix_b', {})
        
        # Récupérer les dimensions des matrices
        n = matrix_a.get('dimensions', [0, 0])[0]  # Lignes
        m = matrix_a.get('dimensions', [0, 0])[1]  # Colonnes
        
        # Vérifier que les matrices sont de mêmes dimensions
        if matrix_a.get('dimensions') != matrix_b.get('dimensions'):
            error_msg = "Les matrices A et B doivent avoir les mêmes dimensions pour l'addition"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Déterminer la taille optimale de bloc
        bloc_size = TaskSplitter._calculate_optimal_block_size(
            workflow.min_volunteers, 
            workflow.max_volunteers, 
            n, m
        )
        
        tasks = []
        
        # Créer une tâche principale pour coordonner les sous-tâches
        main_task = Task.objects.create(
            workflow=workflow,
            name=f"Addition de Matrices - {workflow.name}",
            description=f"Tâche principale d'addition de matrices",
            command="matrix_add_main",
            parameters=[
                {"name": "matrix_a_dimensions", "value": matrix_a.get('dimensions')},
                {"name": "matrix_b_dimensions", "value": matrix_b.get('dimensions')},
                {"name": "output_format", "value": input_data.get('output_format', 'json')}
            ],
            status=TaskStatus.PENDING,
            progress=0,
            required_resources={
                "cpu": "low",
                "memory": "256MB"
            },
            docker_image="matrix-processor:latest"
        )
        
        tasks.append(main_task)
        
        # Nombre de blocs dans chaque dimension
        num_block_rows = math.ceil(n / bloc_size)
        num_block_cols = math.ceil(m / bloc_size)
        
        # Créer des sous-tâches pour chaque bloc
        for i in range(num_block_rows):
            for j in range(num_block_cols):
                start_row = i * bloc_size
                end_row = min((i + 1) * bloc_size - 1, n - 1)
                start_col = j * bloc_size
                end_col = min((j + 1) * bloc_size - 1, m - 1)
                
                sub_task = Task.objects.create(
                    workflow=workflow,
                    name=f"Addition Matrice - Bloc [{i},{j}]",
                    description=f"Additionner le bloc de [{start_row}:{end_row}, {start_col}:{end_col}]",
                    command="matrix_add_block",
                    parameters=[
                        {"name": "matrix_a_source", "value": matrix_a.get('source')},
                        {"name": "matrix_b_source", "value": matrix_b.get('source')},
                        {"name": "output_format", "value": input_data.get('output_format', 'json')}
                    ],
                    parent_task=main_task,
                    is_subtask=True,
                    status=TaskStatus.PENDING,
                    progress=0,
                    matrix_block_row_start=start_row,
                    matrix_block_row_end=end_row,
                    matrix_block_col_start=start_col,
                    matrix_block_col_end=end_col,
                    required_resources=TaskSplitter._estimate_matrix_add_resources(end_row - start_row + 1, end_col - start_col + 1),
                    docker_image="matrix-processor:latest"
                )
                
                tasks.append(sub_task)
        
        return tasks
    
    @staticmethod
    def _split_matrix_multiplication_workflow(workflow):
        """
        Algorithme de découpage pour la multiplication matricielle.
        Voir Algorithm 4 du document.
        
        Args:
            workflow (Workflow): Workflow de multiplication matricielle
            
        Returns:
            list: Liste des tâches créées
        """
        metadata = workflow.metadata or {}
        input_data = metadata.get('input_data', {})
        matrix_a = input_data.get('matrix_a', {})
        matrix_b = input_data.get('matrix_b', {})
        algorithm = input_data.get('algorithm', 'standard')
        
        # Récupérer les dimensions des matrices
        n = matrix_a.get('dimensions', [0, 0])[0]  # Lignes de A
        k = matrix_a.get('dimensions', [0, 0])[1]  # Colonnes de A = Lignes de B
        m = matrix_b.get('dimensions', [0, 0])[1]  # Colonnes de B
        
        # Vérifier que les matrices sont compatibles pour la multiplication
        if matrix_a.get('dimensions', [0, 0])[1] != matrix_b.get('dimensions', [0, 0])[0]:
            error_msg = "Les dimensions des matrices A et B ne sont pas compatibles pour la multiplication"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Déterminer la taille optimale de bloc
        bloc_size = TaskSplitter._calculate_optimal_block_size(
            workflow.min_volunteers, 
            workflow.max_volunteers, 
            n, m
        )
        
        tasks = []
        
        # Créer une tâche principale pour coordonner les sous-tâches
        main_task = Task.objects.create(
            workflow=workflow,
            name=f"Multiplication de Matrices - {workflow.name}",
            description=f"Tâche principale de multiplication de matrices avec algorithme {algorithm}",
            command="matrix_multiply_main",
            parameters=[
                {"name": "matrix_a_dimensions", "value": matrix_a.get('dimensions')},
                {"name": "matrix_b_dimensions", "value": matrix_b.get('dimensions')},
                {"name": "algorithm", "value": algorithm},
                {"name": "output_format", "value": input_data.get('output_format', 'json')}
            ],
            status=TaskStatus.PENDING,
            progress=0,
            required_resources={
                "cpu": "medium",
                "memory": "512MB"
            },
            docker_image="matrix-processor:latest"
        )
        
        tasks.append(main_task)
        
        # Nombre de blocs dans chaque dimension du résultat
        num_block_rows = math.ceil(n / bloc_size)
        num_block_cols = math.ceil(m / bloc_size)
        
        # Créer des sous-tâches pour chaque bloc
        for i in range(num_block_rows):
            for j in range(num_block_cols):
                start_row = i * bloc_size
                end_row = min((i + 1) * bloc_size - 1, n - 1)
                start_col = j * bloc_size
                end_col = min((j + 1) * bloc_size - 1, m - 1)
                
                # Pour la multiplication, chaque tâche a besoin de lignes complètes
                # de A et de colonnes complètes de B
                sub_task = Task.objects.create(
                    workflow=workflow,
                    name=f"Multiplication Matrice - Bloc [{i},{j}]",
                    description=f"Multiplier pour obtenir le bloc de [{start_row}:{end_row}, {start_col}:{end_col}]",
                    command="matrix_multiply_block",
                    parameters=[
                        {"name": "matrix_a_source", "value": matrix_a.get('source')},
                        {"name": "matrix_b_source", "value": matrix_b.get('source')},
                        {"name": "algorithm", "value": algorithm},
                        {"name": "output_format", "value": input_data.get('output_format', 'json')}
                    ],
                    parent_task=main_task,
                    is_subtask=True,
                    status=TaskStatus.PENDING,
                    progress=0,
                    matrix_block_row_start=start_row,
                    matrix_block_row_end=end_row,
                    matrix_block_col_start=start_col,
                    matrix_block_col_end=end_col,
                    matrix_data={
                        "requires_full_rows_A": True,
                        "requires_full_cols_B": True,
                        "k_dimension": k  # Dimension commune pour multiplication
                    },
                    required_resources=TaskSplitter._estimate_matrix_multiply_resources(
                        end_row - start_row + 1, 
                        end_col - start_col + 1, 
                        k,
                        algorithm
                    ),
                    docker_image="matrix-processor:latest"
                )
                
                tasks.append(sub_task)
        
        return tasks
    
    @staticmethod
    def _calculate_optimal_block_size(min_volunteers, max_volunteers, n, m):
        """
        Calcule la taille optimale des blocs en fonction du nombre de volontaires
        et des dimensions des matrices.
        
        Args:
            min_volunteers (int): Nombre minimum de volontaires
            max_volunteers (int): Nombre maximum de volontaires
            n (int): Nombre de lignes de la matrice
            m (int): Nombre de colonnes de la matrice
            
        Returns:
            int: Taille optimale des blocs
        """
        # Estimer le nombre total d'éléments
        total_elements = n * m
        
        # Déterminer le nombre optimal de volontaires
        num_volunteers = min(
            max(min_volunteers, 1),
            min(max_volunteers, total_elements)
        )
        
        # Calculer combien d'éléments par volontaire
        elements_per_volunteer = math.ceil(total_elements / num_volunteers)
        
        # Calculer une taille de bloc carrée approximative
        block_size = math.ceil(math.sqrt(elements_per_volunteer))
        
        # S'assurer que le bloc n'est pas plus grand que la matrice elle-même
        block_size = min(block_size, n, m)
        
        return max(block_size, 1)  # Taille minimale de 1
    
    @staticmethod
    def _estimate_matrix_add_resources(rows, cols):
        """
        Estime les ressources nécessaires pour une tâche d'addition matricielle.
        
        Args:
            rows (int): Nombre de lignes du bloc
            cols (int): Nombre de colonnes du bloc
            
        Returns:
            dict: Estimation des ressources
        """
        # Estimation de base
        resources = {
            "cpu": "low",
            "memory": "256MB",
            "network": "low"
        }
        
        # Nombre total d'éléments dans le bloc
        num_elements = rows * cols
        
        # Ajuster selon le nombre d'éléments
        if num_elements > 10000:  # 100x100
            resources["cpu"] = "medium"
            resources["memory"] = "512MB"
        
        if num_elements > 1000000:  # 1000x1000
            resources["cpu"] = "high"
            resources["memory"] = "1GB"
        
        return resources
    
    @staticmethod
    def _estimate_matrix_multiply_resources(rows, cols, k, algorithm='standard'):
        """
        Estime les ressources nécessaires pour une tâche de multiplication matricielle.
        
        Args:
            rows (int): Nombre de lignes du bloc résultat
            cols (int): Nombre de colonnes du bloc résultat
            k (int): Dimension commune (colonnes de A = lignes de B)
            algorithm (str): Algorithme de multiplication utilisé
            
        Returns:
            dict: Estimation des ressources
        """
        # Estimation de base
        resources = {
            "cpu": "medium",
            "memory": "512MB",
            "network": "medium"
        }
        
        # Nombre total d'opérations dans le bloc (n*m*k pour multiplication standard)
        num_operations = rows * cols * k
        
        # Ajuster selon l'algorithme
        if algorithm in ['strassen', 'winograd']:
            resources["cpu"] = "high"
            
        # Ajuster selon le nombre d'opérations
        if num_operations > 100000:  # 10x10x1000
            resources["cpu"] = "high"
            resources["memory"] = "1GB"
        
        if num_operations > 10000000:  # 100x100x1000
            resources["cpu"] = "high"
            resources["memory"] = "2GB"
            resources["gpu"] = "preferred"  # GPU préféré pour les grandes multiplications
        
        return resources