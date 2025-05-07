#backend/tasks/services/result_processor.py


import os
import json
import logging
import requests
import numpy as np
from django.utils import timezone
from django.conf import settings
from workflows.models import Workflow, WorkflowStatus
from tasks.models import Task, TaskStatus

logger = logging.getLogger(__name__)

class ResultProcessor:
    """
    Service pour le traitement et l'agrégation des résultats des tâches matricielles
    """
    
    def __init__(self, result_directory=None):
        """
        Initialise le service de traitement des résultats
        
        Args:
            result_directory (str, optional): Répertoire pour stocker les résultats téléchargés
        """
        # Utiliser le répertoire configuré dans settings ou un par défaut
        self.result_directory = result_directory or getattr(
            settings, 'RESULT_DIRECTORY', os.path.join(settings.BASE_DIR, 'data', 'results')
        )
        # Créer le répertoire s'il n'existe pas
        os.makedirs(self.result_directory, exist_ok=True)
    
    def download_task_result(self, task, result_location):
        """
        Télécharge les résultats d'une tâche depuis un volontaire
        
        Args:
            task (Task): La tâche dont les résultats doivent être téléchargés
            result_location (dict): Informations sur l'emplacement des résultats
                                   (protocol, host, port, basePath, files)
        
        Returns:
            bool: True si les résultats ont été téléchargés avec succès, False sinon
        """
        logger.info(f"Téléchargement des résultats pour la tâche {task.id}")
        
        if not result_location:
            error_msg = "Emplacement des résultats non spécifié"
            logger.error(error_msg)
            self._mark_task_as_failed(task, error_msg)
            return False
        
        protocol = result_location.get('protocol', 'http')
        host = result_location.get('host')
        port = result_location.get('port', 80)
        base_path = result_location.get('basePath', '/')
        files = result_location.get('files', [])
        
        if not host or not files:
            error_msg = "Informations de localisation incomplètes"
            logger.error(error_msg)
            self._mark_task_as_failed(task, error_msg)
            return False
        
        # Créer un répertoire pour les résultats de cette tâche
        task_result_dir = os.path.join(
            self.result_directory, 
            str(task.workflow.id), 
            str(task.id)
        )
        os.makedirs(task_result_dir, exist_ok=True)
        
        successful_downloads = []
        failed_downloads = []
        
        for file_path in files:
            file_url = f"{protocol}://{host}:{port}{base_path}{file_path}"
            file_name = os.path.basename(file_path)
            local_path = os.path.join(task_result_dir, file_name)
            
            try:
                logger.info(f"Téléchargement du fichier {file_url}")
                
                # Télécharger le fichier
                response = requests.get(file_url, timeout=30)
                response.raise_for_status()
                
                # Enregistrer le fichier localement
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                
                successful_downloads.append({
                    'remote_path': file_path,
                    'local_path': local_path
                })
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Erreur lors du téléchargement de {file_url}: {e}")
                failed_downloads.append({
                    'remote_path': file_path,
                    'error': str(e)
                })
        
        # Mettre à jour le statut de la tâche
        if not failed_downloads:
            task.status = TaskStatus.COMPLETED
            task.results = {
                'path': task_result_dir,
                'files': [item['remote_path'] for item in successful_downloads],
                'local_files': [item['local_path'] for item in successful_downloads]
            }
            task.end_time = timezone.now()
            task.save()
            
            logger.info(f"Tous les fichiers téléchargés pour la tâche {task.id}")
            
            # Vérifier si toutes les tâches du workflow sont terminées
            self.check_workflow_completion(task.workflow)
            
            return True
        else:
            error_msg = "Échec du téléchargement de certains fichiers"
            logger.warning(f"{error_msg} pour la tâche {task.id}: {len(failed_downloads)} fichiers en échec")
            
            if successful_downloads:
                # Si certains téléchargements ont réussi, enregistrer ces résultats partiels
                task.results = {
                    'path': task_result_dir,
                    'files': [item['remote_path'] for item in successful_downloads],
                    'local_files': [item['local_path'] for item in successful_downloads],
                    'partial': True
                }
            
            task.status = TaskStatus.FAILED
            task.error_details = {
                'message': error_msg,
                'failedFiles': [item['remote_path'] for item in failed_downloads],
                'errors': [item['error'] for item in failed_downloads]
            }
            task.end_time = timezone.now()
            task.save()
            
            return False
    
    def check_workflow_completion(self, workflow):
        """
        Vérifie si toutes les tâches d'un workflow sont terminées et déclenche l'agrégation si nécessaire
        
        Args:
            workflow (Workflow): Le workflow à vérifier
            
        Returns:
            bool: True si le workflow est terminé, False sinon
        """
        # Récupérer toutes les tâches du workflow
        tasks = Task.objects.filter(workflow=workflow)
        
        # Si aucune tâche n'est associée au workflow, ne rien faire
        if not tasks:
            logger.warning(f"Aucune tâche associée au workflow {workflow.id}")
            return False
        
        # Vérifier si toutes les tâches sont terminées
        all_completed = all(task.status == TaskStatus.COMPLETED for task in tasks)
        any_failed = any(task.status == TaskStatus.FAILED for task in tasks)
        
        if all_completed:
            # Toutes les tâches sont terminées, aggréger les résultats
            workflow.status = WorkflowStatus.AGGREGATING
            workflow.save()
            
            # Agréger les résultats
            result = self.aggregate_workflow_results(workflow)
            
            if result:
                workflow.status = WorkflowStatus.COMPLETED
                workflow.completed_at = timezone.now()
                workflow.save()
                
                logger.info(f"Workflow {workflow.id} terminé avec succès")
                
                return True
            else:
                workflow.status = WorkflowStatus.FAILED
                workflow.save()
                
                logger.error(f"Échec de l'agrégation des résultats pour le workflow {workflow.id}")
                
                return False
        
        elif any_failed:
            # Certaines tâches ont échoué
            if all(task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED] for task in tasks):
                # Toutes les tâches sont soit terminées, soit en échec
                
                # Compter le nombre de tâches terminées et échouées
                completed_count = sum(1 for task in tasks if task.status == TaskStatus.COMPLETED)
                failed_count = sum(1 for task in tasks if task.status == TaskStatus.FAILED)
                total_count = len(tasks)
                
                # Si plus de la moitié des tâches ont réussi, essayer d'agréger les résultats partiels
                if completed_count > total_count // 2:
                    workflow.status = WorkflowStatus.PARTIAL_FAILURE
                    workflow.save()
                    
                    logger.warning(f"Workflow {workflow.id} partiellement échoué "
                                 f"({completed_count}/{total_count} tâches terminées)")
                    
                    # Agréger les résultats disponibles
                    result = self.aggregate_workflow_results(workflow)
                    
                    if result:
                        logger.info(f"Agrégation partielle réussie pour le workflow {workflow.id}")
                    else:
                        logger.warning(f"Échec de l'agrégation partielle pour le workflow {workflow.id}")
                    
                    return True
                else:
                    # Trop peu de tâches ont réussi, marquer le workflow comme échoué
                    workflow.status = WorkflowStatus.FAILED
                    workflow.save()
                    
                    logger.error(f"Workflow {workflow.id} échoué "
                                f"({failed_count}/{total_count} tâches échouées)")
                    
                    return False
        
        # Le workflow est toujours en cours
        pending_count = sum(1 for task in tasks if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING])
        if pending_count > 0:
            logger.info(f"Workflow {workflow.id} toujours en cours "
                       f"({pending_count}/{len(tasks)} tâches en attente ou en cours)")
        
        return False
    
    def aggregate_workflow_results(self, workflow):
        """
        Agrège les résultats de toutes les tâches d'un workflow
        
        Args:
            workflow (Workflow): Le workflow dont les résultats doivent être agrégés
            
        Returns:
            bool: True si l'agrégation a réussi, False sinon
        """
        logger.info(f"Agrégation des résultats pour le workflow {workflow.id} "
                   f"de type {workflow.workflow_type}")
        
        # Récupérer toutes les tâches terminées
        completed_tasks = Task.objects.filter(
            workflow=workflow, 
            status=TaskStatus.COMPLETED
        )
        
        # Vérifier s'il y a des tâches terminées
        if not completed_tasks:
            logger.warning(f"Aucune tâche terminée pour le workflow {workflow.id}")
            return False
        
        # Créer un répertoire pour les résultats agrégés
        workflow_result_dir = os.path.join(
            self.result_directory, 
            str(workflow.id), 
            "aggregated"
        )
        os.makedirs(workflow_result_dir, exist_ok=True)
        
        # Choisir la stratégie d'agrégation selon le type de workflow
        try:
            workflow_type = workflow.workflow_type
            
            if workflow_type in ['MATRIX_ADDITION', 'Addition de matrices']:
                result = self._aggregate_matrix_addition_results(workflow, completed_tasks, workflow_result_dir)
            elif workflow_type in ['MATRIX_MULTIPLICATION', 'Multiplication de matrices']:
                result = self._aggregate_matrix_multiplication_results(workflow, completed_tasks, workflow_result_dir)
            else:
                raise ValueError(f"Type de workflow non supporté: {workflow_type}")
            
            # Mettre à jour les métadonnées du workflow avec les informations d'agrégation
            workflow.metadata['aggregation'] = {
                'timestamp': timezone.now().isoformat(),
                'resultPath': workflow_result_dir,
                'statistics': result
            }
            
            workflow.save()
            
            logger.info(f"Résultats agrégés pour le workflow {workflow.id}")
            
            return True
        
        except Exception as e:
            logger.error(f"Erreur lors de l'agrégation des résultats: {e}")
            return False
    
    def _aggregate_matrix_addition_results(self, workflow, tasks, output_dir):
        """
        Agrège les résultats de l'addition matricielle
        Correspond à l'Algorithm 3 du document.
        
        Args:
            workflow (Workflow): Le workflow d'addition matricielle
            tasks (QuerySet): Les tâches terminées
            output_dir (str): Répertoire de sortie pour les résultats agrégés
            
        Returns:
            dict: Statistiques sur l'agrégation
        """
        logger.info(f"Agrégation des résultats d'addition matricielle pour le workflow {workflow.id}")
        
        # Récupérer les dimensions de la matrice résultante
        input_data = workflow.metadata.get('input_data', {})
        matrix_a = input_data.get('matrix_a', {})
        dimensions = matrix_a.get('dimensions', [0, 0])
        
        if not dimensions or len(dimensions) != 2:
            raise ValueError("Dimensions de la matrice non valides")
        
        n, m = dimensions  # Lignes, colonnes
        
        # Initialiser la matrice résultante avec des NaN
        result_matrix = np.full((n, m), np.nan)
        
        # Parcourir les tâches terminées
        successful_blocks = 0
        failed_blocks = 0
        
        for task in tasks:
            try:
                # Vérifier que la tâche a des résultats
                if not task.results or 'path' not in task.results:
                    logger.warning(f"Pas de résultats pour la tâche {task.id}")
                    failed_blocks += 1
                    continue
                
                # Récupérer le fichier de résultats
                result_files = task.results.get('local_files', [])
                
                if not result_files:
                    logger.warning(f"Pas de fichiers de résultats pour la tâche {task.id}")
                    failed_blocks += 1
                    continue
                
                result_file = None
                for file_path in result_files:
                    if 'addition_result.json' in file_path:
                        result_file = file_path
                        break
                
                if not result_file or not os.path.exists(result_file):
                    logger.warning(f"Fichier de résultats non trouvé pour la tâche {task.id}")
                    failed_blocks += 1
                    continue
                
                # Charger les résultats
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
                
                # Récupérer les données du bloc
                block_data = result_data.get('block_data', [])
                position = result_data.get('position', {})
                
                row_start = position.get('row_start', -1)
                row_end = position.get('row_end', -1)
                col_start = position.get('col_start', -1)
                col_end = position.get('col_end', -1)
                
                if row_start < 0 or row_end < 0 or col_start < 0 or col_end < 0:
                    logger.warning(f"Coordonnées de bloc invalides pour la tâche {task.id}")
                    failed_blocks += 1
                    continue
                
                # Convertir en tableau numpy
                try:
                    block_result = np.array(block_data)
                    
                    # Vérifier les dimensions du bloc
                    expected_shape = (row_end - row_start + 1, col_end - col_start + 1)
                    if block_result.shape != expected_shape:
                        logger.warning(f"Dimensions du bloc incorrectes pour la tâche {task.id}: "
                                      f"attendu {expected_shape}, reçu {block_result.shape}")
                        failed_blocks += 1
                        continue
                    
                    # Intégrer le bloc dans la matrice résultante
                    result_matrix[row_start:row_end+1, col_start:col_end+1] = block_result
                    
                    successful_blocks += 1
                    
                except Exception as e:
                    logger.error(f"Erreur lors du traitement du bloc pour la tâche {task.id}: {e}")
                    failed_blocks += 1
                    continue
            
            except Exception as e:
                logger.error(f"Erreur lors du traitement de la tâche {task.id}: {e}")
                failed_blocks += 1
                continue
        
        # Vérifier s'il reste des valeurs NaN (blocs manquants)
        missing_values = np.isnan(result_matrix).sum()
        missing_percentage = (missing_values / (n * m)) * 100
        
        # Enregistrer la matrice résultante
        result_path = os.path.join(output_dir, "result_matrix.json")
        
        # Remplacer les NaN par None pour la sérialisation JSON
        result_matrix_serializable = result_matrix.tolist()
        for i in range(n):
            for j in range(m):
                if np.isnan(result_matrix[i, j]):
                    result_matrix_serializable[i][j] = None
        
        with open(result_path, 'w') as f:
            json.dump({
                'matrix': result_matrix_serializable,
                'dimensions': dimensions,
                'operation': 'addition',
                'timestamp': timezone.now().isoformat()
            }, f, indent=2)
        
        # Créer un rapport sur l'agrégation
        report_path = os.path.join(output_dir, "aggregation_report.json")
        report = {
            'workflow_id': str(workflow.id),
            'operation': 'matrix_addition',
            'dimensions': dimensions,
            'successful_blocks': successful_blocks,
            'failed_blocks': failed_blocks,
            'total_blocks': successful_blocks + failed_blocks,
            'missing_values': int(missing_values),
            'missing_percentage': round(missing_percentage, 2),
            'complete': missing_values == 0,
            'result_file': result_path,
            'timestamp': timezone.now().isoformat()
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Agrégation des résultats d'addition matricielle terminée: "
                   f"{successful_blocks} blocs réussis, {failed_blocks} blocs échoués, "
                   f"{missing_values} valeurs manquantes ({missing_percentage:.2f}%)")
        
        return report
    
    def _aggregate_matrix_multiplication_results(self, workflow, tasks, output_dir):
        """
        Agrège les résultats de la multiplication matricielle
        Correspond à l'Algorithm 6 du document.
        
        Args:
            workflow (Workflow): Le workflow de multiplication matricielle
            tasks (QuerySet): Les tâches terminées
            output_dir (str): Répertoire de sortie pour les résultats agrégés
            
        Returns:
            dict: Statistiques sur l'agrégation
        """
        logger.info(f"Agrégation des résultats de multiplication matricielle pour le workflow {workflow.id}")
        
        # Récupérer les dimensions des matrices d'entrée
        input_data = workflow.metadata.get('input_data', {})
        matrix_a = input_data.get('matrix_a', {})
        matrix_b = input_data.get('matrix_b', {})
        
        dim_a = matrix_a.get('dimensions', [0, 0])
        dim_b = matrix_b.get('dimensions', [0, 0])
        
        if not dim_a or not dim_b or len(dim_a) != 2 or len(dim_b) != 2:
            raise ValueError("Dimensions des matrices non valides")
        
        # Dimensions de la matrice résultante
        n = dim_a[0]  # Lignes de A
        k = dim_a[1]  # Colonnes de A = Lignes de B
        m = dim_b[1]  # Colonnes de B
        
        if k != dim_b[0]:
            raise ValueError(f"Les matrices ne sont pas compatibles pour la multiplication: "
                           f"A({dim_a[0]}x{dim_a[1]}) et B({dim_b[0]}x{dim_b[1]})")
        
        # Initialiser la matrice résultante avec des NaN
        result_matrix = np.full((n, m), np.nan)
        
        # Parcourir les tâches terminées
        successful_blocks = 0
        failed_blocks = 0
        algorithms_used = {}
        
        for task in tasks:
            try:
                # Vérifier que la tâche a des résultats
                if not task.results or 'path' not in task.results:
                    logger.warning(f"Pas de résultats pour la tâche {task.id}")
                    failed_blocks += 1
                    continue
                
                # Récupérer le fichier de résultats
                result_files = task.results.get('local_files', [])
                
                if not result_files:
                    logger.warning(f"Pas de fichiers de résultats pour la tâche {task.id}")
                    failed_blocks += 1
                    continue
                
                result_file = None
                for file_path in result_files:
                    if 'multiplication_result.json' in file_path:
                        result_file = file_path
                        break
                
                if not result_file or not os.path.exists(result_file):
                    logger.warning(f"Fichier de résultats non trouvé pour la tâche {task.id}")
                    failed_blocks += 1
                    continue
                
                # Charger les résultats
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
                
                # Récupérer les données du bloc
                block_data = result_data.get('block_data', [])
                position = result_data.get('position', {})
                algorithm = result_data.get('algorithm', 'standard')
                
                # Enregistrer l'algorithme utilisé pour les statistiques
                algorithms_used[algorithm] = algorithms_used.get(algorithm, 0) + 1
                
                row_start = position.get('row_start', -1)
                row_end = position.get('row_end', -1)
                col_start = position.get('col_start', -1)
                col_end = position.get('col_end', -1)
                
                if row_start < 0 or row_end < 0 or col_start < 0 or col_end < 0:
                    logger.warning(f"Coordonnées de bloc invalides pour la tâche {task.id}")
                    failed_blocks += 1
                    continue
                
                # Convertir en tableau numpy
                try:
                    block_result = np.array(block_data)
                    
                    # Vérifier les dimensions du bloc
                    expected_shape = (row_end - row_start + 1, col_end - col_start + 1)
                    if block_result.shape != expected_shape:
                        logger.warning(f"Dimensions du bloc incorrectes pour la tâche {task.id}: "
                                      f"attendu {expected_shape}, reçu {block_result.shape}")
                        failed_blocks += 1
                        continue
                    
                    # Intégrer le bloc dans la matrice résultante
                    result_matrix[row_start:row_end+1, col_start:col_end+1] = block_result
                    
                    successful_blocks += 1
                    
                except Exception as e:
                    logger.error(f"Erreur lors du traitement du bloc pour la tâche {task.id}: {e}")
                    failed_blocks += 1
                    continue
            
            except Exception as e:
                logger.error(f"Erreur lors du traitement de la tâche {task.id}: {e}")
                failed_blocks += 1
                continue
        
        # Vérifier s'il reste des valeurs NaN (blocs manquants)
        missing_values = np.isnan(result_matrix).sum()
        missing_percentage = (missing_values / (n * m)) * 100
        
        # Enregistrer la matrice résultante
        result_path = os.path.join(output_dir, "result_matrix.json")
        
        # Remplacer les NaN par None pour la sérialisation JSON
        result_matrix_serializable = result_matrix.tolist()
        for i in range(n):
            for j in range(m):
                if np.isnan(result_matrix[i, j]):
                    result_matrix_serializable[i][j] = None
        
        with open(result_path, 'w') as f:
            json.dump({
                'matrix': result_matrix_serializable,
                'dimensions': [n, m],
                'operation': 'multiplication',
                'matrix_a_dimensions': dim_a,
                'matrix_b_dimensions': dim_b,
                'timestamp': timezone.now().isoformat()
            }, f, indent=2)
        
        # Créer un rapport sur l'agrégation
        report_path = os.path.join(output_dir, "aggregation_report.json")
        report = {
            'workflow_id': str(workflow.id),
            'operation': 'matrix_multiplication',
            'dimensions': [n, m],
            'successful_blocks': successful_blocks,
            'failed_blocks': failed_blocks,
            'total_blocks': successful_blocks + failed_blocks,
            'missing_values': int(missing_values),
            'missing_percentage': round(missing_percentage, 2),
            'complete': missing_values == 0,
            'algorithms_used': algorithms_used,
            'result_file': result_path,
            'timestamp': timezone.now().isoformat()
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Agrégation des résultats de multiplication matricielle terminée: "
                   f"{successful_blocks} blocs réussis, {failed_blocks} blocs échoués, "
                   f"{missing_values} valeurs manquantes ({missing_percentage:.2f}%)")
        
        return report
    
    def _mark_task_as_failed(self, task, error_message):
        """
        Marque une tâche comme échouée avec un message d'erreur
        
        Args:
            task (Task): La tâche à marquer comme échouée
            error_message (str): Message d'erreur
        """
        task.status = TaskStatus.FAILED
        task.error_details = {
            'message': error_message,
            'timestamp': timezone.now().isoformat()
        }
        task.end_time = timezone.now()
        task.save()