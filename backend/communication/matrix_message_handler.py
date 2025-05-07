# backend/communication/matrix_message_handler.py

import json
import logging
import threading
from django.db import transaction
from django.utils import timezone
import numpy as np
import os

# Imports pour les outils mais pas les modèles
from utils.matrix_utils import load_matrix_from_file, save_matrix_to_file
from .mqtt_client import MQTTClient
from .coordinator_api import CoordinatorAPI

logger = logging.getLogger(__name__)

class MatrixMessageHandler:
    """
    Gestionnaire de messages pour les opérations matricielles.
    S'occupe de la communication entre le Manager, le Coordinateur et les Volontaires
    pour les workflows d'addition et de multiplication de matrices.
    """
    
    def __init__(self):
        """Initialisation du gestionnaire de messages"""
        self.mqtt_client = MQTTClient()
        self.coordinator_api = CoordinatorAPI()
        self.running = False
        self.thread = None
        
        # Initialiser une map pour les callbacks de résultats
        self.result_callbacks = {}
    
    def start(self):
        """
        Démarre le gestionnaire de messages et se connecte au broker MQTT
        """
        if self.running:
            logger.warning("Le gestionnaire de messages est déjà en cours d'exécution")
            return False
        
        # Se connecter au broker MQTT
        if not self.mqtt_client.connect():
            logger.error("Impossible de se connecter au broker MQTT")
            return False
        
        # S'abonner aux topics pertinents
        self.mqtt_client.subscribe("tasks/status/#", self._handle_task_status)
        self.mqtt_client.subscribe("tasks/results/#", self._handle_task_result)
        self.mqtt_client.subscribe("volunteers/status/#", self._handle_volunteer_status)
        
        # Démarrer la surveillance en arrière-plan
        self.running = True
        self.thread = threading.Thread(target=self._monitor_workflows)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info("Gestionnaire de messages matriciels démarré")
        return True
    
    def stop(self):
        """
        Arrête le gestionnaire de messages et se déconnecte du broker MQTT
        """
        if not self.running:
            logger.warning("Le gestionnaire de messages n'est pas en cours d'exécution")
            return True
        
        # Arrêter la surveillance
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
            self.thread = None
        
        # Se déconnecter du broker MQTT
        self.mqtt_client.disconnect()
        
        logger.info("Gestionnaire de messages matriciels arrêté")
        return True
    
    def register_workflow(self, workflow):
        """
        Enregistre un workflow auprès du coordinateur
        
        Args:
            workflow (Workflow): Le workflow à enregistrer
            
        Returns:
            bool: True si l'enregistrement a réussi, False sinon
        """
        result = self.coordinator_api.register_workflow(workflow)
        return result is not None
    
    def publish_task_assignments(self, tasks):
        """
        Publie les attributions de tâches pour un ensemble de tâches
        
        Args:
            tasks (list): Liste de tâches avec leur volontaire assigné
            
        Returns:
            dict: Résultats des publications
        """
        # Import des modèles à l'intérieur de la fonction
        from tasks.models import TaskStatus
        
        results = {
            "success": 0,
            "failed": 0,
            "tasks": []
        }
        
        for task in tasks:
            if task.assigned_to:
                # Publier l'attribution
                success = self.mqtt_client.publish_task_assignment(task, task.assigned_to)
                
                # Enregistrer le résultat
                task_result = {
                    "task_id": str(task.id),
                    "volunteer_id": task.assigned_to,
                    "success": success
                }
                
                results["tasks"].append(task_result)
                
                if success:
                    results["success"] += 1
                else:
                    results["failed"] += 1
                    
                    # Mettre à jour le statut de la tâche en cas d'échec
                    if not success:
                        task.status = TaskStatus.PENDING
                        task.assigned_to = None
                        task.save(update_fields=['status', 'assigned_to'])
        
        return results
    
    def publish_workflow_status(self, workflow):
        """
        Publie le statut d'un workflow
        
        Args:
            workflow (Workflow): Le workflow à publier
            
        Returns:
            bool: True si la publication a réussi, False sinon
        """
        return self.mqtt_client.publish_workflow_status(workflow)
    
    def register_for_task_result(self, task_id, callback):
        """
        Enregistre un callback pour le résultat d'une tâche
        
        Args:
            task_id (str): ID de la tâche
            callback (callable): Fonction à appeler lors de la réception du résultat
            
        Returns:
            bool: True si l'enregistrement a réussi, False sinon
        """
        self.result_callbacks[str(task_id)] = callback
        return True
    
    def aggregate_matrix_results(self, workflow):
        """
        Agrège les résultats d'un workflow matriciel
        
        Args:
            workflow (Workflow): Le workflow dont les résultats doivent être agrégés
            
        Returns:
            str: Chemin du fichier de résultat
        """
        try:
            # Import des modèles à l'intérieur de la fonction
            from workflows.models import WorkflowStatus
            from tasks.models import Task, TaskStatus
            
            with transaction.atomic():
                # Mettre à jour le statut du workflow
                workflow.status = WorkflowStatus.AGGREGATING
                workflow.save()
                
                # Récupérer toutes les tâches terminées
                tasks = Task.objects.filter(
                    workflow=workflow,
                    status=TaskStatus.COMPLETED,
                    parent_task=None
                )
                
                # Si aucune tâche n'est terminée, lever une erreur
                if not tasks.exists():
                    raise ValueError("Aucune tâche terminée pour ce workflow")
                
                # Vérifier le type de workflow
                if workflow.workflow_type in ['MATRIX_ADDITION', 'Addition de matrices']:
                    result_file = self._aggregate_addition_results(workflow, tasks)
                elif workflow.workflow_type in ['MATRIX_MULTIPLICATION', 'Multiplication de matrices']:
                    result_file = self._aggregate_multiplication_results(workflow, tasks)
                else:
                    raise ValueError(f"Type de workflow non supporté: {workflow.workflow_type}")
                
                # Mettre à jour les métadonnées du workflow
                if not workflow.metadata:
                    workflow.metadata = {}
                
                workflow.metadata['result_file'] = result_file
                workflow.status = WorkflowStatus.COMPLETED
                workflow.completed_at = timezone.now()
                workflow.save()
                
                # Notifier le coordinateur
                self.coordinator_api.notify_matrix_result_ready(workflow.id, result_file)
                self.mqtt_client.publish_matrix_result(workflow, result_file)
                
                return result_file
        
        except Exception as e:
            logger.error(f"Unhandled exception: {e}")
            logger.error(f"Erreur lors de l'agrégation des résultats: {e}")
            
            # Mettre à jour le statut en cas d'erreur
            workflow.status = WorkflowStatus.FAILED
            if not workflow.metadata:
                workflow.metadata = {}
            workflow.metadata['aggregation_error'] = str(e)
            workflow.save()
            
            raise
    
    def _aggregate_addition_results(self, workflow, tasks):
        """
        Agrège les résultats d'une addition matricielle
        
        Args:
            workflow (Workflow): Le workflow d'addition
            tasks (QuerySet): Les tâches terminées
            
        Returns:
            str: Chemin du fichier de résultat
        """
        # Import des modèles à l'intérieur de la fonction
        from tasks.models import Task, TaskStatus
        
        # Récupérer les dimensions des matrices
        input_data = workflow.metadata.get('input_data', {})
        matrix_a = input_data.get('matrix_a', {})
        dimensions = matrix_a.get('dimensions', [0, 0])
        
        if not dimensions or len(dimensions) != 2:
            raise ValueError("Dimensions de matrice invalides")
        
        # Extraire les dimensions
        n, m = dimensions
        
        # Initialiser la matrice résultante avec des NaN
        result_matrix = np.full((n, m), np.nan)
        
        # Récupérer tous les blocs calculés
        for task in tasks:
            # Récupérer les sous-tâches
            subtasks = Task.objects.filter(
                parent_task=task,
                status=TaskStatus.COMPLETED
            )
            
            # Parcourir les sous-tâches
            for subtask in subtasks:
                # Vérifier que la sous-tâche a des résultats
                if not subtask.results:
                    logger.warning(f"La sous-tâche {subtask.id} n'a pas de résultats")
                    continue
                
                # Récupérer les coordonnées du bloc
                row_start = subtask.matrix_block_row_start
                row_end = subtask.matrix_block_row_end
                col_start = subtask.matrix_block_col_start
                col_end = subtask.matrix_block_col_end
                
                # Vérifier que les coordonnées sont valides
                if None in [row_start, row_end, col_start, col_end]:
                    logger.warning(f"Coordonnées invalides pour la sous-tâche {subtask.id}")
                    continue
                
                # Récupérer les données du bloc
                block_data = subtask.results.get('block_data')
                
                if not block_data:
                    logger.warning(f"Pas de données de bloc pour la sous-tâche {subtask.id}")
                    continue
                
                # Convertir les données en tableau numpy
                block_array = np.array(block_data)
                
                # Vérifier les dimensions du bloc
                expected_rows = row_end - row_start + 1
                expected_cols = col_end - col_start + 1
                
                if block_array.shape != (expected_rows, expected_cols):
                    logger.warning(f"Dimensions incorrectes pour la sous-tâche {subtask.id}: "
                                  f"attendu ({expected_rows}, {expected_cols}), "
                                  f"reçu {block_array.shape}")
                    continue
                
                # Insérer le bloc dans la matrice résultante
                result_matrix[row_start:row_end+1, col_start:col_end+1] = block_array
        
        # Vérifier s'il reste des valeurs NaN
        nan_count = np.isnan(result_matrix).sum()
        if nan_count > 0:
            logger.warning(f"La matrice résultante contient {nan_count} valeurs NaN")
        
        # Créer le répertoire de résultats
        result_dir = os.path.join("media", "results", str(workflow.id))
        os.makedirs(result_dir, exist_ok=True)
        
        # Enregistrer la matrice résultante
        result_file = os.path.join(result_dir, "result_matrix.npy")
        save_matrix_to_file(result_matrix, result_file)
        
        return result_file
    
    def _aggregate_multiplication_results(self, workflow, tasks):
        """
        Agrège les résultats d'une multiplication matricielle
        
        Args:
            workflow (Workflow): Le workflow de multiplication
            tasks (QuerySet): Les tâches terminées
            
        Returns:
            str: Chemin du fichier de résultat
        """
        # Import des modèles à l'intérieur de la fonction
        from tasks.models import Task, TaskStatus
        
        # Récupérer les dimensions des matrices
        input_data = workflow.metadata.get('input_data', {})
        matrix_a = input_data.get('matrix_a', {})
        matrix_b = input_data.get('matrix_b', {})
        
        dim_a = matrix_a.get('dimensions', [0, 0])
        dim_b = matrix_b.get('dimensions', [0, 0])
        
        if not dim_a or not dim_b or len(dim_a) != 2 or len(dim_b) != 2:
            raise ValueError("Dimensions de matrice invalides")
        
        # Vérifier la compatibilité des dimensions
        if dim_a[1] != dim_b[0]:
            raise ValueError(f"Dimensions incompatibles: A({dim_a[0]}x{dim_a[1]}) et B({dim_b[0]}x{dim_b[1]})")
        
        # Dimensions de la matrice résultante
        n = dim_a[0]  # Lignes de A
        m = dim_b[1]  # Colonnes de B
        
        # Initialiser la matrice résultante avec des NaN
        result_matrix = np.full((n, m), np.nan)
        
        # Récupérer tous les blocs calculés
        for task in tasks:
            # Récupérer les sous-tâches
            subtasks = Task.objects.filter(
                parent_task=task,
                status=TaskStatus.COMPLETED
            )
            
            # Parcourir les sous-tâches
            for subtask in subtasks:
                # Vérifier que la sous-tâche a des résultats
                if not subtask.results:
                    logger.warning(f"La sous-tâche {subtask.id} n'a pas de résultats")
                    continue
                
                # Récupérer les coordonnées du bloc
                row_start = subtask.matrix_block_row_start
                row_end = subtask.matrix_block_row_end
                col_start = subtask.matrix_block_col_start
                col_end = subtask.matrix_block_col_end
                
                # Vérifier que les coordonnées sont valides
                if None in [row_start, row_end, col_start, col_end]:
                    logger.warning(f"Coordonnées invalides pour la sous-tâche {subtask.id}")
                    continue
                
                # Récupérer les données du bloc
                block_data = subtask.results.get('block_data')
                
                if not block_data:
                    logger.warning(f"Pas de données de bloc pour la sous-tâche {subtask.id}")
                    continue
                
                # Convertir les données en tableau numpy
                block_array = np.array(block_data)
                
                # Vérifier les dimensions du bloc
                expected_rows = row_end - row_start + 1
                expected_cols = col_end - col_start + 1
                
                if block_array.shape != (expected_rows, expected_cols):
                    logger.warning(f"Dimensions incorrectes pour la sous-tâche {subtask.id}: "
                                  f"attendu ({expected_rows}, {expected_cols}), "
                                  f"reçu {block_array.shape}")
                    continue
                
                # Insérer le bloc dans la matrice résultante
                result_matrix[row_start:row_end+1, col_start:col_end+1] = block_array
        
        # Vérifier s'il reste des valeurs NaN
        nan_count = np.isnan(result_matrix).sum()
        if nan_count > 0:
            logger.warning(f"La matrice résultante contient {nan_count} valeurs NaN")
        
        # Créer le répertoire de résultats
        result_dir = os.path.join("media", "results", str(workflow.id))
        os.makedirs(result_dir, exist_ok=True)
        
        # Enregistrer la matrice résultante
        result_file = os.path.join(result_dir, "result_matrix.npy")
        save_matrix_to_file(result_matrix, result_file)
        
        return result_file
    
    def _handle_task_status(self, topic, message):
        """
        Gère les messages de statut des tâches
        
        Args:
            topic (str): Topic du message
            message (dict): Contenu du message
        """
        try:
            # Import des modèles à l'intérieur de la fonction
            from tasks.models import Task, TaskStatus
            
            # Extraire les informations du message
            task_id = message.get('task_id')
            status = message.get('status')
            progress = message.get('progress')
            results_location = message.get('results_location', None)
            
            if not task_id or not status:
                logger.warning(f"Message de statut incomplet: {message}")
                return
            
            # Mettre à jour la tâche dans la base de données
            with transaction.atomic():
                try:
                    task = Task.objects.get(id=task_id)
                except Task.DoesNotExist:
                    logger.warning(f"Tâche {task_id} non trouvée")
                    return
                
                # Traiter selon le statut
                if status == 'completed':
                    # Marquer la tâche comme terminée
                    task.status = TaskStatus.COMPLETED
                    task.end_time = timezone.now()
                    task.progress = 100
                    
                    # Enregistrer l'emplacement des résultats
                    if results_location:
                        task.results = {
                            'location': results_location,
                            'timestamp': timezone.now().isoformat()
                        }
                    
                    task.save()
                    
                    # Lancer le téléchargement des résultats en arrière-plan
                    threading.Thread(
                        target=self._download_task_results,
                        args=(task, results_location)
                    ).start()
                    
                elif status == 'failed':
                    # Marquer la tâche comme échouée
                    task.status = TaskStatus.FAILED
                    task.end_time = timezone.now()
                    
                    # Enregistrer les détails de l'erreur
                    if 'error_details' in message:
                        task.error_details = message.get('error_details')
                    
                    task.save()
                
                # Appeler le callback enregistré pour cette tâche, s'il existe
                callback = self.result_callbacks.get(str(task_id))
                if callback:
                    callback(task, message)
                
                # Vérifier si toutes les tâches du workflow sont terminées
                self._check_workflow_completion(task.workflow)
        
        except Exception as e:
            logger.error(f"Erreur lors du traitement du message de résultat: {e}")
    
    def _handle_volunteer_status(self, topic, message):
        """
        Gère les messages de statut des volontaires
        
        Args:
            topic (str): Topic du message
            message (dict): Contenu du message
        """
        try:
            # Extraire les informations du message
            volunteer_id = message.get('volunteer_id')
            status = message.get('status')
            
            if not volunteer_id or not status:
                logger.warning(f"Message de statut de volontaire incomplet: {message}")
                return
            
            # Traiter selon le statut
            if status == 'offline':
                # Le volontaire est hors ligne, réattribuer ses tâches
                self._reassign_volunteer_tasks(volunteer_id)
        
        except Exception as e:
            logger.error(f"Erreur lors du traitement du message de statut de volontaire: {e}")
    
    def _download_task_results(self, task, location):
        """
        Télécharge les résultats d'une tâche
        
        Args:
            task (Task): La tâche concernée
            location (dict): Emplacement des résultats
        """
        try:
            if not location:
                logger.warning(f"Emplacement des résultats non spécifié pour la tâche {task.id}")
                return
            
            # Extraire les informations d'emplacement
            protocol = location.get('protocol', 'http')
            host = location.get('host')
            port = location.get('port', 80)
            base_path = location.get('basePath', '/')
            files = location.get('files', [])
            
            if not host or not files:
                logger.warning(f"Informations d'emplacement incomplètes pour la tâche {task.id}")
                return
            
            # Créer le répertoire pour les résultats
            task_result_dir = os.path.join("media", "tasks", str(task.workflow.id), str(task.id))
            os.makedirs(task_result_dir, exist_ok=True)
            
            # Télécharger chaque fichier
            downloaded_files = []
            
            for file_path in files:
                # Construire l'URL
                file_url = f"{protocol}://{host}:{port}{base_path}{file_path}"
                file_name = os.path.basename(file_path)
                local_path = os.path.join(task_result_dir, file_name)
                
                try:
                    # Télécharger le fichier
                    import requests
                    response = requests.get(file_url, timeout=30)
                    response.raise_for_status()
                    
                    # Enregistrer le fichier
                    with open(local_path, 'wb') as f:
                        f.write(response.content)
                    
                    downloaded_files.append({
                        'remote_path': file_path,
                        'local_path': local_path
                    })
                
                except Exception as e:
                    logger.error(f"Erreur lors du téléchargement du fichier {file_url}: {e}")
            
            # Mettre à jour les résultats de la tâche
            if downloaded_files:
                task.results['downloaded_files'] = downloaded_files
                task.save(update_fields=['results'])
                
                # Traiter les résultats si c'est une tâche de bloc matriciel
                if task.matrix_block_row_start is not None:
                    self._process_matrix_block_result(task, downloaded_files)
            
        except Exception as e:
            logger.error(f"Erreur lors du téléchargement des résultats de la tâche {task.id}: {e}")
    
    def _process_matrix_block_result(self, task, downloaded_files):
        """
        Traite les résultats d'un bloc matriciel
        
        Args:
            task (Task): La tâche concernée
            downloaded_files (list): Fichiers téléchargés
        """
        try:
            # Rechercher le fichier de résultat
            result_file = None
            for file_info in downloaded_files:
                if 'result' in file_info['remote_path'] or 'block' in file_info['remote_path']:
                    result_file = file_info['local_path']
                    break
            
            if not result_file:
                logger.warning(f"Fichier de résultat non trouvé pour la tâche {task.id}")
                return
            
            # Charger le fichier selon son extension
            ext = os.path.splitext(result_file)[1].lower()
            
            if ext == '.npy':
                # Fichier NumPy binaire
                block_data = np.load(result_file).tolist()
            elif ext == '.json':
                # Fichier JSON
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    # Extraire le bloc de données
                    if isinstance(data, dict) and 'block_data' in data:
                        block_data = data['block_data']
                    else:
                        block_data = data
            else:
                logger.warning(f"Format de fichier non supporté: {ext}")
                return
            
            # Mettre à jour les résultats
            if not task.results:
                task.results = {}
            
            task.results['block_data'] = block_data
            task.save(update_fields=['results'])
            
            logger.info(f"Résultats de bloc matriciel traités pour la tâche {task.id}")
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement des résultats de bloc matriciel: {e}")
    
    def _reassign_volunteer_tasks(self, volunteer_id):
        """
        Réattribue les tâches d'un volontaire hors ligne
        
        Args:
            volunteer_id (str): ID du volontaire
        """
        try:
            # Import des modèles à l'intérieur de la fonction
            from tasks.models import Task, TaskStatus
            
            # Récupérer les tâches en cours d'exécution par ce volontaire
            tasks = Task.objects.filter(
                assigned_to=volunteer_id,
                status=TaskStatus.RUNNING
            )
            
            logger.info(f"Réattribution de {tasks.count()} tâches du volontaire {volunteer_id}")
            
            # Marquer ces tâches comme en attente
            for task in tasks:
                task.status = TaskStatus.PENDING
                task.assigned_to = None
                task.save(update_fields=['status', 'assigned_to'])
                
                # Notifier le coordinateur
                self.coordinator_api.update_task_status(
                    task_id=str(task.id),
                    status='pending'
                )
        
        except Exception as e:
            logger.error(f"Erreur lors de la réattribution des tâches du volontaire {volunteer_id}: {e}")
    
    def _check_workflow_completion(self, workflow):
        """
        Vérifie si toutes les tâches d'un workflow sont terminées
        
        Args:
            workflow (Workflow): Le workflow à vérifier
        """
        try:
            # Import des modèles à l'intérieur de la fonction
            from tasks.models import Task, TaskStatus
            from workflows.models import WorkflowStatus
            
            # Récupérer toutes les tâches du workflow
            tasks = Task.objects.filter(workflow=workflow)
            
            # Vérifier s'il y a des tâches
            if not tasks.exists():
                logger.warning(f"Aucune tâche pour le workflow {workflow.id}")
                return
            
            # Compter les tâches terminées et échouées
            completed_count = tasks.filter(status=TaskStatus.COMPLETED).count()
            failed_count = tasks.filter(status=TaskStatus.FAILED).count()
            total_count = tasks.count()
            
            # Si toutes les tâches sont terminées
            if completed_count + failed_count == total_count:
                # Vérifier s'il y a eu des échecs
                if failed_count > 0:
                    # Si trop de tâches ont échoué, marquer le workflow comme échoué
                    if failed_count > total_count / 2:
                        workflow.status = WorkflowStatus.FAILED
                    else:
                        # Sinon, marquer comme partiellement échoué
                        workflow.status = WorkflowStatus.PARTIAL_FAILURE
                else:
                    # Si toutes les tâches sont terminées avec succès, passer à l'agrégation
                    workflow.status = WorkflowStatus.AGGREGATING
                
                workflow.save()
                
                # Si le workflow est en état d'agrégation, lancer l'agrégation
                if workflow.status == WorkflowStatus.AGGREGATING:
                    threading.Thread(
                        target=self.aggregate_matrix_results,
                        args=(workflow,)
                    ).start()
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de la complétion du workflow {workflow.id}: {e}")
    
    def _monitor_workflows(self):
        """
        Surveille les workflows en cours d'exécution
        """
        while self.running:
            try:
                # Import des modèles à l'intérieur de la fonction
                from workflows.models import Workflow, WorkflowStatus
                from tasks.models import Task, TaskStatus
            except ImportError as e:
                logger.error(f"Erreur d'importation des modèles: {e}")
                return
                
                # Récupérer les workflows en cours d'exécution
                running_workflows = Workflow.objects.filter(
                    status__in=[
                        WorkflowStatus.RUNNING,
                        WorkflowStatus.PENDING,
                        WorkflowStatus.ASSIGNING
                    ]
                )
                
                # Vérifier l'état de chaque workflow
                for workflow in running_workflows:
                    # Récupérer les tâches du workflow
                    tasks = Task.objects.filter(workflow=workflow)
                    
                    # Calculer la progression
                    if tasks.exists():
                        total_tasks = tasks.count()
                        completed_tasks = tasks.filter(status=TaskStatus.COMPLETED).count()
                        running_tasks = tasks.filter(status=TaskStatus.RUNNING).count()
                        
                        # Calculer la progression moyenne
                        progress_sum = sum(task.progress or 0 for task in tasks)
                        overall_progress = progress_sum / total_tasks
                        
                        # Mettre à jour la progression du workflow
                        workflow.progress = overall_progress
                        workflow.save(update_fields=['progress'])
                        
                        # Mettre à jour le coordinateur
                        self.coordinator_api.update_workflow_status(
                            workflow_id=str(workflow.id),
                            status=workflow.status,
                            progress=overall_progress
                        )
                        
                        # Publier la progression
                        self.mqtt_client.publish_workflow_status(workflow)
                        
                        # Vérifier si le workflow est terminé
                        if completed_tasks == total_tasks:
                            # Toutes les tâches sont terminées
                            self._check_workflow_completion(workflow)
                        elif running_tasks == 0 and workflow.status == WorkflowStatus.RUNNING:
                            # Aucune tâche n'est en cours d'exécution
                            # Vérifier s'il y a des tâches en attente
                            pending_tasks = tasks.filter(status=TaskStatus.PENDING).count()
                            
                            if pending_tasks == 0:
                                # Aucune tâche en attente, vérifier si le workflow est terminé
                                self._check_workflow_completion(workflow)
                
                # Attendre avant la prochaine vérification
                import time
                time.sleep(10)  # Vérification toutes les 10 secondes
            
            except Exception as e:
                logger.error(f"Erreur lors de la surveillance des workflows: {e}")
                import time
                time.sleep(30)  # Attendre plus longtemps en cas d'erreur

            def _handle_task_result(self, topic, message):
                """
                Gère les messages de résultat des tâches
                
                Args:
                    topic (str): Topic du message
                    message (dict): Contenu du message
                """
                try:
                    # Import des modèles à l'intérieur de la fonction
                    from tasks.models import Task, TaskStatus
                    
                    # Extraire les informations du message
                    task_id = message.get('task_id')
                    status = message.get('status')
                    results_location = message.get('results_location')
                    
                    if not task_id or not status:
                        logger.warning(f"Message de résultat incomplet: {message}")
                        return
                    
                    # Mettre à jour la tâche dans la base de données
                    with transaction.atomic():
                        try:
                            task = Task.objects.get(id=task_id)
                        except Task.DoesNotExist:
                            logger.warning(f"Tâche {task_id} non trouvée")
                            return
                        
                        # Mettre à jour le statut
                        if status == 'completed':
                            task.status = TaskStatus.COMPLETED
                            task.end_time = timezone.now()
                            task.progress = 100
                            
                            # Enregistrer l'emplacement des résultats
                            if results_location:
                                task.results = {
                                    'location': results_location,
                                    'timestamp': timezone.now().isoformat()
                                }
                            
                            task.save()
                            
                            # Lancer le téléchargement des résultats en arrière-plan
                            threading.Thread(
                                target=self._download_task_results,
                                args=(task, results_location)
                            ).start()
                            
                            # Appeler le callback enregistré pour cette tâche, s'il existe
                            callback = self.result_callbacks.get(str(task_id))
                            if callback:
                                callback(task, message)
                            
                            # Vérifier si toutes les tâches du workflow sont terminées
                            self._check_workflow_completion(task.workflow)
                    
                except Exception as e:
                    logger.error(f"Erreur lors du traitement du message de résultat: {e}")


            # Variable singleton pour le gestionnaire de messages
            _matrix_handler = None

            def get_matrix_handler():
                """
                Récupère l'instance du gestionnaire de messages matriciels
                S'assure que l'instance n'est créée qu'après le chargement complet de Django
                
                Returns:
                    MatrixMessageHandler: L'instance du gestionnaire
                """
                global _matrix_handler
                if _matrix_handler is None:
                    _matrix_handler = MatrixMessageHandler()
                return _matrix_handler

            def start_message_handler():
                """
                Démarre le gestionnaire de messages matriciels
                
                Returns:
                    bool: True si le démarrage a réussi, False sinon
                """
                handler = get_matrix_handler()
                return handler.start()

            def stop_message_handler():
                """
                Arrête le gestionnaire de messages matriciels
                
                Returns:
                    bool: True si l'arrêt a réussi, False sinon
                """
                handler = get_matrix_handler()
                return handler.stop()