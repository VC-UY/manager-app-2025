import json
import logging
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET
from django.conf import settings
from workflows.models import Workflow, WorkflowStatus
from tasks.models import Task, TaskStatus
from . import matrix_handler, start_message_handler, stop_message_handler

logger = logging.getLogger(__name__)

@csrf_exempt
@require_POST
def coordinator_notification(request):
    """
    Point d'entrée pour les notifications du coordinateur
    
    Args:
        request (HttpRequest): Requête HTTP
        
    Returns:
        JsonResponse: Réponse JSON
    """
    try:
        # Vérifier l'authentification
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        secret_key = getattr(settings, 'COORDINATOR_SECRET_KEY', None)
        
        if not token or token != secret_key:
            return JsonResponse({'error': 'Authentification invalide'}, status=401)
        
        # Analyser le corps de la requête
        data = json.loads(request.body)
        notification_type = data.get('type')
        
        # Traiter selon le type de notification
        if notification_type == 'volunteer_available':
            # Un nouveau volontaire est disponible
            volunteer_id = data.get('volunteer_id')
            return _handle_volunteer_available(volunteer_id, data)
        
        elif notification_type == 'volunteer_unavailable':
            # Un volontaire n'est plus disponible
            volunteer_id = data.get('volunteer_id')
            return _handle_volunteer_unavailable(volunteer_id, data)
        
        elif notification_type == 'task_status_update':
            # Mise à jour du statut d'une tâche
            task_id = data.get('task_id')
            status = data.get('status')
            return _handle_task_status_update(task_id, status, data)
        
        elif notification_type == 'workflow_status_update':
            # Mise à jour du statut d'un workflow
            workflow_id = data.get('workflow_id')
            status = data.get('status')
            return _handle_workflow_status_update(workflow_id, status, data)
            
        else:
            return JsonResponse({'error': f'Type de notification non supporté: {notification_type}'}, status=400)
    
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Corps de requête JSON invalide'}, status=400)
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement de la notification du coordinateur: {e}")
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_POST
def volunteer_notification(request):
    """
    Point d'entrée pour les notifications des volontaires
    
    Args:
        request (HttpRequest): Requête HTTP
        
    Returns:
        JsonResponse: Réponse JSON
    """
    try:
        # Vérifier l'authentification
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        
        # Pour les volontaires, vérifier le token auprès du coordinateur
        # En production, implémenter une vérification de token plus robuste
        if not _validate_volunteer_token(token):
            return JsonResponse({'error': 'Authentification invalide'}, status=401)
        
        # Analyser le corps de la requête
        data = json.loads(request.body)
        notification_type = data.get('type')
        
        # Traiter selon le type de notification
        if notification_type == 'task_status':
            # Mise à jour du statut d'une tâche
            task_id = data.get('task_id')
            status = data.get('status')
            return _handle_task_status_update(task_id, status, data)
        
        elif notification_type == 'task_result':
            # Notification de résultat de tâche
            task_id = data.get('task_id')
            result_location = data.get('result_location')
            return _handle_task_result(task_id, result_location, data)
            
        else:
            return JsonResponse({'error': f'Type de notification non supporté: {notification_type}'}, status=400)
    
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Corps de requête JSON invalide'}, status=400)
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement de la notification du volontaire: {e}")
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_POST
def task_status_update(request, task_id):
    """
    Point d'entrée pour les mises à jour de statut de tâche
    
    Args:
        request (HttpRequest): Requête HTTP
        task_id (str): ID de la tâche
        
    Returns:
        JsonResponse: Réponse JSON
    """
    try:
        # Vérifier l'authentification
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        
        # Vérifier si c'est le coordinateur ou un volontaire
        if token == getattr(settings, 'COORDINATOR_SECRET_KEY', None):
            # C'est le coordinateur, autoriser
            pass
        elif _validate_volunteer_token(token):
            # C'est un volontaire valide, autoriser
            pass
        else:
            return JsonResponse({'error': 'Authentification invalide'}, status=401)
        
        # Analyser le corps de la requête
        data = json.loads(request.body)
        status = data.get('status')
        
        return _handle_task_status_update(task_id, status, data)
    
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Corps de requête JSON invalide'}, status=400)
    
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour du statut de la tâche {task_id}: {e}")
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_POST
def workflow_status_update(request, workflow_id):
    """
    Point d'entrée pour les mises à jour de statut de workflow
    
    Args:
        request (HttpRequest): Requête HTTP
        workflow_id (str): ID du workflow
        
    Returns:
        JsonResponse: Réponse JSON
    """
    try:
        # Vérifier l'authentification
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        
        # Seul le coordinateur peut mettre à jour le statut d'un workflow
        if token != getattr(settings, 'COORDINATOR_SECRET_KEY', None):
            return JsonResponse({'error': 'Authentification invalide'}, status=401)
        
        # Analyser le corps de la requête
        data = json.loads(request.body)
        status = data.get('status')
        
        return _handle_workflow_status_update(workflow_id, status, data)
    
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Corps de requête JSON invalide'}, status=400)
    
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour du statut du workflow {workflow_id}: {e}")
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_POST
def task_result_notification(request, task_id):
    """
    Point d'entrée pour les notifications de résultat de tâche
    
    Args:
        request (HttpRequest): Requête HTTP
        task_id (str): ID de la tâche
        
    Returns:
        JsonResponse: Réponse JSON
    """
    try:
        # Vérifier l'authentification
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        
        # Vérifier si c'est un volontaire valide
        if not _validate_volunteer_token(token):
            return JsonResponse({'error': 'Authentification invalide'}, status=401)
        
        # Analyser le corps de la requête
        data = json.loads(request.body)
        result_location = data.get('result_location')
        
        # Vérifier que la tâche existe
        try:
            task = Task.objects.get(id=task_id)
        except Task.DoesNotExist:
            return JsonResponse({'error': f'Tâche {task_id} non trouvée'}, status=404)
        
        # Vérifier que la tâche est assignée à ce volontaire
        volunteer_id = data.get('volunteer_id')
        if task.assigned_to != volunteer_id:
            return JsonResponse({'error': 'La tâche n\'est pas assignée à ce volontaire'}, status=403)
        
        # Traiter la notification
        return _handle_task_result(task_id, result_location, data)
    
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Corps de requête JSON invalide'}, status=400)
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement de la notification de résultat pour la tâche {task_id}: {e}")
        return JsonResponse({'error': str(e)}, status=500)

@require_GET
def start_matrix_handler(request):
    """
    Point d'entrée pour démarrer le gestionnaire de messages matriciels
    
    Args:
        request (HttpRequest): Requête HTTP
        
    Returns:
        JsonResponse: Réponse JSON
    """
    try:
        # Vérifier l'authentification (en production, utiliser une authentification plus robuste)
        if not request.user.is_authenticated and request.user.is_staff:
            return JsonResponse({'error': 'Authentification requise'}, status=401)
        
        # Démarrer le gestionnaire
        success = start_message_handler()
        
        if success:
            return JsonResponse({'status': 'success', 'message': 'Gestionnaire de messages démarré'})
        else:
            return JsonResponse({'error': 'Échec du démarrage du gestionnaire de messages'}, status=500)
    
    except Exception as e:
        logger.error(f"Erreur lors du démarrage du gestionnaire de messages: {e}")
        return JsonResponse({'error': str(e)}, status=500)

@require_GET
def stop_matrix_handler(request):
    """
    Point d'entrée pour arrêter le gestionnaire de messages matriciels
    
    Args:
        request (HttpRequest): Requête HTTP
        
    Returns:
        JsonResponse: Réponse JSON
    """
    try:
        # Vérifier l'authentification (en production, utiliser une authentification plus robuste)
        if not request.user.is_authenticated and request.user.is_staff:
            return JsonResponse({'error': 'Authentification requise'}, status=401)
        
        # Arrêter le gestionnaire
        success = stop_message_handler()
        
        if success:
            return JsonResponse({'status': 'success', 'message': 'Gestionnaire de messages arrêté'})
        else:
            return JsonResponse({'error': 'Échec de l\'arrêt du gestionnaire de messages'}, status=500)
    
    except Exception as e:
        logger.error(f"Erreur lors de l'arrêt du gestionnaire de messages: {e}")
        return JsonResponse({'error': str(e)}, status=500)

@require_GET
def matrix_handler_status(request):
    """
    Point d'entrée pour vérifier le statut du gestionnaire de messages matriciels
    
    Args:
        request (HttpRequest): Requête HTTP
        
    Returns:
        JsonResponse: Réponse JSON
    """
    try:
        # Vérifier l'authentification (en production, utiliser une authentification plus robuste)
        if not request.user.is_authenticated:
            return JsonResponse({'error': 'Authentification requise'}, status=401)
        
        # Vérifier le statut
        is_running = matrix_handler.running
        mqtt_connected = matrix_handler.mqtt_client.connected if hasattr(matrix_handler, 'mqtt_client') else False
        
        return JsonResponse({
            'status': 'running' if is_running else 'stopped',
            'mqtt_connected': mqtt_connected,
            'subscribed_topics': list(matrix_handler.mqtt_client.subscribed_topics) if hasattr(matrix_handler.mqtt_client, 'subscribed_topics') else [],
            'workflows_monitored': Workflow.objects.filter(status__in=[
                WorkflowStatus.RUNNING,
                WorkflowStatus.PENDING,
                WorkflowStatus.ASSIGNING
            ]).count()
        })
    
    except Exception as e:
        logger.error(f"Erreur lors de la vérification du statut du gestionnaire de messages: {e}")
        return JsonResponse({'error': str(e)}, status=500)

# Fonctions utilitaires pour le traitement des notifications

def _handle_volunteer_available(volunteer_id, data):
    """
    Traite la notification de disponibilité d'un volontaire
    
    Args:
        volunteer_id (str): ID du volontaire
        data (dict): Données de la notification
        
    Returns:
        JsonResponse: Réponse JSON
    """
    try:
        # Récupérer les tâches en attente qui pourraient être assignées à ce volontaire
        pending_tasks = Task.objects.filter(status=TaskStatus.PENDING, assigned_to=None)
        
        # Vérifier les capacités du volontaire
        resources = data.get('resources', {})
        
        # Stocker temporairement les capacités du volontaire pour les utiliser dans l'algorithme d'attribution
        # En production, ces données devraient être persistées dans une base de données
        from django.core.cache import cache
        cache_key = f"volunteer:{volunteer_id}:resources"
        cache.set(cache_key, resources, timeout=3600)  # Expire après 1 heure
        
        return JsonResponse({
            'status': 'success',
            'volunteer_id': volunteer_id,
            'pending_tasks': pending_tasks.count()
        })
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement de la notification de disponibilité du volontaire {volunteer_id}: {e}")
        return JsonResponse({'error': str(e)}, status=500)

def _handle_volunteer_unavailable(volunteer_id, data):
    """
    Traite la notification d'indisponibilité d'un volontaire
    
    Args:
        volunteer_id (str): ID du volontaire
        data (dict): Données de la notification
        
    Returns:
        JsonResponse: Réponse JSON
    """
    try:
        # Récupérer les tâches assignées à ce volontaire
        assigned_tasks = Task.objects.filter(assigned_to=volunteer_id)
        running_tasks = assigned_tasks.filter(status=TaskStatus.RUNNING)
        
        # Marquer les tâches en cours comme en attente
        for task in running_tasks:
            task.status = TaskStatus.PENDING
            task.assigned_to = None
            task.save(update_fields=['status', 'assigned_to'])
        
        # Supprimer les informations de capacité du volontaire
        from django.core.cache import cache
        cache_key = f"volunteer:{volunteer_id}:resources"
        cache.delete(cache_key)
        
        return JsonResponse({
            'status': 'success',
            'volunteer_id': volunteer_id,
            'reassigned_tasks': running_tasks.count()
        })
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement de la notification d'indisponibilité du volontaire {volunteer_id}: {e}")
        return JsonResponse({'error': str(e)}, status=500)

def _handle_task_status_update(task_id, status, data):
    """
    Traite une mise à jour de statut de tâche
    
    Args:
        task_id (str): ID de la tâche
        status (str): Nouveau statut
        data (dict): Données supplémentaires
        
    Returns:
        JsonResponse: Réponse JSON
    """
    try:
        # Vérifier que la tâche existe
        try:
            task = Task.objects.get(id=task_id)
        except Task.DoesNotExist:
            return JsonResponse({'error': f'Tâche {task_id} non trouvée'}, status=404)
        
        # Mettre à jour le statut
        if status == 'pending':
            task.status = TaskStatus.PENDING
        elif status == 'running':
            task.status = TaskStatus.RUNNING
            if task.start_time is None:
                task.start_time = data.get('timestamp', timezone.now())
        elif status == 'completed':
            task.status = TaskStatus.COMPLETED
            task.end_time = data.get('timestamp', timezone.now())
            task.progress = 100
        elif status == 'failed':
            task.status = TaskStatus.FAILED
            task.end_time = data.get('timestamp', timezone.now())
            
            # Enregistrer les détails de l'erreur
            if 'error' in data:
                task.error_details = data.get('error')
        else:
            return JsonResponse({'error': f'Statut non supporté: {status}'}, status=400)
        
        # Mettre à jour la progression si fournie
        if 'progress' in data:
            task.progress = float(data.get('progress', 0))
        
        # Sauvegarder les changements
        task.save()
        
        # Publier la mise à jour via MQTT
        message = {
            'task_id': str(task.id),
            'status': status,
            'progress': task.progress,
            'timestamp': timezone.now().isoformat()
        }
        matrix_handler.mqtt_client.publish(f"tasks/status/{task_id}", message)
        
        # Vérifier si toutes les tâches du workflow sont terminées
        _check_workflow_completion(task.workflow)
        
        return JsonResponse({
            'status': 'success',
            'task_id': str(task.id),
            'current_status': status
        })
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement de la mise à jour de statut de la tâche {task_id}: {e}")
        return JsonResponse({'error': str(e)}, status=500)

def _handle_workflow_status_update(workflow_id, status, data):
    """
    Traite une mise à jour de statut de workflow
    
    Args:
        workflow_id (str): ID du workflow
        status (str): Nouveau statut
        data (dict): Données supplémentaires
        
    Returns:
        JsonResponse: Réponse JSON
    """
    try:
        # Vérifier que le workflow existe
        try:
            workflow = Workflow.objects.get(id=workflow_id)
        except Workflow.DoesNotExist:
            return JsonResponse({'error': f'Workflow {workflow_id} non trouvé'}, status=404)
        
        # Mettre à jour le statut
        if status == 'submitted':
            workflow.status = WorkflowStatus.SUBMITTED
        elif status == 'splitting':
            workflow.status = WorkflowStatus.SPLITTING
        elif status == 'assigning':
            workflow.status = WorkflowStatus.ASSIGNING
        elif status == 'pending':
            workflow.status = WorkflowStatus.PENDING
        elif status == 'running':
            workflow.status = WorkflowStatus.RUNNING
        elif status == 'aggregating':
            workflow.status = WorkflowStatus.AGGREGATING
        elif status == 'completed':
            workflow.status = WorkflowStatus.COMPLETED
            workflow.completed_at = data.get('timestamp', timezone.now())
        elif status == 'failed':
            workflow.status = WorkflowStatus.FAILED
            
            # Enregistrer les détails de l'erreur
            if 'error' in data:
                if not workflow.metadata:
                    workflow.metadata = {}
                workflow.metadata['error'] = data.get('error')
        elif status == 'paused':
            workflow.status = WorkflowStatus.PAUSED
        else:
            return JsonResponse({'error': f'Statut non supporté: {status}'}, status=400)
        
        # Mettre à jour la progression si fournie
        if 'progress' in data:
            workflow.progress = float(data.get('progress', 0))
        
        # Sauvegarder les changements
        workflow.save()
        
        # Publier la mise à jour via MQTT
        message = {
            'workflow_id': str(workflow.id),
            'status': status,
            'progress': workflow.progress,
            'timestamp': timezone.now().isoformat()
        }
        matrix_handler.mqtt_client.publish(f"workflows/status/{workflow_id}", message)
        
        return JsonResponse({
            'status': 'success',
            'workflow_id': str(workflow.id),
            'current_status': status
        })
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement de la mise à jour de statut du workflow {workflow_id}: {e}")
        return JsonResponse({'error': str(e)}, status=500)

def _handle_task_result(task_id, result_location, data):
    """
    Traite une notification de résultat de tâche
    
    Args:
        task_id (str): ID de la tâche
        result_location (dict): Emplacement des résultats
        data (dict): Données supplémentaires
        
    Returns:
        JsonResponse: Réponse JSON
    """
    try:
        # Vérifier que la tâche existe
        try:
            task = Task.objects.get(id=task_id)
        except Task.DoesNotExist:
            return JsonResponse({'error': f'Tâche {task_id} non trouvée'}, status=404)
        
        # Mettre à jour la tâche
        task.status = TaskStatus.COMPLETED
        task.end_time = data.get('timestamp', timezone.now())
        task.progress = 100
        
        # Enregistrer les informations de résultat
        task.results = {
            'location': result_location,
            'timestamp': timezone.now().isoformat()
        }
        
        # Sauvegarder les changements
        task.save()
        
        # Publier la mise à jour via MQTT
        message = {
            'task_id': str(task.id),
            'status': 'completed',
            'result_location': result_location,
            'timestamp': timezone.now().isoformat()
        }
        matrix_handler.mqtt_client.publish(f"tasks/results/{task_id}", message)
        
        # Lancer le téléchargement des résultats en arrière-plan
        import threading
        threading.Thread(
            target=matrix_handler._download_task_results,
            args=(task, result_location)
        ).start()
        
        # Vérifier si toutes les tâches du workflow sont terminées
        _check_workflow_completion(task.workflow)
        
        return JsonResponse({
            'status': 'success',
            'task_id': str(task.id),
            'message': 'Résultats reçus et en cours de traitement'
        })
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement du résultat de la tâche {task_id}: {e}")
        return JsonResponse({'error': str(e)}, status=500)

def _check_workflow_completion(workflow):
    """
    Vérifie si toutes les tâches d'un workflow sont terminées et déclenche l'agrégation si nécessaire
    
    Args:
        workflow (Workflow): Le workflow à vérifier
    """
    try:
        # Récupérer toutes les tâches du workflow
        tasks = Task.objects.filter(workflow=workflow)
        
        # Vérifier s'il y a des tâches
        if not tasks.exists():
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
                import threading
                threading.Thread(
                    target=matrix_handler.aggregate_matrix_results,
                    args=(workflow,)
                ).start()
    
    except Exception as e:
        logger.error(f"Erreur lors de la vérification de la complétion du workflow {workflow.id}: {e}")

def _validate_volunteer_token(token):
    """
    Valide un token de volontaire
    
    Args:
        token (str): Token à valider
        
    Returns:
        bool: True si le token est valide, False sinon
    """
    # Pour cet exemple, nous validons simplement que le token n'est pas vide
    if not token:
        return False
    
    try:
        # Valider le token auprès du coordinateur
        coordinator_api = CoordinatorAPI()
        result = coordinator_api._make_request("POST", "/auth/validate_token", {
            "token": token,
            "client_type": "volunteer"
        })
        
        return result and result.get('valid', False)
    
    except Exception as e:
        logger.error(f"Erreur lors de la validation du token: {e}")
        return False


# Importer les classes nécessaires pour les vues
from .coordinator_api import CoordinatorAPI