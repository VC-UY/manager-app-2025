"""
Gestionnaires pour les réponses aux demandes de réassignation de tâches.
"""

import logging
import traceback
from django.utils import timezone

from redis_communication.message import Message
from tasks.models import Task, TaskStatus
from volunteers.models import Volunteer, VolunteerTask

logger = logging.getLogger(__name__)

def handle_task_reassignment_response(channel: str, message: Message):
    """
    Gestionnaire pour les réponses aux demandes de réassignation de tâches.
    
    Args:
        channel: Canal sur lequel le message a été reçu
        message: Message reçu
    """
    try:
        logger.info(f"Réponse à une demande de réassignation de tâche reçue: {message.to_dict()}")
        
        # Extraire les données du message
        data = message.data
        
        # Vérifier si les données nécessaires sont présentes
        if not data or 'task_id' not in data or 'success' not in data:
            logger.error(f"Données manquantes dans le message: {data}")
            return
        
        task_id = data.get('task_id')
        success = data.get('success')
        volunteer_id = data.get('volunteer_id')
        error = data.get('error')
        
        # Récupérer la tâche
        try:
            task = Task.objects.get(id=task_id)
        except Task.DoesNotExist:
            logger.error(f"Tâche non trouvée: {task_id}")
            return
        
        # Récupérer le workflow
        try:
            workflow = task.workflow
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du workflow pour la tâche {task_id}: {e}")
            return
        
        if success and volunteer_id:
            logger.info(f"Réassignation réussie pour la tâche {task_id} au volontaire {volunteer_id}")
            
            # Mettre à jour le statut de la tâche
            task.status = TaskStatus.ASSIGNED
            task.save()
            
            # Récupérer le volontaire
            try:
                volunteer = Volunteer.objects.get(coordinator_volunteer_id=volunteer_id)
            except Volunteer.DoesNotExist:
                # Le volontaire n'existe pas encore dans notre base de données
                # Créer un nouveau volontaire avec les informations minimales
                volunteer = Volunteer.objects.create(
                    coordinator_volunteer_id=volunteer_id,
                    name=f"Volunteer-{volunteer_id[:8]}",  # Nom temporaire
                    status="ACTIVE"
                )
                logger.info(f"Nouveau volontaire créé: {volunteer.name} ({volunteer_id})")
            
            # Créer ou mettre à jour l'assignation de tâche
            volunteer_task, created = VolunteerTask.objects.update_or_create(
                task=task,
                defaults={
                    'volunteer': volunteer,
                    'assigned_at': timezone.now(),
                    'status': "ASSIGNED"
                }
            )

            # --- Notification WebSocket de l'assignation ---
            from websocket_service.client import notify_event
            notify_event('task_assignment', {
                'task_id': str(task.id),
                'volunteer_id': str(volunteer.id),
                'status': task.status,
                'message': f"Tâche {task.name} assignée à {volunteer.name}"
            })

            assignments_by_volunteer = {}
            # Préparer les informations de fichiers d'entrée
            input_files = []
            if task.input_files:
                for file_path in task.input_files:
                    input_files.append({
                        'path': file_path,
                        'size': 0  
                    })
            
            # Recuperer les information sur le serveur de fichiers
            file_server_info = get_file_server_url(workflow_id)
            if not file_server_info:
                logger.error(f"Serveur de fichiers non trouvé pour le workflow {workflow_id}")
                return
            
            file_server_info = {
                'server_url': file_server_info,
                'workflow_id': workflow_id
            }
            
            # Créer les données de la tâche pour ce volontaire
            task_data = {
                'task_id': str(task.id),
                'name': task.name,
                'parameters': task.parameters,
                'input_data': {
                    'files': input_files,
                    'file_server': file_server_info
                },
                'estimated_execution_time': task.estimated_max_time,
                'docker_information': task.docker_info or {}
            }
            
            assignments_by_volunteer[volunteer_id].append(task_data)
        
        # Envoyer les assignations aux volontaires
        from redis_communication.client import RedisClient
        import uuid
        from redis_communication.utils import get_manager_login_token
        
        redis_client = RedisClient.get_instance()
        for volunteer_id, tasks_data in assignments_by_volunteer.items():
            thread_logger.info(f"Envoi de {len(tasks_data)} tâches au volontaire {volunteer_id}")
            
            # Préparer le message d'assignation
            assignment_message = {
                'workflow_id': str(workflow_id),
                'assignments': {
                    volunteer_id: tasks_data
                }
            }
            
            # Publier le message d'assignation
            redis_client.publish('task/assignment', 
                            assignment_message,
                            str(uuid.uuid4()),
                            get_manager_login_token(),
                            'request'
                        )
            
            if created:
                logger.info(f"Nouvelle assignation créée: tâche {task.name} assignée au volontaire {volunteer.name}")
            else:
                logger.info(f"Assignation mise à jour: tâche {task.name} réassignée au volontaire {volunteer.name}")
            
            # Notifier le changement de statut via WebSocket
            try:
                from websocket_service.client import notify_event
                notify_event('task_status_change', {
                    'workflow_id': str(workflow.id),
                    'task_id': str(task.id),
                    'status': 'ASSIGNED',
                    'volunteer': volunteer.name,
                    'message': f"Tâche {task.name} réassignée à {volunteer.name}"
                })
            except Exception as e:
                logger.error(f"Erreur lors de la notification WebSocket: {e}")
        
        else:
            logger.warning(f"Échec de la réassignation pour la tâche {task_id}: {error}")
            
            # Mettre à jour le statut de la tâche
            task.status = TaskStatus.FAILED
            task.save()
            
            # Notifier l'échec via WebSocket
            try:
                from websocket_service.client import notify_event
                notify_event('task_status_change', {
                    'workflow_id': str(workflow.id),
                    'task_id': str(task.id),
                    'status': 'FAILED',
                    'message': f"Échec de la réassignation de la tâche {task.name}: {error}"
                })
            except Exception as e:
                logger.error(f"Erreur lors de la notification WebSocket: {e}")
    
    except Exception as e:
        logger.error(f"Erreur dans le gestionnaire de réponse d'assignation de tâche: {e}")
        logger.error(traceback.format_exc())

def register_handlers():
    """
    Enregistre les gestionnaires d'événements pour les réponses aux demandes de réassignation de tâches.
    """
    from redis_communication.client import RedisClient
    client = RedisClient.get_instance()
    
    # Enregistrer le gestionnaire pour les réponses aux demandes de réassignation
    client.subscribe('task/assignment/response', handle_task_reassignment_response)
    
    logger.info("Gestionnaires de réponses aux demandes de réassignation de tâches enregistrés")
