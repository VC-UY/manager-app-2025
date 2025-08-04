"""
Gestionnaires d'événements pour les messages Redis.
Inclut les gestionnaires pour l'authentification des managers et des volontaires.
"""

import logging
import json
from math import log
from django.conf import settings
from django.utils import timezone
from redis_communication.message import Message
logger = logging.getLogger(__name__)




def handle_task_accept(channel: str, message: Message):
    """
    Gestionnaire pour l'écoute des messages de type 'task/accept'.
    
    Args:
        channel: Canal sur lequel le message a été reçu
        message: Message reçu
        
    Returns:
        True si le message a été traité avec succès, False sinon
    """
    logger.info(f"Reçu un message de type 'task/accept' sur le canal {channel}")
    logger.debug(f"Contenu du message: {message.data}")
    
    # Récupérer les informations
    data = message.data
    
    # Vérifier que le message contient les informations nécessaires
    if 'workflow_id' not in data or 'task_id' not in data or 'volunteer_id' not in data:
        logger.error("Le message ne contient pas les informations nécessaires (workflow_id, task_id, volunteer_id)")
        return False
    
    workflow_id = data['workflow_id']
    task_id = data['task_id']
    volunteer_id = data['volunteer_id']
    
    logger.info(f"Traitement de l'acceptation de la tâche {task_id} par le volontaire {volunteer_id} pour le workflow {workflow_id}")
    
    try:
        # Importer les modèles nécessaires
        from workflows.models import Workflow, WorkflowStatus
        from tasks.models import Task, TaskStatus
        from volunteers.models import Volunteer, VolunteerTask
        
        # Récupérer les objets
        workflow = Workflow.objects.get(id=workflow_id)
        task = Task.objects.get(id=task_id)
        volunteer = Volunteer.objects.get(coordinator_volunteer_id=volunteer_id)

        # Verifier si le workflow est en cours
        if workflow.status != WorkflowStatus.RUNNING:
            workflow.status = WorkflowStatus.RUNNING
            workflow.save()
            
        # Verifier si la tâche est en cours
        if task.status != TaskStatus.RUNNING:
            task.status = TaskStatus.RUNNING
            task.start_time = timezone.now()
            task.save()
            
        # Vérifier si la tâche est déjà assignée à ce volontaire
        volunteer_task = VolunteerTask.objects.filter(task=task, volunteer=volunteer).first()
        
        if volunteer_task:
            # Mettre à jour le statut de l'assignation
            volunteer_task.status = "ACCEPTED"
            volunteer_task.accepted_at = timezone.now()
            volunteer_task.save()
            logger.info(f"Assignation existante mise à jour: tâche {task.name} acceptée par le volontaire {volunteer.name}")
        else:
            # Créer une nouvelle assignation
            volunteer_task = VolunteerTask.objects.create(
                task=task,
                volunteer=volunteer,
                assigned_at=timezone.now(),
                started_at=timezone.now(),
                status="STARTED"
            )
            logger.info(f"Nouvelle assignation créée: tâche {task.name} acceptée par le volontaire {volunteer.name}")
        
        
        # Notifier le changement de statstarted_atut via WebSocket
        from websocket_service.client import notify_event
        notify_event('task_status_change', {
            'workflow_id': str(workflow.id),
            'task_id': str(task.id),
            'status': 'ACCEPTED',
            'volunteer': volunteer.name,
            'message': f"Tâche {task.name} démarrée par {volunteer.name}"
        })
        
        return True
        
    except Workflow.DoesNotExist:
        logger.error(f"Le workflow {workflow_id} n'existe pas")
    except Task.DoesNotExist:
        logger.error(f"La tâche {task_id} n'existe pas")
    except Volunteer.DoesNotExist:
        logger.error(f"Le volontaire {volunteer_id} n'existe pas")
    except Exception as e:
        logger.error(f"Erreur lors du traitement de l'acceptation de la tâche: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    return False




def handle_task_progress(channel: str, message: Message):
    """
    Gere la reception de la progression des taches via le cannal 'task/progress'

    Args:
        channel (str): Nom du cannal 'task/progress'
        message (Message): Message recu
    """

    logger.info("Progression de tache recu")
    logger.warning(f"Contenu du message: {message.data}")

    try:
        # Récupérer les informations
        data = message.data
        
        # Vérifier que le message contient les informations nécessaires
        if 'workflow_id' not in data or 'task_id' not in data or 'volunteer_id' not in data:
            logger.error("Le message ne contient pas les informations nécessaires (workflow_id, task_id, volunteer_id)")
            return False
        
        workflow_id = data['workflow_id']
        task_id = data['task_id']
        volunteer_id = data['volunteer_id']
        progress = data['progress']


        # Récupérer les objets

        from workflows.models import Workflow
        from tasks.models import Task
        from volunteers.models import Volunteer, VolunteerTask
        workflow = Workflow.objects.get(id=workflow_id)
        task = Task.objects.get(id=task_id)
        volunteer = Volunteer.objects.get(coordinator_volunteer_id=volunteer_id)


                
            # Vérifier si la tâche est déjà assignée à ce volontaire
        volunteer_task = VolunteerTask.objects.filter(task=task, volunteer=volunteer).first()


        if volunteer_task:
            # Mettre à jour le statut de l'assignation
            volunteer_task.progress = progress
            volunteer_task.save()
            logger.info(f"Assignation mise à jour: tâche {task.name} en cours par le volontaire {volunteer.name}, progression: {progress}%")
            
            # Notifier la progression via WebSocket
            from websocket_service.client import notify_event
            notify_event('task_progress', {
                'workflow_id': str(workflow.id),
                'task_id': str(task.id),
                'progress': progress,
                'volunteer': volunteer.name,
                'message': f"Progression de la tâche {task.name}: {progress}%"
            })
            
            return True

        else:
            # Generer un message d'erreur
            logger.error(f"Pas d'assignation de tache entre le volontaire {volunteer.name} et la tache {task.name}")
            return False
    except Workflow.DoesNotExist:
        logger.error(f"Le workflow {workflow_id} n'existe pas")
    except Task.DoesNotExist:
        logger.error(f"La tâche {task_id} n'existe pas")
    except Volunteer.DoesNotExist:
        logger.error(f"Le volontaire {volunteer_id} n'existe pas")
    except Exception as e:
        logger.error(f"Erreur lors du traitement de l'acceptation de la tâche: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    return False



def handle_task_status(channel: str, message: Message):
    """
    Mets à jour le statut de la tache

    Args:
        channel (str): canal
        message (Message): Message recu
    """
    logger.info("Statut de tache recu")
    logger.debug(f"Message de statut recu: {message.data}")

    try:
        # Récupérer les informations
        data = message.data
        
        # Vérifier que le message contient les informations nécessaires
        if 'workflow_id' not in data or 'task_id' not in data or 'volunteer_id' not in data:
            logger.error(f"Le message ne contient pas les informations nécessaires (workflow_id, task_id, volunteer_id): {message.data} {type(message.data)}")
            return False
        
        workflow_id = data['workflow_id']
        task_id = data['task_id']
        volunteer_id = data['volunteer_id']
        status = data['status']

        # Récupérer les objets
        from workflows.models import Workflow
        from tasks.models import Task
        from volunteers.models import Volunteer, VolunteerTask
        workflow = Workflow.objects.get(id=workflow_id)
        task = Task.objects.get(id=task_id)
        volunteer = Volunteer.objects.get(coordinator_volunteer_id=volunteer_id)

        # Vérifier si la tâche est déjà assignée à ce volontaire
        volunteer_task = VolunteerTask.objects.filter(task=task, volunteer=volunteer).first()

        if not volunteer_task:
            # Generer un message d'erreur
            logger.error(f"Pas d'assignation de tache entre le volontaire {volunteer.name} et la tache {task.name}")
            return False

        # Mettre à jour le statut de l'assignation
        volunteer_task.status = status
        volunteer_task.save()
        logger.info(f"Statut de la tâche mise à jour: tâche {task.name} statut {status} par le volontaire {volunteer.name}")
        
        # Traiter les différents statuts
        if status.lower() == 'completed':
            # La tâche est terminée, télécharger les fichiers de sortie
            volunteer_task.completed_at = timezone.now()
            volunteer_task.progress = 100.0
            logger.info(f"Tâche {task.name} terminée par le volontaire {volunteer.name}, téléchargement des fichiers de sortie")

            # Notifier le changement de statut via WebSocket
            from websocket_service.client import notify_event
            notify_event('task_status_change', {
                'workflow_id': str(workflow.id),
                'task_id': str(task.id),
                'status': 'COMPLETED',
                'volunteer': volunteer.name,
                'message': f"Tâche {task.name} terminée par {volunteer.name}"
            })
            
            # Vérifier si les informations du serveur de fichiers sont disponibles
            if 'file_server' in data:
                task.results = data['file_server']
                volunteer_task.result = data['file_server']
                file_server = data['file_server']
                host = file_server.get('host', '127.0.0.1')
                port = file_server.get('port')
                path = file_server.get('path', '/files/')
                output_files = file_server.get('output_files', [])
                
                if port and output_files:
                    # Créer le répertoire de sortie pour cette tâche
                    import os
                    output_dir = os.path.join(workflow.output_path, str(task.id))
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Télécharger les fichiers
                    import requests
                    import shutil
                    
                    success = True
                    downloaded_files = []
                    logger.info(f"Téléchargement des fichiers de sortie: {output_files}")
                    for file in output_files:
                        file_url = f"http://{host}:{port}{path}{file}"
                        output_path = os.path.join(output_dir, file)
                        
                        try:
                            response = requests.get(file_url, stream=True)
                            if response.status_code == 200:
                                with open(output_path, 'wb') as f:
                                    shutil.copyfileobj(response.raw, f)
                                downloaded_files.append(output_path)
                                logger.info(f"Fichier téléchargé: {file}")
                            else:
                                logger.error(f"Erreur lors du téléchargement du fichier {file}: {response.status_code}")
                                success = False
                        except Exception as e:
                            logger.error(f"Erreur lors du téléchargement du fichier {file}: {e}")
                            success = False
                    
                    if success:
                        # Mettre à jour la tâche avec les fichiers de sortie
                        task.output_files = downloaded_files
                        task.status = 'completed'
                        task.end_date = timezone.now()
                        task.save()
                        logger.info(f"Tâche {task.name} terminée avec succès, fichiers téléchargés")
                        
                        # Envoyer une notification de fin de tâche au volontaire
                        from redis_communication.client import RedisClient
                        from redis_communication.utils import get_manager_login_token
                        import uuid
                        redis_client = RedisClient.get_instance()
                        
                        redis_client.publish('task/terminate', {
                                'task_id': str(task.id),
                                'volunteer_id': str(volunteer.coordinator_volunteer_id),
                                'workflow_id': str(workflow.id),
                                'status': 'terminated',
                                'clean_files': True,  # Supprimer les fichiers côté volontaire
                                'timestamp': timezone.now().isoformat()
                            },
                            str(uuid.uuid4()),
                            get_manager_login_token(),
                            'request'
                        )
                        
                        logger.info(f"Tâche {task.name} terminée avec succès, fichiers téléchargés")
                        
                        # Vérifier si toutes les tâches du workflow sont terminées
                        all_tasks_completed = Task.objects.filter(workflow=workflow).exclude(status='completed').count() 
                        terminated_tasks = all_tasks_completed == 0
                        if terminated_tasks:

                            # Mettre à jour le workflow
                            workflow.status = 'completed'
                            workflow.end_date = timezone.now()
                            workflow.save()
                            logger.info(f"Workflow {workflow.name} terminé avec succès")

                            # Envoyer une notification de fin de workflow au volontaire
                            redis_client.publish('workflow/terminate', {
                                'workflow_id': str(workflow.id),
                                'volunteer_id': str(volunteer.coordinator_volunteer_id),
                                'status': 'terminated',
                                'clean_files': True,  # Supprimer les fichiers côté volontaire
                                'timestamp': timezone.now().isoformat()
                            },
                            str(uuid.uuid4()),
                            get_manager_login_token(),
                            'request'
                            )
                            
                            logger.info(f"Workflow {workflow.name} terminé avec succès, fichiers téléchargés")

                            # Notifier le changement de statut via WebSocket
                            
                            from websocket_service.client import notify_event
                            notify_event('workflow_status_change', {
                                'workflow_id': str(workflow.id),
                                'status': 'terminated',
                                'message': f"Workflow {workflow.name} terminé avec succès"
                            })
                            
                            # Lancer l'agregation des résultats
                            from workflows.models import WorkflowType
                            if workflow.workflow_type == WorkflowType.ML_TRAINING:
                                from workflows.examples.distributed_training_demo.merge_models import   merge_models
                                input_path = workflow.output_path
                                output_path = os.path.join(workflow.output_path, 'merged_model.pt')
                                merge_models(input_path, output_path)
                                logger.info(f"Modèles fusionnés avec succès")

                                # Supprimer les sous dossier de l'output sauf merged_model.pt
                                import os
                                for item in os.listdir(workflow.output_path):
                                    item_path = os.path.join(workflow.output_path, item)
                                    if os.path.isdir(item_path):
                                        # Supprimer le sous-dossier et tout son contenu
                                        shutil.rmtree(item_path)
                                logger.info(f"Fichiers de sortie supprimés")
                        
                        else:
                            logger.info(f"Taches terminées: {all_tasks_completed}/{workflow.tasks.count()}")

                                

                    else:
                        logger.error(f"Erreur lors du téléchargement des fichiers de sortie pour la tâche {task.name}")
                else:
                    logger.error(f"Informations de serveur de fichiers incomplètes: port={port}, files={output_files}")
            else:
                logger.error(f"Aucune information de serveur de fichiers dans le message de statut pour la tâche {task.name}")
        
        elif status.lower() == 'paused':

            # Notifier le changement de statut via WebSocket
            from websocket_service.client import notify_event
            notify_event('task_status_change', {
                'workflow_id': str(workflow.id),
                'task_id': str(task.id),
                'status': 'paused',
                'volunteer': volunteer.name,
                'message': f"Tâche {task.name} mise en pause par {volunteer.name}"
            })
            
            # La tâche est en pause
            task.status = 'paused'
            task.save()
            logger.info(f"Tâche {task.name} mise en pause par le volontaire {volunteer.name}")
        
        elif status.lower() == 'progress':

            # Notifier le changement de statut via WebSocket
            from websocket_service.client import notify_event
            notify_event('task_status_change', {
                'workflow_id': str(workflow.id),
                'task_id': str(task.id),
                'status': 'progress',
                'volunteer': volunteer.name,
                'message': f"Tâche {task.name} en cours d'exécution par {volunteer.name}"
            })
            
            # La tâche est en cours
            task.status = 'running'
            task.save()
            volunteer_task.status = 'running'
            logger.info(f"Tâche {task.name} en cours d'exécution par le volontaire {volunteer.name}")
        
        elif status.lower() == 'stopped' or status.lower() == 'cancel':

            # Notifier le changement de statut via WebSocket
            from websocket_service.client import notify_event
            notify_event('task_status_change', {
                'workflow_id': str(workflow.id),
                'task_id': str(task.id),
                'status': 'STOPPED',
                'volunteer': volunteer.name,
                'message': f"Tâche {task.name} arrêtée par {volunteer.name}"
            })
            
            # La tâche est arrêtée
            volunteer_task.status = 'CANCEL'
            # La tâche est arrêtée
            task.status = 'cancelled'
            task.end_date = timezone.now()
            task.save()
            logger.info(f"Tâche {task.name} arrêtée par le volontaire {volunteer.name}")
        
        elif status.lower() == 'error' or status.lower() == 'failed':

            # Notifier le changement de statut via WebSocket
            from websocket_service.client import notify_event
            notify_event('task_status_change', {
                'workflow_id': str(workflow.id),
                'task_id': str(task.id),
                'status': 'FAILED',
                'volunteer': volunteer.name,
                'message': f"Tâche {task.name} a échoué sur {volunteer.name}"
            })
            
            # La tâche a échoué
            task.status = 'failed'
            task.end_date = timezone.now()
            
            # Vérifier le type d'erreur
            error_type = data.get('error_type', 'unknown')
            error_message = data.get('error_message', '')

            task.error_details = {
                'error_type': error_type,
                'error_message': error_message
            }
            task.save()

            volunteer_task.result =  {
                'error_type': error_type,
                'error_message': error_message
            }
            logger.info(f"Tâche {task.name} a échoué sur le volontaire {volunteer.name}")
            
            logger.info(f"Type d'erreur: {error_type}, Message: {error_message}")
            
            # Gérer différemment selon le type d'erreur
            if error_type.lower() == 'docker':
                # Ignorer les erreurs Docker, elles sont généralement temporaires
                # ou liées à l'infrastructure et non à la tâche elle-même
                logger.info(f"Erreur Docker ignorée pour la tâche {task.name}")
                
            elif error_type.lower() in ['user_pause', 'user_stop']:
                # Erreur due à une pause ou un arrêt utilisateur
                # Attendre 1 minute puis demander une nouvelle attribution
                logger.info(f"Erreur due à une pause/arrêt utilisateur pour la tâche {task.name}. Planification d'une réassignation.")
                
                # Importer le module pour les tâches asynchrones
                import threading
                import time
                
                def request_reassignment():
                    # Attendre 1 minute
                    time.sleep(60)
                    
                    logger.info(f"Demande de réassignation pour la tâche {task.name} après pause/arrêt utilisateur")
                    
                    # Récupérer les ressources estimées pour la tâche
                    estimated_resources = {
                        'cpu_cores': task.cpu_cores,
                        'ram': task.ram,
                        'gpu_required': task.gpu_required,
                        'gpu_memory': task.gpu_memory,
                        'storage': task.storage
                    }
                    
                    # Demander une nouvelle attribution au coordinateur
                    from redis_communication.client import RedisClient
                    from redis_communication.utils import get_manager_login_token
                    import uuid
                    
                    redis_client = RedisClient.get_instance()
                    
                    redis_client.publish(
                        'task/reassignment',
                        {
                            'task_id': str(task.id),
                            'workflow_id': str(workflow.id),
                            'estimated_resources': estimated_resources,
                            'manager_id': str(workflow.owner.remote_id) if hasattr(workflow, 'owner') and workflow.owner else None,
                            'timestamp': timezone.now().isoformat()
                        },
                        str(uuid.uuid4()),
                        get_manager_login_token(),
                        'request'
                    )
                    
                    logger.info(f"Demande de réassignation envoyée pour la tâche {task.name}")
                
                # Lancer la demande de réassignation dans un thread séparé
                reassignment_thread = threading.Thread(target=request_reassignment)
                reassignment_thread.daemon = True
                reassignment_thread.start()
                
            else:
                # Autres types d'erreurs
                logger.error(f"Erreur non gérée pour la tâche {task.name}: {error_type} - {error_message}")
                # On pourrait implémenter d'autres stratégies de gestion d'erreurs ici
        
        return True

    except Workflow.DoesNotExist:
        logger.error(f"Le workflow {workflow_id} n'existe pas")
    except Task.DoesNotExist:
        logger.error(f"La tâche {task_id} n'existe pas")
    except Volunteer.DoesNotExist:
        logger.error(f"Le volontaire {volunteer_id} n'existe pas")
    except Exception as e:
        logger.error(f"Erreur lors du traitement du statut de la tâche: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        volunteer_task.save()
        task.save()
    return False

    







def handle_task_complete(channel: str, message: Message):
    """
    Gestionnaire pour l'écoute des messages de type 'task/complete'.
    
    Args:
        channel: Canal sur lequel le message a été reçu
        message: Message reçu
        
    Returns:
        True si le message a été traité avec succès, False sinon
    """
    logger.info(f"Reçu un message de type 'task/complete' sur le canal {channel}")
    logger.debug(f"Contenu du message: {message.data}")
    
    # Récupérer les informations
    data = message.data
    
    # Vérifier que le message contient les informations nécessaires
    if 'workflow_id' not in data or 'task_id' not in data or 'volunteer_id' not in data:
        logger.error("Le message ne contient pas les informations nécessaires (workflow_id, task_id, volunteer_id)")
        return False
    
    workflow_id = data['workflow_id']
    task_id = data['task_id']
    volunteer_id = data['volunteer_id']
    result = data.get('result', {})
    
    logger.info(f"Traitement de la complétion de la tâche {task_id} par le volontaire {volunteer_id} pour le workflow {workflow_id}")
    
    try:
        # Importer les modèles nécessaires
        from django.utils import timezone
        from workflows.models import Workflow, WorkflowStatus
        from tasks.models import Task, TaskStatus
        from volunteers.models import Volunteer, VolunteerTask
        
        # Récupérer les objets
        workflow = Workflow.objects.get(id=workflow_id)
        task = Task.objects.get(id=task_id)
        volunteer = Volunteer.objects.get(coordinator_volunteer_id=volunteer_id)
        
        # Récupérer l'assignation de la tâche
        volunteer_task = VolunteerTask.objects.filter(task=task, volunteer=volunteer).first()
        
        if volunteer_task:
            # Mettre à jour le statut de l'assignation
            volunteer_task.status = "COMPLETED"
            volunteer_task.completed_at = timezone.now()
            volunteer_task.result = result
            volunteer_task.progress = 100.0
            volunteer_task.save()
            logger.info(f"Assignation mise à jour: tâche {task.name} complétée par le volontaire {volunteer.name}")

            # Mettre à jour le statut de la tâche
            task.status = TaskStatus.COMPLETED
            task.save()
            logger.info(f"Statut de la tâche {task.name} mis à jour: COMPLETED")

            # Notifier le changement de statut via WebSocket
            from websocket_service.client import notify_event
            notify_event('task_status_change', {
                'workflow_id': str(workflow.id),
                'task_id': str(task.id),
                'status': 'COMPLETED',
                'volunteer': volunteer.name,
                'message': f"Tâche {task.name} complétée par {volunteer.name}"
            })



            # Verifier si c'etait la derniere tache du workflow qui etait en running et qu'il n'y a pas de tache echouée
            running = workflow.tasks.filter(status="RUNNING").count()
            failed = workflow.tasks.filter(status="FAILED").count()
            if running == 0 and failed == 0:
                # Toutes les taches sont complétées, mettre à jour le statut du workflow
                workflow.status = WorkflowStatus.COMPLETED
                workflow.save()
                logger.info(f"Toutes les taches du workflow {workflow.name} sont complétées, statut mis à jour: COMPLETED")

                # Notifier la complétion du workflow via WebSocket
                from websocket_service.client import notify_event
                notify_event('workflow_status_change', {
                    'workflow_id': str(workflow.id),
                    'status': 'COMPLETED',
                    'message': f"Workflow {workflow.name} complété"
                })
            
            elif running==0 and failed > 0:
                # TODO estimer les ressources des taches echouées et demander une liste de volontaire pour cela
                pass

        else:
            logger.warning(f"Aucune assignation trouvée pour la tâche {task.name} et le volontaire {volunteer.name}")
            # Créer une nouvelle assignation complétée
            volunteer_task = VolunteerTask.objects.create(
                task=task,
                volunteer=volunteer,
                assigned_at=timezone.now(),
                started_at=timezone.now(),
                completed_at=timezone.now(),
                status="COMPLETED",
                result=result,
                progress=100.0
            )
            logger.info(f"Nouvelle assignation créée (complétée): tâche {task.name} complétée par le volontaire {volunteer.name}")
        
        # Mettre à jour le statut de la tâche
        task.status = "COMPLETED"
        task.save()
        logger.info(f"Statut de la tâche {task.name} mis à jour: COMPLETED")

        # Notifier le changement de statut via WebSocket
        from websocket_service.client import notify_event
        notify_event('task_status_change', {
            'workflow_id': str(workflow.id),
            'task_id': str(task.id),
            'status': 'COMPLETED',
            'volunteer': volunteer.name,
            'message': f"Tâche {task.name} complétée par {volunteer.name}"
        })
        
        # Libérer les ressources du volontaire
        volunteer.status = "available"
        volunteer.save()
        logger.info(f"Volontaire {volunteer.name} marqué comme disponible") 
        
        return True
        
    except Workflow.DoesNotExist:
        logger.error(f"Le workflow {workflow_id} n'existe pas")
    except Task.DoesNotExist:
        logger.error(f"La tâche {task_id} n'existe pas")
    except Volunteer.DoesNotExist:
        logger.error(f"Le volontaire {volunteer_id} n'existe pas")
    except Exception as e:
        logger.error(f"Erreur lors du traitement de la complétion de la tâche: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    return False


def listen_for_task_complete():
    """
    Fonction qui écoute le canal de la complétion des tâches
    
    Cette fonction s'abonne au canal 'task/complete' pour recevoir les notifications
    de complétion de tâches par les volontaires.
    
    Returns:
        bool: True si la souscription a réussi, False sinon
    """
    import logging
    from django.utils import timezone
    logger = logging.getLogger(__name__)
    
    try:
        from redis_communication.client import RedisClient
        
        client = RedisClient.get_instance()
        if not 'handle_task_complete' in client.handlers.values():
            logger.info(f"[{timezone.now()}] Souscription au canal 'task/complete'")
            client.subscribe('task/complete', handle_task_complete)
            logger.info(f"[{timezone.now()}] Souscription au canal 'task/complete' réussie")
        
        # Vérifier que le client Redis est bien connecté
        if client.running:
            logger.info(f"[{timezone.now()}] Client Redis connecté avec succès")
        else:
            logger.warning(f"[{timezone.now()}] Client Redis non connecté, les messages ne seront pas reçus")
            return False
            
        return True
    except Exception as e:
        logger.error(f"[{timezone.now()}] Erreur lors de la souscription au canal 'task/complete': {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False




def listen_for_task_accept():
    """
    Fonction qui écoute le canal de l'acceptation des tâches
    
    Cette fonction s'abonne au canal 'task/accept' pour recevoir les notifications
    d'acceptation de tâches par les volontaires.
    
    Returns:
        bool: True si la souscription a réussi, False sinon
    """
    import logging
    from django.utils import timezone
    logger = logging.getLogger(__name__)
    
    try:
        from redis_communication.client import RedisClient
        
        client = RedisClient.get_instance()
        if not 'handle_task_accept' in client.handlers.values():
            logger.info(f"[{timezone.now()}] Souscription au canal 'task/accept'")
            client.subscribe('task/accept', handle_task_accept)
            logger.info(f"[{timezone.now()}] Souscription au canal 'task/accept' réussie")
        
        # Vérifier que le client Redis est bien connecté
        if client.running:
            logger.info(f"[{timezone.now()}] Client Redis connecté avec succès")
        else:
            logger.warning(f"[{timezone.now()}] Client Redis non connecté, les messages ne seront pas reçus")
            return False
            
        return True
    except Exception as e:
        logger.error(f"[{timezone.now()}] Erreur lors de la souscription au canal 'task/accept': {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False




def listen_task_progress():
    """
    Fonction qui écoute le canal de la progression des tâches
    
    Cette fonction s'abonne au canal 'task/progress' pour recevoir les notifications
    de progression des tâches par les volontaires.
    
    Returns:
        bool: True si la souscription a réussi, False sinon
    """
    import logging
    from django.utils import timezone
    logger = logging.getLogger(__name__)
    
    try:
        from redis_communication.client import RedisClient
        
        client = RedisClient.get_instance()
        if not 'handle_task_progress' in client.handlers.values():
            logger.info(f"[{timezone.now()}] Souscription au canal 'task/progress'")
            client.subscribe('task/progress', handle_task_progress)
            logger.info(f"[{timezone.now()}] Souscription au canal 'task/progress' réussie")
        
        # Vérifier que le client Redis est bien connecté
        if client.running:
            logger.info(f"[{timezone.now()}] Client Redis connecté avec succès")
        else:
            logger.warning(f"[{timezone.now()}] Client Redis non connecté, les messages ne seront pas reçus")
            return False
            
        return True
    except Exception as e:
        logger.error(f"[{timezone.now()}] Erreur lors de la souscription au canal 'task/progress': {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False



def listen_for_task_status():
    """
        Gere les status des taches et mets à jour le statut des taches
    Returns:
        bool: True si la souscription a réussi, False sinon
    """
    

    import logging
    from django.utils import timezone
    logger = logging.getLogger(__name__)
    
    try:
        from redis_communication.client import RedisClient
        
        client = RedisClient.get_instance()
        if not 'handle_task_status' in client.handlers.values():
            logger.info(f"[{timezone.now()}] Souscription au canal 'task/status'")
            client.subscribe('task/status', handle_task_status)
            logger.info(f"[{timezone.now()}] Souscription au canal 'task/status' réussie")
        
        # Vérifier que le client Redis est bien connecté
        if client.running:
            logger.info(f"[{timezone.now()}] Client Redis connecté avec succès")
        else:
            logger.warning(f"[{timezone.now()}] Client Redis non connecté, les messages ne seront pas reçus")
            return False
            
        return True
    except Exception as e:
        logger.error(f"[{timezone.now()}] Erreur lors de la souscription au canal 'task/progress': {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    


    