from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.utils import timezone
import logging
import threading
import uuid
import traceback
from workflows.models import Workflow, WorkflowStatus, WorkflowType
from tasks.models import Task
from redis_communication.client import RedisClient
from websocket_service.client import notify_event
import os
logger = logging.getLogger(__name__)

@api_view(['POST'])
def submit_openmalaria_workflow_view(request, workflow_id):
    """
    Soumet un workflow OpenMalaria pour traitement avec un nombre de tâches et une population par tâche.
    
    Args:
        workflow_id (str): ID du workflow.
        request.data: Doit contenir 'num_tasks' (int) et 'population_per_task' (int).
    
    Returns:
        JsonResponse: Statut de la soumission.
    """
    try:
        # Valider les paramètres
        num_tasks = 4 # request.data.get('num_tasks')
        request.timeout = 300  # 5 minutes
        population_per_task = 2000 # request.data.get('population_per_task')
        if not isinstance(num_tasks, int) or not isinstance(population_per_task, int):
            return JsonResponse({
                'error': 'num_tasks et population_per_task doivent être des entiers'
            }, status=400)
        if num_tasks < 1 or population_per_task < 1:
            return JsonResponse({
                'error': 'num_tasks et population_per_task doivent être positifs'
            }, status=400)
        
        # Récupérer le workflow
        workflow = get_object_or_404(Workflow, id=workflow_id)
        if workflow.workflow_type != WorkflowType.OPEN_MALARIA:
            return JsonResponse({
                'error': 'Le workflow doit être de type OPEN_MALARIA'
            }, status=400)
        
        # Notifier le début de la soumission
        notify_event('workflow_status_change', {
            'workflow_id': str(workflow_id),
            'status': 'SUBMITTING',
            'message': 'Soumission du workflow OpenMalaria en cours...'
        })
        
        # Vérifier les volontaires disponibles
        from workflows.handlers import submit_workflow_handler
        success, response = submit_workflow_handler(str(workflow_id))
        logger.info(f"Submit workflow response: {response}")
        
        if not success:
            notify_event('workflow_status_change', {
                'workflow_id': str(workflow_id),
                'status': 'SUBMISSION_FAILED',
                'message': f"Échec de la soumission: {response.get('message', 'Erreur inconnue')}"
            })
            return JsonResponse({'success': False, 'response': response}, status=400)
        
        # Mettre à jour le statut
        workflow.status = WorkflowStatus.SPLITTING
        workflow.submitted_at = timezone.now()
        workflow.save()
        logger.info(f"Workflow {workflow_id} soumis avec succès")
        
        notify_event('workflow_status_change', {
            'workflow_id': str(workflow_id),
            'status': 'SPLITTING',
            'message': f'Découpage en {num_tasks} tâches avec population {population_per_task} en cours...'
        })
        
        response_data = {'success': True, 'message': 'Workflow OpenMalaria soumis, traitement en cours'}
        
        # Lancer le traitement asynchrone
        def process_workflow_async():
            thread_logger = logging.getLogger(f"workflow_thread_{workflow_id}")
            thread_logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - [THREAD %(threadName)s] - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            thread_logger.addHandler(handler)
            
            thread_logger.info(f"===== Début du traitement asynchrone du workflow {workflow_id} =====")
            try:
                # Démarrer le serveur de fichiers
                from tasks.file_server import start_file_server
                server_port = start_file_server(workflow)
                if not server_port:
                    thread_logger.error(f"Impossible de démarrer le serveur de fichiers")
                    workflow.status = WorkflowStatus.FAILED
                    workflow.save()
                    notify_event('workflow_status_change', {
                        'workflow_id': str(workflow_id),
                        'status': 'FAILED',
                        'message': 'Impossible de démarrer le serveur de fichiers'
                    })
                    return
                
                thread_logger.info(f"Serveur de fichiers démarré sur le port {server_port}")
                
                # Découpage du workflow
                thread_logger.info(f"Lancement du découpage pour {num_tasks} tâches")
                from workflows.split_workflow import split_workflow
                tasks = split_workflow(
                    id=workflow_id,
                    workflow_type=WorkflowType.OPEN_MALARIA,
                    logger=thread_logger,
                    num_tasks=num_tasks,
                    population_per_task=population_per_task
                )
                thread_logger.info(f"{len(tasks)} tâches créées")
                
                # Récupérer l'adresse IP
                from redis_communication.utils import get_local_ip
                ip_address = get_local_ip()
                if not ip_address:
                    thread_logger.error(f"Impossible de récupérer l'adresse IP locale")
                    raise Exception("Erreur lors de la récupération de l'adresse IP")
                thread_logger.info(f"Adresse IP du serveur: {ip_address}")
                file_server_url = f"http://{ip_address}:{server_port}"
                
                # Notifier la fin du découpage
                notify_event('workflow_status_change', {
                    'workflow_id': str(workflow_id),
                    'status': 'SPLIT_COMPLETED',
                    'message': f'Découpage terminé, {len(tasks)} tâches créées'
                })
                
                # Assigner les tâches
                if response.get('volunteers'):
                    thread_logger.info(f"Volontaires disponibles: {len(response.get('volunteers'))}")
                    workflow.status = WorkflowStatus.ASSIGNING
                    workflow.save()
                    
                    notify_event('workflow_status_change', {
                        'workflow_id': str(workflow_id),
                        'status': 'ASSIGNING',
                        'message': 'Attribution des tâches aux volontaires...'
                    })
                    
                    from tasks.scheduller import assign_workflow_to_volunteers
                    assignment_result = assign_workflow_to_volunteers(workflow, response.get('volunteers'))
                    thread_logger.info(f"Résultat de l'assignation: {assignment_result}")
                    

                    redis_client = RedisClient.get_instance()

                    notify_event('workflow_status_change', {
                        'workflow_id': str(workflow_id),
                        'status': 'ASSIGNED',
                        'message': 'Tâches attribuées avec succès'
                    })
                    
                    # Écoute des canaux
                    from tasks.handlers import listen_for_task_accept, listen_for_task_complete, listen_for_task_status, listen_task_progress
                    accept_success = listen_for_task_accept()
                    complete_success = listen_for_task_complete()
                    status_success = listen_for_task_status()
                    progress_success = listen_task_progress()
                    if accept_success and complete_success and status_success and progress_success:
                        thread_logger.info(f"Souscription aux canaux réussie")
                    else:
                        thread_logger.warning(f"Problème lors de la souscription aux canaux")
                    
                    # Publier les assignations enrichies
                    from redis_communication.utils import get_local_ip, get_manager_login_token
                    server_ip = get_local_ip()
                    for volunteer_id, task_list in assignment_result.items():
                        enriched_tasks = []
                        for task_info in task_list:
                            task_id = task_info['task_id']
                            task = Task.objects.get(id=task_id)
                            enriched_task = {
                                'task_id': task_id,
                                'name': task.name,
                                'description': task.description,
                                'command': task.command,
                                'dependencies': task.dependencies,
                                'is_subtask': task.is_subtask,
                                'status': task.status,
                                'required_resources': task.required_resources,
                                'attempts': task.attempts,
                                'workflow_id': str(workflow.id),
                                'parameters': task.parameters,
                                'estimated_execution_time': task.estimated_max_time,
                                'input_data': {
                                    'files': task.input_files,
                                    'file_server': {
                                        'host': server_ip,
                                        'port': server_port,
                                        'base_url': f'http://{server_ip}:{server_port}'
                                    }
                                },
                                'input_data_size': task.input_size,
                                'docker_information': task.docker_info or {}
                            }
                            enriched_tasks.append(enriched_task)
                        
                        # Publish assignments for each volunteer individually
                        redis_client.publish('task/assignment', {
                            'workflow_id': str(workflow_id),
                            'assignments': {
                                volunteer_id: enriched_tasks
                            },
                        }, str(uuid.uuid4()), get_manager_login_token(), 'request')
                
                else:
                    thread_logger.info(f"Aucun volontaire reçu, écoute sur le canal d'assignment")
                    pubsub = RedisClient.get_instance()
                    pubsub.subscribe('workflow/assignment')
                    notify_event('workflow_status_change', {
                        'workflow_id': str(workflow_id),
                        'status': 'WAITING_VOLUNTEERS',
                        'message': 'En attente de volontaires disponibles...'
                    })
                    from workflows.handlers import listen_for_volunteers
                    listen_for_volunteers(workflow_id)
                
                notify_event('workflow_status_change', {
                    'workflow_id': str(workflow_id),
                    'status': workflow.status,
                    'message': 'Processus de soumission terminé'
                })
                thread_logger.info(f"===== Fin du traitement asynchrone du workflow {workflow_id} =====")
            
            except Exception as e:
                thread_logger.error(f"ERREUR lors du traitement: {e}")
                thread_logger.error(traceback.format_exc())
                workflow.status = WorkflowStatus.FAILED
                workflow.save()
                notify_event('workflow_status_change', {
                    'workflow_id': str(workflow_id),
                    'status': 'ERROR',
                    'message': f'Erreur lors du traitement: {str(e)}'
                })
                thread_logger.error(f"===== Fin du traitement avec ERREUR =====")
        
        thread_name = f"workflow-{workflow_id}-thread"
        thread = threading.Thread(target=process_workflow_async, name=thread_name)
        thread.daemon = True
        logger.info(f"Démarrage du thread '{thread_name}'")
        thread.start()
        
        return JsonResponse(response_data, status=200)
    
    except Workflow.DoesNotExist:
        logger.error(f"Workflow {workflow_id} non trouvé")
        return JsonResponse({'error': 'Workflow not found.'}, status=404)
    except Exception as e:
        logger.error(f"Erreur inattendue: {e}")
        logger.error(traceback.format_exc())
        notify_event('workflow_status_change', {
            'workflow_id': str(workflow_id),
            'status': 'ERROR',
            'message': f'Erreur inattendue: {str(e)}'
        })
        return JsonResponse({'error': f'Unexpected error: {str(e)}'}, status=500)