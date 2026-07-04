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


def process_openmalaria_submission(workflow_id, request=None):
    """
    Soumet un workflow OpenMalaria pour traitement avec un nombre de tâches et une population par tâche.
    
    Args:
        workflow_id (str): ID du workflow.
        request.data: Doit contenir 'num_tasks' (int) et 'population_per_task' (int).
    
    Returns:
        JsonResponse: Statut de la soumission.
    """
    try:
        # Récupérer le workflow
        workflow = get_object_or_404(Workflow, id=workflow_id)

        if workflow.workflow_type != WorkflowType.OPEN_MALARIA:
            return JsonResponse({
                'error': 'Le workflow doit être de type OPEN_MALARIA'
            }, status=400)

        # Étude globale partitionnée (valeurs par défaut = charge réaliste)
        metadata = workflow.metadata or {}
        num_tasks = int(metadata.get('num_tasks', 8))
        total_population = int(metadata.get('total_population', 160000))
        population_per_task = int(
            metadata.get('population_per_task', max(1, total_population // num_tasks))
        )
        if request is not None and hasattr(request, 'data'):
            if request.data.get('num_tasks') is not None:
                num_tasks = int(request.data.get('num_tasks'))
            if request.data.get('total_population') is not None:
                total_population = int(request.data.get('total_population'))
            if request.data.get('population_per_task') is not None:
                population_per_task = int(request.data.get('population_per_task'))
            if request.data.get('simulation_days') is not None:
                metadata['simulation_days'] = int(request.data.get('simulation_days'))
            if request.data.get('monte_carlo_runs') is not None:
                metadata['monte_carlo_runs'] = int(request.data.get('monte_carlo_runs'))

        if num_tasks < 1 or population_per_task < 1 or total_population < 1:
            return JsonResponse({
                'error': 'num_tasks, total_population et population_per_task doivent être positifs'
            }, status=400)
        # Cohérence: population totale = somme des partitions
        if metadata.get('total_population') is None:
            total_population = num_tasks * population_per_task

        # Chemins de travail persistants
        data_root = '/data' if os.path.isdir('/data') else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not workflow.executable_path:
            workflow.executable_path = os.path.join(
                data_root, 'workflow_data', str(workflow.owner_id or 'anon'), str(workflow.id)
            )
        if not workflow.input_path:
            workflow.input_path = os.path.join(workflow.executable_path, 'inputs')
        if not workflow.output_path:
            workflow.output_path = os.path.join(workflow.executable_path, 'outputs')
        os.makedirs(workflow.executable_path, exist_ok=True)
        os.makedirs(workflow.input_path, exist_ok=True)
        os.makedirs(workflow.output_path, exist_ok=True)
        metadata['num_tasks'] = num_tasks
        metadata['total_population'] = total_population
        metadata['population_per_task'] = population_per_task
        metadata.setdefault('simulation_days', 3650)
        metadata.setdefault('monte_carlo_runs', 12)
        metadata['paradigm'] = 'partition_simulate_aggregate'
        workflow.metadata = metadata
        workflow.save()
        
        # Notifier le début de la soumission
        notify_event('workflow_status_change', {
            'workflow_id': str(workflow_id),
            'status': 'SUBMITTING',
            'message': 'Soumission du workflow OpenMalaria en cours...'
        })
        
        # Vérifier les volontaires disponibles avec gestion robuste
        from workflows.handlers import submit_workflow_handler
        
        try:
            result = submit_workflow_handler(str(workflow_id))
            
            # Vérification robuste du résultat
            if result is None:
                logger.error(f"submit_workflow_handler a retourné None pour le workflow {workflow_id}")
                notify_event('workflow_status_change', {
                    'workflow_id': str(workflow_id),
                    'status': 'SUBMISSION_FAILED',
                    'message': 'Erreur interne: la fonction de soumission n\'a pas retourné de résultat'
                })
                return JsonResponse({
                    'success': False,
                    'error': 'Erreur interne lors de la soumission du workflow'
                }, status=500)
            
            # Déballage sécurisé
            if not isinstance(result, (tuple, list)) or len(result) != 2:
                logger.error(f"submit_workflow_handler a retourné un format invalide: {type(result)}")
                notify_event('workflow_status_change', {
                    'workflow_id': str(workflow_id),
                    'status': 'SUBMISSION_FAILED',
                    'message': 'Erreur interne: format de réponse invalide'
                })
                return JsonResponse({
                    'success': False,
                    'error': 'Format de réponse invalide du gestionnaire de workflow'
                }, status=500)
            
            success, response = result
            logger.info(f"Submit workflow response: {response}")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'appel à submit_workflow_handler: {e}")
            logger.error(traceback.format_exc())
            notify_event('workflow_status_change', {
                'workflow_id': str(workflow_id),
                'status': 'SUBMISSION_FAILED',
                'message': f'Erreur lors de la vérification des volontaires: {str(e)}'
            })
            return JsonResponse({
                'success': False,
                'error': f'Erreur lors de la vérification des volontaires: {str(e)}'
            }, status=500)
        
        # Vérifier le succès de la soumission
        if not success:
            error_message = response.get('message', 'Erreur inconnue') if isinstance(response, dict) else 'Erreur inconnue'
            logger.warning(f"Échec de la soumission du workflow {workflow_id}: {error_message}")
            notify_event('workflow_status_change', {
                'workflow_id': str(workflow_id),
                'status': 'SUBMISSION_FAILED',
                'message': f"Échec de la soumission: {error_message}"
            })
            return JsonResponse({
                'success': False,
                'response': response
            }, status=400)
        
        logger.info(f"Dossier de sortie: {workflow.output_path}")

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
        
        # Lancer le traitement asynchrone
        def process_workflow_async():
            """Traitement asynchrone du workflow avec gestion complète des erreurs"""
            thread_logger = logging.getLogger(f"workflow_thread_{workflow_id}")
            thread_logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '%(asctime)s - [THREAD %(threadName)s] - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            thread_logger.addHandler(handler)
            
            thread_logger.info(f"===== Début du traitement asynchrone du workflow {workflow_id} =====")
            
            try:
                # Démarrer le serveur de fichiers
                from tasks.file_server import start_file_server
                
                thread_logger.info("Démarrage du serveur de fichiers...")
                server_port = start_file_server(workflow)
                
                if not server_port:
                    thread_logger.error("Impossible de démarrer le serveur de fichiers")
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
                
                if not tasks:
                    raise Exception("Aucune tâche n'a été créée lors du découpage")
                
                thread_logger.info(f"{len(tasks)} tâches créées avec succès")
                
                # Récupérer l'adresse IP
                from redis_communication.utils import get_local_ip
                
                ip_address = get_local_ip()
                if not ip_address:
                    thread_logger.error("Impossible de récupérer l'adresse IP locale")
                    raise Exception("Erreur lors de la récupération de l'adresse IP")
                
                thread_logger.info(f"Adresse IP du serveur: {ip_address}")
                file_server_url = f"http://{ip_address}:{server_port}"
                
                # Notifier la fin du découpage
                notify_event('workflow_status_change', {
                    'workflow_id': str(workflow_id),
                    'status': 'SPLIT_COMPLETED',
                    'message': f'Découpage terminé, {len(tasks)} tâches créées'
                })
                
                # Assigner les tâches (ou attendre un volontaire — jamais d'echec pour ca)
                volunteers = response.get('volunteers') if isinstance(response, dict) else None
                from tasks.assignment import assign_and_publish

                assign_result = assign_and_publish(workflow, volunteers, server_port)
                thread_logger.info("Assignation: %s", assign_result)
                workflow.refresh_from_db()

                # Ecoute des retours de taches (best-effort)
                try:
                    from tasks.handlers import (
                        listen_for_task_accept,
                        listen_for_task_complete,
                        listen_for_task_status,
                        listen_task_progress,
                    )
                    listen_for_task_accept()
                    listen_for_task_complete()
                    listen_for_task_status()
                    listen_task_progress()
                except Exception as listen_err:
                    thread_logger.warning("Ecoute taches partielle: %s", listen_err)

                notify_event('workflow_status_change', {
                    'workflow_id': str(workflow_id),
                    'status': workflow.status,
                    'message': assign_result.get(
                        'message',
                        'Soumission terminee',
                    ),
                })
                
                thread_logger.info(f"===== Fin du traitement asynchrone du workflow {workflow_id} =====")
            
            except Exception as e:
                thread_logger.error(f"ERREUR lors du traitement: {e}")
                thread_logger.error(traceback.format_exc())

                # Si les taches existent deja, on attend un volontaire au lieu d'echouer
                try:
                    workflow.refresh_from_db()
                    if workflow.tasks.exists():
                        workflow.status = WorkflowStatus.PENDING
                        workflow.save(update_fields=['status', 'updated_at'])
                        notify_event('workflow_status_change', {
                            'workflow_id': str(workflow_id),
                            'status': 'PENDING',
                            'message': (
                                'Soumission OK. Taches creees, en attente de volontaires '
                                f'(detail: {e})'
                            ),
                        })
                    else:
                        workflow.status = WorkflowStatus.FAILED
                        workflow.save(update_fields=['status', 'updated_at'])
                        notify_event('workflow_status_change', {
                            'workflow_id': str(workflow_id),
                            'status': 'FAILED',
                            'message': f'Erreur lors du traitement: {str(e)}',
                        })
                except Exception as save_error:
                    thread_logger.error(f"Impossible de sauvegarder le statut: {save_error}")
                
                thread_logger.error("===== Fin du traitement avec ERREUR =====")
        
        # Démarrer le thread
        thread_name = f"workflow-{workflow_id}-thread"
        thread = threading.Thread(target=process_workflow_async, name=thread_name, daemon=True)
        logger.info(f"Démarrage du thread '{thread_name}'")
        thread.start()
        
        # Retour immédiat avec code 202 (Accepted) pour éviter les timeouts
        return JsonResponse({
            'success': True,
            'message': 'Workflow OpenMalaria soumis, traitement en cours en arrière-plan',
            'workflow_id': str(workflow_id),
            'num_tasks': num_tasks,
            'population_per_task': population_per_task
        }, status=202)
    
    except Workflow.DoesNotExist:
        logger.error(f"Workflow {workflow_id} non trouvé")
        return JsonResponse({
            'success': False,
            'error': 'Workflow non trouvé'
        }, status=404)
    
    except Exception as e:
        logger.error(f"Erreur inattendue: {e}")
        logger.error(traceback.format_exc())
        
        notify_event('workflow_status_change', {
            'workflow_id': str(workflow_id),
            'status': 'ERROR',
            'message': f'Erreur inattendue: {str(e)}'
        })
        
        return JsonResponse({
            'success': False,
            'error': f'Erreur inattendue: {str(e)}'
        }, status=500)


@api_view(['POST'])
def submit_openmalaria_workflow_view(request, workflow_id):
    """API view pour la soumission OpenMalaria (délègue à process_openmalaria_submission)."""
    return process_openmalaria_submission(workflow_id, request)