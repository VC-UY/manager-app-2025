"""
Gestionnaires pour les workflows dans le système de communication Redis.
"""

import json
import logging
import math
import uuid
from typing import Dict, Any, Optional, Callable
from django.utils import timezone
import time
from redis_communication.message import Message
from redis_communication.client import RedisClient
from redis_communication.auth_client import save_response, get_response, delete_response
from redis_communication.utils import get_manager_login_token, NoLoginError
from workflows.models import Workflow, WorkflowStatus, WorkflowType
from workflows.examples.distributed_training_demo.estimate_resources import estimate_resources as estimate_ml_training_resources




def estimate_ml_inference_resources(input_data_size: int) -> Dict[str, Any]:
    """Estime les ressources pour l'inférence ML à partir de la taille des données (Mo)."""
    memory_mb = max(1024, int(input_data_size * 64))
    return {
        'cpu_cores': max(1, min(8, math.ceil(input_data_size / 500))),
        'memory_mb': memory_mb,
        'disk_space_mb': max(2048, input_data_size * 10),
        'gpu': input_data_size > 1000,
    }


def estimate_matrix_resources(input_data_size: int, operation: str = 'add') -> Dict[str, Any]:
    """Estime les ressources pour des opérations matricielles distribuées."""
    size_mb = max(1, input_data_size or 64)
    multiplier = 3 if operation == 'multiply' else 2
    memory_mb = max(512, size_mb * multiplier * 8)
    return {
        'cpu_cores': max(1, min(16, math.ceil(size_mb / 256))),
        'memory_mb': memory_mb,
        'disk_space_mb': max(1024, size_mb * 4),
        'gpu': False,
    }


def estimate_matrix_addition_resources(input_data_size: int) -> Dict[str, Any]:
    return estimate_matrix_resources(input_data_size, 'add')


def estimate_matrix_multiplication_resources(input_data_size: int) -> Dict[str, Any]:
    return estimate_matrix_resources(input_data_size, 'multiply')


def estimate_custom_resources(workflow_metadata: dict) -> Dict[str, Any]:
    """Estime les ressources d'un workflow CUSTOM à partir de ses métadonnées."""
    metadata = workflow_metadata or {}
    tasks = metadata.get('tasks', [])
    num_tasks = len(tasks) if tasks else int(metadata.get('num_tasks', 1))
    per_task_ram = int(metadata.get('ram_mb_per_task', 512))
    return {
        'cpu_cores': max(1, min(32, num_tasks)),
        'memory_mb': per_task_ram * num_tasks,
        'disk_space_mb': int(metadata.get('disk_space_mb', 1024)),
        'gpu': bool(metadata.get('gpu_required', False)),
    }

logger = logging.getLogger(__name__)


def estimate_open_malaria_resources(num_task: int) -> Dict[str, Any]:
    """
    Estime les ressources nécessaires pour un workflow Open Malaria.
    
    Args:
        num_task: Nombre de tâches à exécuter
        
    Returns:
        Dict: Ressources estimées
    """

    return {
        'cpu_cores': 2 * num_task,
        'memory_mb': 2048 * num_task,  # 1 Go par tâche
        'disk_space_mb': 1,
        'gpu': False  # Open Malaria n'utilise pas de GPU
    }

def submit_workflow_handler(workflow_id: str, callback: Optional[Callable[[Dict[str, Any]], None]] = None, timeout: int = 60) -> (bool, Dict[str, Any]):
    """
    Gestionnaire pour soumettre un workflow au coordinateur.
    
    Args:
        workflow_id: ID du workflow à soumettre
        callback: Fonction de rappel appelée avec la réponse (optionnel)
        timeout: Délai d'attente en secondes (défaut: 60)
        
    Returns:
        bool, Dict: Résultat de la soumission
    """
    try:
        # Récupérer le workflow
        workflow = Workflow.objects.get(id=workflow_id)
        
        # Vérifier que le workflow est dans un état valide pour la soumission
        if workflow.status != WorkflowStatus.CREATED:
            logger.warning(f"Workflow {workflow_id} n'est pas dans un état valide pour la soumission (status={workflow.status})")
            return False, {
                'status': 'error',
                'message': f"Workflow n'est pas dans un état valide pour la soumission (status={workflow.status})"
            }
        
        # Estimer les ressources
        estimated_resources = None
        if workflow.workflow_type == WorkflowType.ML_TRAINING:
            try:
                path = workflow.input_path or workflow.executable_path
                estimated_resources = estimate_ml_training_resources(path) if path else None
            except Exception as est_err:
                logger.warning(f"Estimation ML fallback: {est_err}")
                estimated_resources = None
            if not estimated_resources:
                estimated_resources = {
                    'cpu_cores': 2,
                    'memory_mb': 2048,
                    'disk_space_mb': 4096,
                    'gpu': False,
                }
        elif workflow.workflow_type == WorkflowType.ML_INFERENCE:
            estimated_resources = estimate_ml_inference_resources(workflow.input_data_size)
        elif workflow.workflow_type == WorkflowType.MATRIX_ADDITION:
            estimated_resources = estimate_matrix_addition_resources(workflow.input_data_size)
        elif workflow.workflow_type == WorkflowType.MATRIX_MULTIPLICATION:
            estimated_resources = estimate_matrix_multiplication_resources(workflow.input_data_size)
        elif workflow.workflow_type == WorkflowType.OPEN_MALARIA:
            estimated_resources = estimate_open_malaria_resources(
                (workflow.metadata or {}).get('num_tasks', 2)
            )
        elif workflow.workflow_type == WorkflowType.CUSTOM:
            estimated_resources = estimate_custom_resources(workflow.metadata)
        else:
            logger.warning(f"Type de workflow non supporté pour l'estimation des ressources: {workflow.workflow_type}")
            return False, {
                'status': 'error',
                'message': f"Type de workflow non supporté pour l'estimation des ressources: {workflow.workflow_type}"
            }
        
        # Mettre à jour le statut du workflow
        if estimated_resources is not None:
            workflow.estimated_resources = estimated_resources
            workflow.save()
            logger.warning("Mise à jour des ressources estimées")
        else:
            logger.warning(f"Estimation des ressources echouée pour le workflow: {estimated_resources}")
        
        owner_remote_id = getattr(workflow.owner, 'remote_id', None)
        if not owner_remote_id:
            return False, {
                'status': 'error',
                'message': "Manager non synchronise avec le coordinateur (remote_id manquant)",
            }

        # Préparer les données pour Redis
        data = {
            'workflow_id': str(workflow.id),
            'workflow_name': workflow.name,
            'workflow_description': workflow.description,
            'description': workflow.description,
            'workflow_status': workflow.status,
            'created_at': workflow.created_at.isoformat() if hasattr(workflow.created_at, 'isoformat') else str(workflow.created_at),
            'workflow_type': workflow.workflow_type,
            'owner': owner_remote_id,
            'priority': workflow.priority,
            'estimated_resources': estimated_resources,
            'max_execution_time': workflow.max_execution_time,
            'input_data_size': workflow.input_data_size,
            'submitted_at': timezone.now().isoformat(),
            'attempts': workflow.retry_count,
            'attemps': workflow.retry_count,
        }

        token = get_manager_login_token(workflow.owner)
        logger.info(f"Soumission du workflow {workflow_id} au coordinateur (RPC)")
        workflow.status = WorkflowStatus.SUBMITTED
        workflow.save(update_fields=['status', 'estimated_resources', 'updated_at'])

        from redis_communication.proxy_rpc import proxy_request_response

        success, response_data = proxy_request_response(
            'workflow/submit',
            'workflow/submit_response',
            data,
            token=token,
            sender_id=str(owner_remote_id),
            timeout=float(timeout),
        )
        if callback:
            try:
                callback(response_data)
            except Exception:
                pass

        if success:
            logger.info(f"Soumission reussie pour {workflow_id}")
            return True, response_data

        logger.info(f"Soumission echouee pour {workflow_id}: {response_data}")
        workflow.status = WorkflowStatus.CREATED
        workflow.save(update_fields=['status', 'updated_at'])
        return False, response_data

    except Workflow.DoesNotExist:
        logger.error(f"Workflow {workflow_id} non trouvé")
        return False, {
            'status': 'error',
            'message': 'Workflow non trouvé'
        }
    except NoLoginError:
        logger.error("Le fichier .manager/manager_login_info.json n'a pas été trouvé. Veuillez vous connecter.")
        return False, {
            'status': 'error',
            'message': 'Le fichier .manager/manager_login_info.json n\'a pas été trouvé. Veuillez vous connecter.'
        }
    except Exception as e:
        logger.error(f"Erreur lors de la soumission du workflow {workflow_id}: {e}")
        return False, {
            'status': 'error',
            'message': f'Erreur lors de la soumission: {str(e)}'
        }




def handle_volunteers_list(channel: str, message: Message):
    """
    Fonction qui gère la réception de la liste des volontaires
    """
    logger.info(f"Réception de la liste des volontaires sur le canal {channel}")
    logger.info(f"Message reçu: {message}")

    # Verifier la liste 

    # Appeler l'ordonnanceur

    #  Afficher le result

    return True, message
    
    


def listen_for_volunteers(workflow_id: str):
    """
    Fonction qui ecoute le canal de la liste des volontaires
    """
    
    client = RedisClient.get_instance()
    client.subscribe('volunteers/list', handle_volunteers_list)
    