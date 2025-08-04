"""
Gestionnaires pour les workflows dans le système de communication Redis.
"""

import json
import logging
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
    """
    Estime les ressources nécessaires pour un workflow d'inférence de modèle ML.
    
    Args:
        input_data_size: Taille des données d'entrée en mégaoctets
        
    Returns:
        Dict: Ressources estimées
    """
    # TODO: Implémenter l'estimation des ressources
    return {
        'cpu_cores': 2,
        'memory_mb': 4096,
        'disk_space_mb': 10000,
        'gpu': True
    }

def estimate_matrix_addition_resources(input_data_size: int) -> Dict[str, Any]:
    """
    Estime les ressources nécessaires pour un workflow d'addition de matrices.
    
    Args:
        input_data_size: Taille des données d'entrée en mégaoctets
        
    Returns:
        Dict: Ressources estimées
    """
    # TODO: Implémenter l'estimation des ressources
    return {
        'cpu_cores': 2,
        'memory_mb': 4096,
        'disk_space_mb': 10000,
        'gpu': True
    }

def estimate_matrix_multiplication_resources(input_data_size: int) -> Dict[str, Any]:
    """
    Estime les ressources nécessaires pour un workflow de multiplication de matrices.
    
    Args:
        input_data_size: Taille des données d'entrée en mégaoctets
        
    Returns:
        Dict: Ressources estimées
    """
    # TODO: Implémenter l'estimation des ressources
    return {
        'cpu_cores': 2,
        'memory_mb': 4096,
        'disk_space_mb': 10000,
        'gpu': True
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
            estimated_resources = estimate_ml_training_resources(workflow.input_data_size)
        elif workflow.workflow_type == WorkflowType.ML_INFERENCE:
            estimated_resources = estimate_ml_inference_resources(workflow.input_data_size)
        elif workflow.workflow_type == WorkflowType.MATRIX_ADDITION:
            estimated_resources = estimate_matrix_addition_resources(workflow.input_data_size)
        elif workflow.workflow_type == WorkflowType.MATRIX_MULTIPLICATION:
            estimated_resources = estimate_matrix_multiplication_resources(workflow.input_data_size)
        elif workflow.workflow_type == WorkflowType.OPEN_MALARIA:
            estimated_resources = estimate_open_malaria_resources(workflow.metadata.get('num_task', 4))
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
        
        # Préparer les données pour Redis
        data = {
            'workflow_id': str(workflow.id),
            'workflow_name': workflow.name,
            'workflow_description': workflow.description,
            'workflow_status': workflow.status,
            'created_at': workflow.created_at.isoformat() if hasattr(workflow.created_at, 'isoformat') else str(workflow.created_at),
            'workflow_type': workflow.workflow_type,
            'owner': workflow.owner.remote_id,
            'priority': workflow.priority,
            'estimated_resources': estimated_resources,
            'max_execution_time': workflow.max_execution_time,
            'input_data_size': workflow.input_data_size,
            'submitted_at': timezone.now().isoformat(),
            'attemps' : workflow.retry_count,
        }
        
        # Générer et enregistrere un request_id 
        request_id = str(uuid.uuid4())
        
        def handle_response(channel: str, message: Message):
            # Vérifier si c'est une réponse à une soumission de workflow
            if message.request_id == request_id:
                logger.info(f"Réponse reçue pour la requête {request_id} sur le canal {channel}")
                # Enregistrer la réponse
                save_response(request_id, message.data)
                logger.info(f"Réponse enregistrée: {message.data}")
                
                # Appeler le callback si fourni
                if callback:
                    logger.info(f"Appel du callback pour la réponse {request_id}")
                    callback(message.data)
                
                # Se désabonner du canal
                logger.info(f"Désabonnement du canal {channel} pour la requête {request_id}")
                client.unsubscribe('workflow/submit_response', handle_response)
            else:
                logger.warning(f"Réponse ignorée: request_id ne correspond pas ({message.request_id} != {request_id})")

        
        # Obtenir l'instance du client Redis
        client = RedisClient.get_instance()
        token = get_manager_login_token()
        # Publier le message avec le token JWT
        logger.info(f"Soumission du workflow {workflow_id} au coordinateur")
        client.subscribe('workflow/submit_response', handle_response)
        workflow.status = WorkflowStatus.SUBMITTED
        workflow.save()
        client.publish(
            'workflow/submit', 
            data, 
            request_id=request_id,
            token=token,
            message_type="request"
        )
        start_time = time.time()
        while time.time() - start_time < timeout:
            response = get_response(request_id)
            if response:
                logger.info(f"Réponse trouvée pour la requête {request_id}: {response}")
                # Supprimer la réponse du fichier
                delete_response(request_id)
                
                # Vérifier le statut
                response_data = response.get('response', {})
                status = response_data.get('status')
                logger.info(f"Statut de la réponse: {status}")
                
                if status == 'success':
                    logger.info(f"Enregistrement réussi pour {workflow_id}")
                    return True, response_data
                else:
                    logger.info(f"Enregistrement échoué pour {workflow_id}")
                    return False, response_data
        
    except Workflow.DoesNotExist:
        logger.error(f"Workflow {workflow_id} non trouvé")
        return {
            'status': 'error',
            'message': 'Workflow non trouvé'
        }
    except NoLoginError:
        logger.error("Le fichier .manager/manager_login_info.json n'a pas été trouvé. Veuillez vous connecter.")
        return {
            'status': 'error',
            'message': 'Le fichier .manager/manager_login_info.json n\'a pas été trouvé. Veuillez vous connecter.'
        }
    except Exception as e:
        logger.error(f"Erreur lors de la soumission du workflow {workflow_id}: {e}")
        return {
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
    