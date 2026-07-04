"""
Gestionnaires d'événements pour les messages Redis.
Inclut les gestionnaires pour l'authentification des managers et des volontaires.
"""

import logging
import json
import os
import uuid
import time
from typing import Dict, Any, Optional
from datetime import datetime
from django.contrib.auth.hashers import make_password, check_password
from django.conf import settings

from .message import Message
from .utils import generate_token

logger = logging.getLogger(__name__)

# Répertoire pour stocker les requêtes en attente
PENDING_REQUESTS_DIR = os.path.join(settings.BASE_DIR, 'pending_requests')
os.makedirs(PENDING_REQUESTS_DIR, exist_ok=True)

def save_pending_request(request_id: str, data: Dict[str, Any]):
    """
    Enregistre une requête en attente dans un fichier.
    
    Args:
        request_id: ID de la requête
        data: Données de la requête
    """
    filename = os.path.join(PENDING_REQUESTS_DIR, f"{request_id}.json")
    with open(filename, 'w') as f:
        json.dump({
            'data': data,
            'timestamp': time.time()
        }, f)
    
    logger.debug(f"Requête {request_id} enregistrée dans {filename}")

def get_pending_request(request_id: str) -> Optional[Dict[str, Any]]:
    """
    Récupère une requête en attente.
    
    Args:
        request_id: ID de la requête
        
    Returns:
        Dict ou None: Données de la requête si trouvée, None sinon
    """
    filename = os.path.join(PENDING_REQUESTS_DIR, f"{request_id}.json")
    if not os.path.exists(filename):
        return None
    
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Erreur lors de la lecture de la requête {request_id}: {e}")
        return None

def delete_pending_request(request_id: str) -> bool:
    """
    Supprime une requête en attente.
    
    Args:
        request_id: ID de la requête
        
    Returns:
        bool: True si supprimée, False sinon
    """
    filename = os.path.join(PENDING_REQUESTS_DIR, f"{request_id}.json")
    if not os.path.exists(filename):
        return False
    
    try:
        os.remove(filename)
        logger.debug(f"Requête {request_id} supprimée")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la suppression de la requête {request_id}: {e}")
        return False

# Gestionnaires génériques

def log_message_handler(channel: str, message: Message):
    """
    Gestionnaire simple qui journalise les messages reçus.
    
    Args:
        channel: Canal sur lequel le message a été reçu
        message: Message reçu
    """
    logger.info(f"Message reçu sur {channel}: {message.request_id} de {message.sender}")
    logger.debug(f"Contenu: {message.data}")

def heartbeat_handler(channel: str, message: Message):
    """
    Gestionnaire pour les messages de heartbeat.
    
    Args:
        channel: Canal sur lequel le message a été reçu
        message: Message reçu
    """
    sender_type = message.sender.get('type', 'unknown') if isinstance(message.sender, dict) else 'unknown'
    sender_id = message.sender.get('id', 'unknown') if isinstance(message.sender, dict) else 'unknown'
    data = message.data or {}
    volunteer_id = data.get('volunteer_id') or (
        sender_id if sender_type == 'volunteer' else None
    )
    if volunteer_id:
        try:
            from volunteers.presence import mark_online
            mark_online(
                str(volunteer_id),
                name=data.get('username') or data.get('name'),
                resources=data.get('resources') or {},
                status=data.get('status') or 'available',
                preferences=data.get('preferences') or None,
            )
        except Exception as exc:
            logger.warning("Heartbeat volontaire ignore: %s", exc)
    logger.debug(f"Heartbeat reçu de {sender_type}:{sender_id}")


def volunteer_heartbeat_handler(channel: str, message: Message):
    """Canal dédié volunteer/heartbeat."""
    heartbeat_handler(channel, message)


def volunteer_disconnect_handler(channel: str, message: Message):
    """Volontaire déconnecté explicitement."""
    data = message.data or {}
    volunteer_id = data.get('volunteer_id')
    if not volunteer_id and isinstance(message.sender, dict):
        volunteer_id = message.sender.get('id')
    if volunteer_id:
        try:
            from volunteers.presence import mark_offline
            mark_offline(str(volunteer_id), reason="disconnect")
        except Exception as exc:
            logger.warning("Disconnect volontaire ignore: %s", exc)

def error_handler(channel: str, message: Message):
    """
    Gestionnaire pour les messages d'erreur.
    
    Args:
        channel: Canal sur lequel le message a été reçu
        message: Message reçu
    """
    error_data = message.data
    error_msg = error_data.get('message', 'Erreur inconnue')
    error_code = error_data.get('code', 0)
    
    logger.error(f"Erreur sur {channel}: [{error_code}] {error_msg}")
    logger.error(f"Détails: {error_data}")

# Gestionnaires pour l'authentification des managers

def manager_registration_handler(channel: str, message: Message):
    """
    Gestionnaire pour l'enregistrement des managers.
    
    Args:
        channel: Canal sur lequel le message a été reçu
        message: Message reçu
    """
    from .client import RedisClient
    
    logger.info(f"Demande d'enregistrement de manager reçue: {message.request_id}")
    
    # Récupérer les données du message
    data = message.data
    request_id = message.request_id
    
    # Vérifier que les données sont complètes
    required_fields = ['username', 'email', 'password']
    for field in required_fields:
        if field not in data:
            logger.error(f"Champ requis manquant: {field}")
            
            # Envoyer une réponse d'erreur
            client = RedisClient.get_instance()
            client.publish('auth/register_response', {
                'status': 'error',
                'message': f"Champ requis manquant: {field}"
            }, request_id=request_id)
            return
    
    # Enregistrer la requête en attente
    save_pending_request(request_id, data)
    
    # Récupérer les données
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    try:
        # Vérifier si le manager existe déjà
        existing_manager = Manager.objects(username=username).first()
        if existing_manager:
            logger.warning(f"Le manager {username} existe déjà")
            
            # Envoyer une réponse d'erreur
            client = RedisClient.get_instance()
            client.publish('auth/register_response', {
                'status': 'error',
                'message': "Ce nom d'utilisateur est déjà utilisé"
            }, request_id=request_id)
            
            # Supprimer la requête en attente
            delete_pending_request(request_id)
            return
        
        existing_email = Manager.objects(email=email).first()
        if existing_email:
            logger.warning(f"L'email {email} est déjà utilisé")
            
            # Envoyer une réponse d'erreur
            client = RedisClient.get_instance()
            client.publish('auth/register_response', {
                'status': 'error',
                'message': "Cet email est déjà utilisé"
            }, request_id=request_id)
            
            # Supprimer la requête en attente
            delete_pending_request(request_id)
            return
        
        # Créer le manager
        hashed_password = make_password(password)
        
        manager = Manager(
            username=username,
            email=email,
            password=hashed_password,
            status='active'  # Activer directement le compte pour simplifier
        )
        manager.save()
        
        logger.info(f"Manager {username} enregistré avec succès (ID: {manager.id})")
        
        # Envoyer une réponse de succès
        client = RedisClient.get_instance()
        client.publish('auth/register_response', {
            'status': 'success',
            'message': 'Manager enregistré avec succès',
            'manager_id': str(manager.id),
            'username': manager.username,
            'email': manager.email
        }, request_id=request_id)
        
        # Supprimer la requête en attente
        delete_pending_request(request_id)
        
    except Exception as e:
        logger.error(f"Erreur lors de l'enregistrement du manager: {e}")
        
        # Envoyer une réponse d'erreur
        client = RedisClient.get_instance()
        client.publish('auth/register_response', {
            'status': 'error',
            'message': str(e)
        }, request_id=request_id)
        
        # Supprimer la requête en attente
        delete_pending_request(request_id)

def manager_login_handler(channel: str, message: Message):
    """
    Gestionnaire pour l'authentification des managers.
    
    Args:
        channel: Canal sur lequel le message a été reçu
        message: Message reçu
    """
    from .client import RedisClient
    
    logger.info(f"Demande d'authentification de manager reçue: {message.request_id}")
    
    # Récupérer les données du message
    data = message.data
    request_id = message.request_id
    
    # Vérifier que les données sont complètes
    required_fields = ['username', 'password']
    for field in required_fields:
        if field not in data:
            logger.error(f"Champ requis manquant: {field}")
            
            # Envoyer une réponse d'erreur
            client = RedisClient.get_instance()
            client.publish('auth/login_response', {
                'status': 'error',
                'message': f"Champ requis manquant: {field}"
            }, request_id=request_id)
            return
    
    # Enregistrer la requête en attente
    save_pending_request(request_id, data)
    
    # Récupérer les données
    username = data.get('username')
    password = data.get('password')
    
    try:
        # Rechercher le manager
        manager = Manager.objects(username=username).first()
        if not manager:
            logger.warning(f"Manager {username} introuvable")
            
            # Envoyer une réponse d'erreur
            client = RedisClient.get_instance()
            client.publish('auth/login_response', {
                'status': 'error',
                'message': 'Identifiants invalides'
            }, request_id=request_id)
            
            # Supprimer la requête en attente
            delete_pending_request(request_id)
            return
        
        # Vérifier le mot de passe
        if not check_password(password, manager.password):
            logger.warning(f"Mot de passe incorrect pour {username}")
            
            # Envoyer une réponse d'erreur
            client = RedisClient.get_instance()
            client.publish('auth/login_response', {
                'status': 'error',
                'message': 'Identifiants invalides'
            }, request_id=request_id)
            
            # Supprimer la requête en attente
            delete_pending_request(request_id)
            return
        
        # Vérifier que le compte est actif
        if manager.status != 'active':
            logger.warning(f"Le compte {username} n'est pas actif")
            
            # Envoyer une réponse d'erreur
            client = RedisClient.get_instance()
            client.publish('auth/login_response', {
                'status': 'error',
                'message': "Ce compte n'est pas actif"
            }, request_id=request_id)
            
            # Supprimer la requête en attente
            delete_pending_request(request_id)
            return
        
        # Générer un token JWT et un refresh token
        token = generate_token(str(manager.id), 'manager', 24)  # 24 heures
        refresh_token = generate_token(str(manager.id), 'manager', 168)  # 7 jours
        
        # Mettre à jour la date de dernière connexion
        manager.last_login = datetime.utcnow()
        manager.save()
        
        logger.info(f"Manager {username} authentifié avec succès")
        
        # Envoyer une réponse de succès
        client = RedisClient.get_instance()
        client.publish('auth/login_response', {
            'status': 'success',
            'message': 'Authentification réussie',
            'token': token,
            'refresh_token': refresh_token,
            'manager_id': str(manager.id),
            'username': manager.username,
            'email': manager.email
        }, request_id=request_id)
        
        # Supprimer la requête en attente
        delete_pending_request(request_id)
        
    except Exception as e:
        logger.error(f"Erreur lors de l'authentification du manager: {e}")
        
        # Envoyer une réponse d'erreur
        client = RedisClient.get_instance()
        client.publish('auth/login_response', {
            'status': 'error',
            'message': str(e)
        }, request_id=request_id)
        
        # Supprimer la requête en attente
        delete_pending_request(request_id)

# Gestionnaires pour l'authentification des volontaires

def volunteer_registration_handler(message: Message):
    """
    Gestionnaire pour l'enregistrement des volontaires.
    
    Args:
        channel: Canal sur lequel le message a été reçu
        message: Message reçu
    """
    from .client import RedisClient
    
    logger.info(f"Demande d'enregistrement de volontaire reçue: {message.request_id}")
    
    # Récupérer les données du message
    data = message.data
    request_id = message.request_id
    
    # Vérifier que les données sont complètes
    required_fields = ['name', 'node_id', 'cpu_model', 'cpu_cores', 'total_ram', 
                      'available_storage', 'operating_system', 'ip_address', 'communication_port']
    for field in required_fields:
        if field not in data:
            logger.error(f"Champ requis manquant: {field}")
            
            # Envoyer une réponse d'erreur
            client = RedisClient.get_instance()
            client.publish('auth/volunteer_register_response', {
                'status': 'error',
                'message': f"Champ requis manquant: {field}"
            }, request_id=request_id)
            return
    

def default_handler(channel: str, message: Message):
    logger.warning(f" (default_handler) - Message reçu sur le canal {channel}: {message}")

def handle_task_assignment_response_wrapper(channel: str, message: Message):
    """
    Wrapper pour le gestionnaire de réponses aux demandes de réassignation de tâches.
    Importe le gestionnaire réel à partir du module tasks.reassignment_handlers.

    Args:
        channel: Canal sur lequel le message a été reçu
        message: Message reçu
    """
    try:
        from tasks.reassignment_handlers import handle_task_assignment_response
        return handle_task_assignment_response(channel, message)
    except Exception as e:
        logger.error(f"Erreur lors de l'appel au gestionnaire de réponses aux demandes de réassignation de tâches: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def handle_task_progress(channel: str, message: Message):
    """
    Handler pour les mises à jour de progression des tâches.
    Reçoit les progressions depuis le coordinateur et met à jour la base locale + WebSocket.
    """
    try:
        data = message.data
        task_id = data.get('task_id')
        workflow_id = data.get('workflow_id')
        progress = data.get('progress', 0)
        volunteer_id = data.get('volunteer_id')

        if not task_id:
            logger.warning("task/progress reçu sans task_id")
            return

        logger.info(f"[task/progress] Tâche {task_id}: {progress}% (volontaire: {volunteer_id})")

        # Mettre à jour la base de données locale
        try:
            from tasks.models import Task
            task = Task.objects.filter(id=task_id).first()
            if task:
                task.progress = progress
                task.save(update_fields=['progress'])
        except Exception as db_error:
            logger.error(f"Erreur mise à jour DB pour task {task_id}: {db_error}")

        # Notifier via WebSocket
        try:
            from channels.layers import get_channel_layer
            from asgiref.sync import async_to_sync

            channel_layer = get_channel_layer()
            if channel_layer:
                async_to_sync(channel_layer.group_send)(
                    "workflow_updates",
                    {
                        "type": "task_progress",
                        "task_id": task_id,
                        "workflow_id": workflow_id,
                        "progress": progress,
                        "volunteer": volunteer_id,
                        "message": f"Progression: {progress}%",
                        "timestamp": data.get('timestamp')
                    }
                )
                logger.info(f"WebSocket notification envoyée pour task {task_id}")
        except Exception as ws_error:
            logger.error(f"Erreur WebSocket pour task {task_id}: {ws_error}")

    except Exception as e:
        logger.error(f"Erreur dans handle_task_progress: {e}")
        import traceback
        logger.error(traceback.format_exc())


def handle_task_status_update(channel: str, message: Message):
    """
    Handler pour les mises à jour de statut des tâches (completed, failed, etc.).
    """
    try:
        data = message.data
        task_id = data.get('task_id')
        workflow_id = data.get('workflow_id')
        status = data.get('status', '').lower()
        volunteer_id = data.get('volunteer_id')

        if not task_id:
            logger.warning("task/status reçu sans task_id")
            return

        logger.info(f"[task/status] Tâche {task_id}: statut={status} (volontaire: {volunteer_id})")

        # Mettre à jour la base de données locale
        try:
            from tasks.models import Task
            from workflows.models import Workflow

            task = Task.objects.filter(id=task_id).first()
            if task:
                # Mapper les statuts
                status_map = {
                    'completed': 'COMPLETED',
                    'failed': 'FAILED',
                    'error': 'FAILED',
                    'running': 'RUNNING',
                    'progress': 'RUNNING',
                    'paused': 'PAUSED',
                    'cancel': 'CANCELLED',
                }
                task.status = status_map.get(status, status.upper())
                
                # Si la tâche est terminée, forcer la progression à 100%
                if task.status == 'COMPLETED':
                    task.progress = 100.0
                    task.save(update_fields=['status', 'progress'])
                else:
                    task.save(update_fields=['status'])

                # Si completed, mettre à jour le workflow si toutes les tâches sont terminées
                if status == 'completed' and workflow_id:
                    workflow = Workflow.objects.filter(workflow_id=workflow_id).first()
                    if workflow:
                        all_tasks = Task.objects.filter(workflow=workflow)
                        completed_count = all_tasks.filter(status='COMPLETED').count()
                        total_count = all_tasks.count()

                        if completed_count == total_count:
                            workflow.status = 'COMPLETED'
                            workflow.save(update_fields=['status'])
                            logger.info(f"Workflow {workflow_id} marqué comme COMPLETED")

        except Exception as db_error:
            logger.error(f"Erreur mise à jour DB pour task {task_id}: {db_error}")

        # Notifier via WebSocket
        try:
            from channels.layers import get_channel_layer
            from asgiref.sync import async_to_sync

            channel_layer = get_channel_layer()
            if channel_layer:
                async_to_sync(channel_layer.group_send)(
                    "workflow_updates",
                    {
                        "type": "task_status_change",
                        "task_id": task_id,
                        "workflow_id": workflow_id,
                        "status": status,
                        "volunteer": volunteer_id,
                        "message": f"Statut: {status}",
                        "timestamp": data.get('timestamp')
                    }
                )
        except Exception as ws_error:
            logger.error(f"Erreur WebSocket pour task {task_id}: {ws_error}")

    except Exception as e:
        logger.error(f"Erreur dans handle_task_status_update: {e}")
        import traceback
        logger.error(traceback.format_exc())


def handle_task_files_ready(channel: str, message: Message):
    """
    Handler pour les notifications de fichiers de sortie prêts.
    Télécharge les fichiers depuis le coordinateur vers le manager.
    """
    import requests
    import os

    try:
        data = message.data
        task_id = data.get('task_id')
        workflow_id = data.get('workflow_id')
        files = data.get('files', [])
        file_server = data.get('file_server', {})

        if not task_id or not files:
            logger.warning("task_files sans task_id ou files")
            return

        logger.info(f"[task_files] Fichiers prêts pour tâche {task_id}: {files}")

        # Créer le répertoire de destination
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, 'workflows', 'outputs', str(workflow_id), str(task_id))
        os.makedirs(output_dir, exist_ok=True)

        # Télécharger les fichiers depuis le coordinateur
        coordinator_host = file_server.get('coordinator_host', 'localhost')
        coordinator_port = file_server.get('coordinator_port', 8001)
        path = file_server.get('path', f'/api/task-outputs/{task_id}/')

        downloaded = []
        for filename in files:
            try:
                url = f"http://{coordinator_host}:{coordinator_port}{path}{filename}"
                logger.info(f"Téléchargement de {url}")

                response = requests.get(url, timeout=60, stream=True)
                if response.status_code == 200:
                    local_path = os.path.join(output_dir, filename)
                    with open(local_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    downloaded.append(filename)
                    logger.info(f"Fichier {filename} téléchargé: {local_path}")
                else:
                    logger.warning(f"Échec téléchargement {filename}: HTTP {response.status_code}")

            except Exception as e:
                logger.error(f"Erreur téléchargement {filename}: {e}")

        # Mettre à jour la tâche avec les fichiers de sortie
        if downloaded:
            try:
                from tasks.models import Task
                task = Task.objects.filter(id=task_id).first()
                if task:
                    task.output_files = downloaded
                    task.output_directory = output_dir
                    task.save(update_fields=['output_files', 'output_directory'])
                    logger.info(f"Tâche {task_id} mise à jour avec {len(downloaded)} fichiers")
            except Exception as db_error:
                logger.error(f"Erreur mise à jour DB: {db_error}")

        # Notifier via WebSocket
        try:
            from channels.layers import get_channel_layer
            from asgiref.sync import async_to_sync

            channel_layer = get_channel_layer()
            if channel_layer:
                async_to_sync(channel_layer.group_send)(
                    "workflow_updates",
                    {
                        "type": "task_update",
                        "task": {
                            "task_id": task_id,
                            "workflow_id": workflow_id,
                            "output_files": downloaded,
                            "output_directory": output_dir
                        },
                        "action": "files_ready",
                        "timestamp": data.get('timestamp')
                    }
                )
        except Exception as ws_error:
            logger.error(f"Erreur WebSocket: {ws_error}")

    except Exception as e:
        logger.error(f"Erreur dans handle_task_files_ready: {e}")
        import traceback
        logger.error(traceback.format_exc())


def handle_coordinator_task_assigned_wrapper(channel: str, message: Message):
    from tasks.handlers import handle_coordinator_task_assigned
    return handle_coordinator_task_assigned(channel, message)


# Dictionnaire des gestionnaires par défaut
DEFAULT_HANDLERS = {
    # Canaux génériques / présence volontaires
    "coord/heartbeat": heartbeat_handler,
    "volunteer/heartbeat": volunteer_heartbeat_handler,
    "volunteer/disconnect": volunteer_disconnect_handler,
    "coord/emergency": error_handler,
    "system/error": error_handler,

    # Canaux de réassignation de tâches
    "task/assignment/response": lambda channel, message: handle_task_assignment_response_wrapper(channel, message),

    # Canaux de progression et statut des tâches
    "task/progress": handle_task_progress,
    "task/status": handle_task_status_update,

    # Canal pour les fichiers de sortie
    "manager/task_files": handle_task_files_ready,

    # Assignation faite par le Coordinateur
    "coordinator/task_assigned": handle_coordinator_task_assigned_wrapper,
}
