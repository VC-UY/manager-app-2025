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
    sender_type = message.sender.get('type', 'unknown')
    sender_id = message.sender.get('id', 'unknown')
    logger.debug(f"Heartbeat reçu de {sender_type}:{sender_id}")

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

# Dictionnaire des gestionnaires par défaut
DEFAULT_HANDLERS = {
    # Canaux génériques
    "coord/heartbeat": heartbeat_handler,
    "coord/emergency": error_handler,
    "system/error": error_handler,
    
    # Canaux de réassignation de tâches
    "task/assignment/response": lambda channel, message: handle_task_assignment_response_wrapper(channel, message),
}
