"""
Utilitaires divers pour le module de communication Redis.
"""

import os
import time
import json
import logging
from typing import Dict, Any, Optional
import jwt
from .exceptions import NoLoginError
logger = logging.getLogger(__name__)

def generate_token(client_id: str, client_type: str, expiration_hours: int = 24) -> str:
    """
    Génère un token JWT pour l'authentification.
    
    Args:
        client_id: ID du client
        client_type: Type de client (coordinator, manager, volunteer)
        expiration_hours: Durée de validité en heures
        
    Returns:
        str: Token JWT
    """
    from django.conf import settings
    secret_key = getattr(settings, 'SECRET_KEY', 'default-secret-key')
    
    payload = {
        'client_id': client_id,
        'client_type': client_type,
        'exp': int(time.time()) + expiration_hours * 3600,
        'iat': int(time.time())
    }
    
    return jwt.encode(payload, secret_key, algorithm='HS256')

def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Vérifie un token JWT.
    
    Args:
        token: Token JWT à vérifier
        
    Returns:
        Dict ou None: Payload du token si valide, None sinon
    """
    from django.conf import settings
    secret_key = getattr(settings, 'SECRET_KEY', 'default-secret-key')
    
    try:
        payload = jwt.decode(token, secret_key, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Token expiré")
        return None
    except jwt.InvalidTokenError:
        logger.warning("Token invalide")
        return None

def format_timestamp(timestamp: float) -> str:
    """
    Formate un timestamp en chaîne ISO 8601.
    
    Args:
        timestamp: Timestamp UNIX
        
    Returns:
        str: Chaîne au format ISO 8601
    """
    from datetime import datetime
    return datetime.fromtimestamp(timestamp).isoformat()


def _manager_state_paths():
    """Chemins persistants (/data) ou locaux."""
    roots = []
    if os.path.isdir('/data'):
        roots.append('/data/.manager')
    roots.append('.manager')
    return roots


def get_manager_login_token(user=None):
    """
    Recupere le token coordinateur du manager connecte ou du fichier legacy.
    """
    if user and getattr(user, 'coordinator_token', None):
        return user.coordinator_token
    try:
        from workflows.models import User
        token_user = (
            User.objects.exclude(coordinator_token__isnull=True)
            .exclude(coordinator_token='')
            .order_by('-id')
            .first()
        )
        if token_user and token_user.coordinator_token:
            return token_user.coordinator_token
    except Exception:
        pass
    for root in _manager_state_paths():
        path = os.path.join(root, 'manager_login_info.json')
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                return data['token']
        except FileNotFoundError:
            continue
    raise NoLoginError("Le fichier .manager/manager_login_info.json n'a pas été trouvé")


def get_coordinator_token_for_workflow(workflow):
    """Token coordinateur propre au proprietaire du workflow."""
    owner = getattr(workflow, 'owner', None)
    if owner and owner.coordinator_token:
        return owner.coordinator_token
    return get_manager_login_token(owner)
    


def get_manager_id():
    """
    Récupère l'ID du manager à partir du fichier .manager/manager_info.json.
    
    Returns:
        str: ID du manager
    """
    try:
        from workflows.models import User
        user = (
            User.objects.exclude(remote_id__isnull=True)
            .exclude(remote_id='')
            .order_by('-id')
            .first()
        )
        if user and user.remote_id:
            return user.remote_id
    except Exception:
        pass
    for root in _manager_state_paths():
        for name in ('manager_info.json', 'manager_login_info.json'):
            path = os.path.join(root, name)
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    return data.get('remote_id') or data.get('manager_id')
            except FileNotFoundError:
                continue
    raise NoLoginError("Le fichier .manager/manager_info.json n'a pas été trouvé")


def get_local_ip():
    try:
        # Connexion fictive pour obtenir l'IP utilisée sur le réseau local
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de l'IP locale : {e}")
        return None


def get_manager_public_url():
    """URL publique du manager (accessible par les volontaires hors Docker)."""
    import os
    return (
        os.environ.get("MANAGER_PUBLIC_URL")
        or os.environ.get("PUBLIC_MANAGER_URL")
        or "https://manager-vc-uy.npe-techs.com"
    ).rstrip("/")


def build_task_file_transfer_info(workflow, task, file_server_port=None):
    """
    Metadonnees de transfert de fichiers pour une tache assignee.
    Utilise l'API publique HTTPS plutot que le serveur de fichiers local.
    """
    public_url = get_manager_public_url()
    return {
        "files": task.input_files,
        "file_server": {
            "host": public_url.replace("https://", "").replace("http://", ""),
            "port": 443 if public_url.startswith("https") else 80,
            "base_url": f"{public_url}/api/workflow-files/{workflow.id}",
            "mode": "public_api",
        },
        "result_upload_url": f"{public_url}/api/tasks/{task.id}/outputs/",
    }