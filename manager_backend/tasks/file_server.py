"""
Serveur de fichiers simple pour servir les fichiers d'entrée aux volontaires.
"""

import os
import threading
import logging
import http.server
import socketserver
from typing import Optional
from workflows.models import Workflow
import json
import time
from datetime import datetime

# Dictionnaire pour stocker les serveurs en cours d'exécution par workflow
active_servers = {}


logger = logging.getLogger('file_server')

# Configuration du logger pour les requêtes
request_logger = logging.getLogger('file_server_requests')
request_logger.setLevel(logging.INFO)

# Créer un handler pour le fichier de log des requêtes
request_handler = logging.FileHandler(os.path.join(os.path.dirname(__file__), 'file_server_requests.log'))
request_handler.setFormatter(
    logging.Formatter('%(message)s')
)
request_logger.addHandler(request_handler)

class WorkflowFileHandler(http.server.SimpleHTTPRequestHandler):
    """Gestionnaire HTTP personnalisé pour servir les fichiers d'un workflow."""
    
    def __init__(self, *args, workflow_base_dir=None, **kwargs):
        self.workflow_base_dir = workflow_base_dir
        self.response_size = 0
        self.response_status = 200
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Gérer les requêtes GET avec logging"""
        # Enregistrer le début de la requête
        start_time = time.time()
        start_datetime = datetime.now().isoformat()
        
        try:
            # Appeler la méthode parente pour traiter la requête
            self.response_status = 200
            super().do_GET()
        except Exception as e:
            self.response_status = 500
            logger.error(f"Erreur lors du traitement de la requête: {e}")
        finally:
            # Calculer la durée et obtenir les informations
            end_time = time.time()
            end_datetime = datetime.now().isoformat()
            duration = end_time - start_time
            
            # Préparer les données de log
            log_data = {
                "request_id": str(int(start_time * 1000)),  # ID unique basé sur le timestamp
                "client_address": self.client_address[0],
                "requested_path": self.path,
                "file_size": self.response_size,
                "start_time": start_datetime,
                "end_time": end_datetime,
                "duration_seconds": round(duration, 3),
                "status_code": self.response_status
            }
            
            # Enregistrer les données au format JSON
            request_logger.info(json.dumps(log_data))

    def send_response(self, code, message=None):
        """Surcharge pour capturer le code de statut"""
        self.response_status = code
        super().send_response(code, message)

    def send_header(self, keyword, value):
        """Surcharge pour capturer la taille du fichier"""
        if keyword.lower() == 'content-length':
            self.response_size = int(value)
        super().send_header(keyword, value)

    def translate_path(self, path):
        """Traduit le chemin de l'URL en chemin de fichier local."""
        # Supprimer les paramètres de requête s'il y en a
        if '?' in path:
            path = path.split('?', 1)[0]
        
        # Supprimer les fragments s'il y en a
        if '#' in path:
            path = path.split('#', 1)[0]
        
        # Normaliser le chemin
        path = path.split('?', 1)[0]
        path = path.split('/', 1)[1] if path.startswith('/') else path
        
        # Construire le chemin complet
        return os.path.join(self.workflow_base_dir, path)
    
    def log_message(self, format, *args):
        """Rediriger les logs vers le logger de l'application."""
        logger.info(f"FileServer: {format % args}")

def start_file_server(workflow: Workflow, port: int = 0) -> int:
    """
    Démarre un serveur de fichiers pour un workflow spécifique.
    
    Args:
        workflow: L'instance du workflow
        port: Port sur lequel démarrer le serveur (0 = port aléatoire)
        
    Returns:
        Le port sur lequel le serveur a été démarré
    """
    workflow_id = str(workflow.id)
    
    # Vérifier si un serveur est déjà en cours pour ce workflow
    if workflow_id in active_servers:
        logger.info(f"Un serveur de fichiers est déjà en cours pour le workflow {workflow_id}")
        return active_servers[workflow_id]['port']
    
    # Déterminer le répertoire de base pour les fichiers du workflow
    workflow_base_dir = workflow.input_path
    
    if not os.path.exists(workflow_base_dir):
        os.makedirs(workflow_base_dir)
        logger.warning(f"Le répertoire du workflow {workflow_id} n'existe pas, il a été créé: {workflow_base_dir}")
    
    # Créer un gestionnaire de requêtes avec le répertoire de base du workflow
    handler = lambda *args, **kwargs: WorkflowFileHandler(*args, workflow_base_dir=workflow_base_dir, **kwargs)
    
    # Créer et démarrer le serveur
    httpd = socketserver.ThreadingTCPServer(("", port), handler)
    server_port = httpd.server_address[1]
    
    # Démarrer le serveur dans un thread séparé
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    # Enregistrer le serveur actif
    active_servers[workflow_id] = {
        'server': httpd,
        'thread': server_thread,
        'port': server_port
    }
    
    logger.info(f"Serveur de fichiers démarré pour le workflow {workflow_id} sur le port {server_port}")
    return server_port

def get_file_server_url(workflow_id: str) -> Optional[str]:
    """
    Récupère l'URL du serveur de fichiers pour un workflow spécifique.
    
    Args:
        workflow_id: ID du workflow
        
    Returns:
        URL du serveur de fichiers si trouvé, None sinon
    """
    if workflow_id not in active_servers:
        logger.warning(f"Aucun serveur de fichiers en cours pour le workflow {workflow_id}")
        return None
    
    # Récupérer l'adresse IP et le port du serveur
    import socket
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    port = active_servers[workflow_id]['port']
    
    return f"http://{ip_address}:{port}"

def stop_file_server(workflow_id: str) -> bool:
    """
    Arrête le serveur de fichiers pour un workflow spécifique.
    
    Args:
        workflow_id: ID du workflow
        
    Returns:
        True si le serveur a été arrêté, False sinon
    """
    if workflow_id not in active_servers:
        logger.warning(f"Aucun serveur de fichiers en cours pour le workflow {workflow_id}")
        return False
    
    # Arrêter le serveur
    try:
        active_servers[workflow_id]['server'].shutdown()
        active_servers[workflow_id]['server'].server_close()
        del active_servers[workflow_id]
        logger.info(f"Serveur de fichiers arrêté pour le workflow {workflow_id}")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de l'arrêt du serveur de fichiers pour le workflow {workflow_id}: {e}")
        return False
