#backend/workflows/coordinator_api.py
#         choices=WorkflowType.choices,

import requests
import logging
import json
from django.conf import settings
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CoordinatorAPI:
    """
    Client HTTP pour la communication avec le service Coordinateur
    """
    
    def __init__(self):
        """Initialisation du client API"""
        self.api_url = getattr(settings, 'COORDINATOR_API_URL', 'http://coordinator-service:8000/api')
        self.auth_token = None
        self.token_expiry = None
    
    def authenticate(self):
        """
        Authentification auprès du coordinateur
        
        Returns:
            bool: True si l'authentification a réussi, False sinon
        """
        try:
            # Si le token existe et est valide, on le réutilise
            if self.auth_token and self.token_expiry and datetime.now() < self.token_expiry:
                logger.debug("Utilisation du token existant")
                return True
            
            # Récupérer les identifiants de configuration
            manager_id = getattr(settings, 'MANAGER_ID', 'default_manager')
            manager_secret = getattr(settings, 'MANAGER_SECRET', None)
            
            if not manager_secret:
                logger.error("Clé secrète du manager non configurée")
                return False
            
            # Authentification auprès du coordinateur
            response = requests.post(
                f"{self.api_url}/auth/login",
                json={
                    "manager_id": manager_id,
                    "secret": manager_secret
                },
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            # Vérification de la réponse
            if response.status_code == 200:
                data = response.json()
                self.auth_token = data.get('token')
                
                # Calcul de l'expiration du token
                expiry_seconds = data.get('expires_in', 3600)  # Par défaut 1 heure
                self.token_expiry = datetime.now() + timedelta(seconds=expiry_seconds)
                
                logger.info(f"Authentification réussie, token valide jusqu'à {self.token_expiry}")
                return True
            else:
                logger.error(f"Échec de l'authentification: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Erreur lors de l'authentification: {e}")
            return False
    
    def _get_headers(self):
        """
        Récupère les en-têtes HTTP avec authentification
        
        Returns:
            dict: En-têtes HTTP
        """
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        return headers
    
    def _make_request(self, method, endpoint, data=None, params=None):
        """
        Méthode générique pour effectuer des requêtes HTTP
        
        Args:
            method (str): Méthode HTTP (GET, POST, PUT, DELETE)
            endpoint (str): Point de terminaison de l'API
            data (dict, optional): Données pour le corps de la requête
            params (dict, optional): Paramètres d'URL
            
        Returns:
            dict/None: Réponse JSON ou None en cas d'erreur
        """
        # S'assurer que l'authentification est valide
        if not self.authenticate():
            return None
        
        # Construire l'URL
        url = f"{self.api_url}/{endpoint.lstrip('/')}"
        
        try:
            # Effectuer la requête avec la méthode appropriée
            if method == "GET":
                response = requests.get(url, headers=self._get_headers(), params=params, timeout=10)
            elif method == "POST":
                response = requests.post(url, headers=self._get_headers(), json=data, timeout=10)
            elif method == "PUT":
                response = requests.put(url, headers=self._get_headers(), json=data, timeout=10)
            elif method == "DELETE":
                response = requests.delete(url, headers=self._get_headers(), params=params, timeout=10)
            else:
                logger.error(f"Méthode HTTP non supportée: {method}")
                return None
            
            # Vérifier si la réponse est un succès
            if response.status_code in [200, 201, 202, 204]:
                # Pour les réponses sans contenu
                if response.status_code == 204 or not response.text:
                    return {"success": True}
                
                # Parser la réponse JSON
                return response.json()
            else:
                logger.error(f"Erreur HTTP {response.status_code}: {response.text}")
                
                # Si l'erreur est due à un token expiré, essayer de se ré-authentifier
                if response.status_code == 401:
                    self.auth_token = None
                    if self.authenticate():
                        # Réessayer la requête
                        return self._make_request(method, endpoint, data, params)
                
                return None
                
        except Exception as e:
            logger.error(f"Erreur lors de la requête {method} {url}: {e}")
            return None
    
    # Méthodes spécifiques pour les différentes fonctionnalités
    
    def get_available_volunteers(self, required_resources=None):
        """
        Récupère la liste des volontaires disponibles
        
        Args:
            required_resources (dict, optional): Ressources nécessaires pour le filtrage
            
        Returns:
            list: Liste des volontaires disponibles
        """
        params = {}
        if required_resources:
            params["resources"] = json.dumps(required_resources)
        
        result = self._make_request("GET", "/volunteers/available", params=params)
        return result or []
    
    def get_volunteer_status(self, volunteer_id):
        """
        Récupère le statut d'un volontaire spécifique
        
        Args:
            volunteer_id (str): ID du volontaire
            
        Returns:
            dict: Informations sur le volontaire
        """
        return self._make_request("GET", f"/volunteers/{volunteer_id}")
    
    def register_workflow(self, workflow):
        """
        Enregistre un nouveau workflow auprès du coordinateur
        
        Args:
            workflow (Workflow): Le workflow à enregistrer
            
        Returns:
            dict: Informations sur l'enregistrement du workflow
        """
        data = {
            "workflow_id": str(workflow.id),
            "name": workflow.name,
            "workflow_type": workflow.workflow_type,
            "description": workflow.description,
            "min_volunteers": workflow.min_volunteers,
            "max_volunteers": workflow.max_volunteers,
            "estimated_duration": workflow.estimated_duration,
            "priority": workflow.priority
        }
        
        return self._make_request("POST", "/workflows/register", data=data)
    
    def register_task(self, task):
        """
        Enregistre une nouvelle tâche auprès du coordinateur
        
        Args:
            task (Task): La tâche à enregistrer
            
        Returns:
            dict: Informations sur l'enregistrement de la tâche
        """
        data = {
            "task_id": str(task.id),
            "workflow_id": str(task.workflow.id),
            "name": task.name,
            "description": task.description,
            "command": task.command,
            "parameters": task.parameters,
            "required_resources": task.required_resources,
            "estimated_duration": task.estimated_duration if hasattr(task, 'estimated_duration') else None,
            "status": task.status,
            "priority": task.priority if hasattr(task, 'priority') else "medium",
            "docker_image": task.docker_image
        }
        
        # Ajouter les informations sur les blocs de matrices si présentes
        if None not in [task.matrix_block_row_start, task.matrix_block_row_end,
                      task.matrix_block_col_start, task.matrix_block_col_end]:
            data["matrix_block"] = {
                "row_start": task.matrix_block_row_start,
                "row_end": task.matrix_block_row_end,
                "col_start": task.matrix_block_col_start,
                "col_end": task.matrix_block_col_end
            }
        
        return self._make_request("POST", "/tasks/register", data=data)
    
    def assign_task(self, task_id, volunteer_id):
        """
        Demande l'attribution d'une tâche à un volontaire spécifique
        
        Args:
            task_id (str): ID de la tâche
            volunteer_id (str): ID du volontaire
            
        Returns:
            dict: Résultat de l'attribution
        """
        data = {
            "task_id": str(task_id),
            "volunteer_id": volunteer_id
        }
        
        return self._make_request("POST", "/tasks/assign", data=data)
    
    def update_task_status(self, task_id, status, progress=None, result=None, error=None):
        """
        Met à jour le statut d'une tâche
        
        Args:
            task_id (str): ID de la tâche
            status (str): Nouveau statut
            progress (float, optional): Progression (0-100)
            result (dict, optional): Résultat de la tâche
            error (dict, optional): Détails d'erreur
            
        Returns:
            dict: Résultat de la mise à jour
        """
        data = {
            "task_id": str(task_id),
            "status": status
        }
        
        if progress is not None:
            data["progress"] = progress
            
        if result is not None:
            data["result"] = result
            
        if error is not None:
            data["error"] = error
        
        return self._make_request("PUT", "/tasks/status", data=data)
    
    def update_workflow_status(self, workflow_id, status, progress=None, result=None, error=None):
        """
        Met à jour le statut d'un workflow
        
        Args:
            workflow_id (str): ID du workflow
            status (str): Nouveau statut
            progress (float, optional): Progression (0-100)
            result (dict, optional): Résultat du workflow
            error (dict, optional): Détails d'erreur
            
        Returns:
            dict: Résultat de la mise à jour
        """
        data = {
            "workflow_id": str(workflow_id),
            "status": status
        }
        
        if progress is not None:
            data["progress"] = progress
            
        if result is not None:
            data["result"] = result
            
        if error is not None:
            data["error"] = error
        
        return self._make_request("PUT", "/workflows/status", data=data)
    
    def get_task_result_location(self, task_id):
        """
        Récupère l'emplacement du résultat d'une tâche
        
        Args:
            task_id (str): ID de la tâche
            
        Returns:
            dict: Informations sur l'emplacement du résultat
        """
        return self._make_request("GET", f"/tasks/{task_id}/result_location")
    
    def notify_docker_image_ready(self, task_id, image_name):
        """
        Notifie le coordinateur qu'une image Docker est prête
        
        Args:
            task_id (str): ID de la tâche
            image_name (str): Nom de l'image Docker
            
        Returns:
            dict: Résultat de la notification
        """
        data = {
            "task_id": str(task_id),
            "docker_image": image_name
        }
        
        return self._make_request("POST", "/tasks/docker_image_ready", data=data)
    
    def notify_matrix_result_ready(self, workflow_id, result_file):
        """
        Notifie le coordinateur qu'un résultat de calcul matriciel est prêt
        
        Args:
            workflow_id (str): ID du workflow
            result_file (str): Chemin vers le fichier de résultat
            
        Returns:
            dict: Résultat de la notification
        """
        data = {
            "workflow_id": str(workflow_id),
            "result_file": result_file
        }
        
        return self._make_request("POST", "/workflows/matrix_result_ready", data=data)