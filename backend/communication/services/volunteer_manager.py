import json
import time
import logging
import threading
from django.conf import settings
from .auth_manager import AuthManager

logger = logging.getLogger(__name__)

class VolunteerManager:
    """
    Gestionnaire des volontaires pour la communication avec le Coordinateur
    """
    
    def __init__(self, mqtt_client, auth_manager):
        self.mqtt_client = mqtt_client
        self.auth_manager = auth_manager
        self.volunteers = {}
        self.last_update = 0
        self.update_lock = threading.Lock()
        self.update_event = threading.Event()
        
        # S'abonner au topic des volontaires disponibles
        def on_connect(client, userdata, flags, rc):
            client.subscribe("volunteers/available")
            client.subscribe("volunteers/status/#")
        
        def on_message(client, userdata, msg):
            try:
                if msg.topic == "volunteers/available":
                    self.handle_volunteers_list(json.loads(msg.payload.decode()))
                elif msg.topic.startswith("volunteers/status/"):
                    volunteer_id = msg.topic.split('/')[-1]
                    self.handle_volunteer_status(volunteer_id, json.loads(msg.payload.decode()))
            except Exception as e:
                logger.error(f"Erreur de traitement du message: {e}")
        
        self.mqtt_client.on_connect = on_connect
        self.mqtt_client.on_message = on_message
    
    def handle_volunteers_list(self, data):
        """
        Traite la liste des volontaires reçue
        """
        with self.update_lock:
            volunteer_list = data.get('volunteers', [])
            
            # Mise à jour de la liste des volontaires
            new_volunteers = {}
            for volunteer in volunteer_list:
                volunteer_id = volunteer.get('id')
                if volunteer_id:
                    # Conserver les informations existantes si le volontaire est déjà connu
                    if volunteer_id in self.volunteers:
                        volunteer['lastSeen'] = self.volunteers[volunteer_id].get('lastSeen')
                        volunteer['status'] = self.volunteers[volunteer_id].get('status')
                    else:
                        volunteer['lastSeen'] = time.time()
                        volunteer['status'] = 'available'
                    
                    new_volunteers[volunteer_id] = volunteer
            
            self.volunteers = new_volunteers
            self.last_update = time.time()
            self.update_event.set()
            logger.info(f"Liste de volontaires mise à jour: {len(self.volunteers)} volontaires")
    
    def handle_volunteer_status(self, volunteer_id, status_data):
        """
        Traite une mise à jour de statut d'un volontaire
        """
        with self.update_lock:
            if volunteer_id in self.volunteers:
                self.volunteers[volunteer_id]['status'] = status_data.get('status', 'unknown')
                self.volunteers[volunteer_id]['lastSeen'] = time.time()
                self.volunteers[volunteer_id]['resources'] = status_data.get('resources', {})
                logger.debug(f"Statut du volontaire {volunteer_id} mis à jour")
    
    def request_volunteers_list(self, timeout=10):
        """
        Demande explicitement la liste des volontaires au Coordinateur
        """
        self.update_event.clear()
        
        # Récupération d'un token valide
        token = self.auth_manager.get_valid_token()
        
        # Envoi de la demande
        request = {
            "token": token,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        
        self.mqtt_client.publish("volunteers/request", json.dumps(request))
        logger.info("Demande de liste de volontaires envoyée")
        
        # Attente de la réponse
        if self.update_event.wait(timeout):
            return True
        else:
            logger.warning("Délai dépassé pour la récupération des volontaires")
            return False
    
    def get_available_volunteers(self, min_resources=None):
        """
        Récupère la liste des volontaires disponibles avec filtre optionnel
        """
        current_time = time.time()
        
        # Si la dernière mise à jour date de plus de 30 secondes, demander une mise à jour
        if current_time - self.last_update > 30:
            self.request_volunteers_list()
        
        with self.update_lock:
            available_volunteers = []
            
            for volunteer_id, volunteer in self.volunteers.items():
                # Vérifier que le volontaire est disponible et a été vu récemment (< 2 minutes)
                if (volunteer.get('status') == 'available' and 
                    current_time - volunteer.get('lastSeen', 0) < 120):
                    
                    # Appliquer le filtre de ressources si spécifié
                    if min_resources:
                        resources = volunteer.get('resources', {})
                        meets_requirements = True
                        
                        for resource_key, min_value in min_resources.items():
                            if resources.get(resource_key, 0) < min_value:
                                meets_requirements = False
                                break
                        
                        if meets_requirements:
                            available_volunteers.append(volunteer)
                    else:
                        available_volunteers.append(volunteer)
            
            return available_volunteers