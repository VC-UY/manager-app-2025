import json
import time
import logging
import threading
import paho.mqtt.client as mqtt
from django.conf import settings

logger = logging.getLogger(__name__)

class AuthManager:
    """
    Gestionnaire d'authentification pour la communication avec le Coordinateur
    """
    
    def __init__(self, broker_url=None, manager_id=None, credentials=None):
        self.broker_url = broker_url or settings.COORDINATOR_API_URL
        self.manager_id = manager_id or f"manager-{settings.COORDINATOR_API_KEY[:8]}"
        self.credentials = credentials or {
            'api_key': settings.COORDINATOR_API_KEY,
            'api_secret': settings.COORDINATOR_API_SECRET
        }
        self.token = None
        self.token_expiry = 0
        self.mqtt_client = None
        self.token_event = threading.Event()
        
        # Paramètres de connexion MQTT
        parts = self.broker_url.split('://')
        protocol = parts[0]
        host_port = parts[1].split(':')
        self.broker_host = host_port[0]
        self.broker_port = int(host_port[1]) if len(host_port) > 1 else 1883
        
        # Initialisation du client MQTT
        self.setup_mqtt()
    
    def setup_mqtt(self):
        """
        Configure le client MQTT pour l'authentification
        """
        self.mqtt_client = mqtt.Client(f"manager-{self.manager_id}")
        
        def on_connect(client, userdata, flags, rc):
            logger.info(f"Connexion au broker MQTT avec le code {rc}")
            client.subscribe(f"authentication/response/{self.manager_id}")
        
        def on_message(client, userdata, msg):
            if msg.topic == f"authentication/response/{self.manager_id}":
                try:
                    response = json.loads(msg.payload.decode())
                    if response.get('status') == 'success':
                        self.token = response.get('token')
                        self.token_expiry = response.get('expiration', time.time() + 3600)
                        self.token_event.set()
                        logger.info("Token d'authentification reçu")
                    else:
                        logger.error(f"Authentification refusée: {response.get('message')}")
                        self.token = None
                        self.token_event.set()
                except Exception as e:
                    logger.error(f"Erreur lors du traitement de la réponse d'authentification: {e}")
        
        self.mqtt_client.on_connect = on_connect
        self.mqtt_client.on_message = on_message
    
    def authenticate(self, timeout=10):
        """
        Authentifie le Manager auprès du Coordinateur
        """
        try:
            # Connexion au broker MQTT
            self.mqtt_client.connect(self.broker_host, self.broker_port)
            self.mqtt_client.loop_start()
            
            # Préparation du message d'authentification
            auth_message = {
                "client_id": self.manager_id,
                "api_key": self.credentials['api_key'],
                "api_secret": self.credentials['api_secret'],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
            
            # Envoi de la demande d'authentification
            self.token_event.clear()
            self.mqtt_client.publish("authentication/request", json.dumps(auth_message))
            logger.info("Demande d'authentification envoyée")
            
            # Attente de la réponse
            if self.token_event.wait(timeout):
                if self.token:
                    return self.token
                else:
                    raise Exception("Authentification refusée")
            else:
                raise Exception("Délai d'authentification dépassé")
        
        except Exception as e:
            logger.error(f"Erreur d'authentification: {e}")
            raise
    
    def is_token_valid(self):
        """
        Vérifie si le token actuel est valide
        """
        if not self.token:
            return False
        
        # Vérification de l'expiration avec une marge de 60 secondes
        current_time = time.time()
        return current_time < (self.token_expiry - 60)
    
    def get_valid_token(self):
        """
        Récupère un token valide, en le renouvelant si nécessaire
        """
        if not self.is_token_valid():
            return self.authenticate()
        return self.token
    
    def stop(self):
        """
        Arrête le client MQTT
        """
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()