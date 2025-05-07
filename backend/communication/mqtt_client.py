import paho.mqtt.client as mqtt
import json
import logging
import time
import threading
import ssl
import uuid
import requests
from django.conf import settings
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MQTTClient:
    """
    Client MQTT pour la communication entre le Manager, le Coordinateur et les Volontaires.
    Gère l'authentification par token et la souscription aux canaux.
    """
    
    def __init__(self):
        """Initialisation du client MQTT"""
        self.client_id = f"manager-{uuid.uuid4().hex[:8]}"
        self.client = mqtt.Client(client_id=self.client_id, protocol=mqtt.MQTTv311)

        
        # Configuration des paramètres du broker MQTT
        self.broker_host = getattr(settings, 'MQTT_BROKER_HOST', 'localhost')
        self.broker_port = getattr(settings, 'MQTT_BROKER_PORT', 1883)
        self.broker_use_tls = getattr(settings, 'MQTT_USE_TLS', False)
        
        # Configuration de l'authentification
        self.coordinator_api_url = getattr(settings, 'COORDINATOR_API_URL', 'http://coordinator-service:8000/api')
        self.auth_token = None
        self.token_expiry = None
        
        # Configuration des callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.on_subscribe = self._on_subscribe
        self.client.on_publish = self._on_publish
        
        # Initialisation du lock pour les opérations thread-safe
        self.lock = threading.Lock()
        
        # État de connexion
        self.connected = False
        self.subscribed_topics = set()
        
        # Mapping des callbacks pour les topics
        self.topic_callbacks = {}
        
        # Configuration TLS si activée
        if self.broker_use_tls:
            self.client.tls_set(
                ca_certs=getattr(settings, 'MQTT_CA_CERTS', None),
                certfile=getattr(settings, 'MQTT_CERTFILE', None),
                keyfile=getattr(settings, 'MQTT_KEYFILE', None),
                tls_version=ssl.PROTOCOL_TLS,
                ciphers=None
            )
    
    def authenticate(self):
        """
        Authentification auprès du coordinateur pour obtenir un token MQTT
        """
        try:
            # Si le token existe et est valide, on le réutilise
            if self.auth_token and self.token_expiry and datetime.now() < self.token_expiry:
                logger.debug("Utilisation du token existant")
                return True
            
            # Authentification auprès du coordinateur
            response = requests.post(
                f"{self.coordinator_api_url}/auth/mqtt",
                json={
                    "client_id": self.client_id,
                    "client_type": "manager"
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
    
    def connect(self):
        """
        Établit la connexion avec le broker MQTT
        """
        # Authentification d'abord
        if not self.authenticate():
            logger.error("Impossible de se connecter sans authentification")
            return False
        
        # Configurer les identifiants après l'authentification
        self.client.username_pw_set(self.client_id, self.auth_token)
        
        try:
            # Connexion au broker
            self.client.connect(self.broker_host, self.broker_port, 60)
            
            # Démarrer la boucle de traitement des messages en arrière-plan
            self.client.loop_start()
            
            # Attendre la connexion
            timeout = 10  # secondes
            start_time = time.time()
            while not self.connected and time.time() - start_time < timeout:
                time.sleep(0.1)
            
            if not self.connected:
                logger.error("Timeout lors de la connexion au broker MQTT")
                self.client.loop_stop()
                return False
            
            logger.info(f"Connecté au broker MQTT {self.broker_host}:{self.broker_port}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la connexion au broker MQTT: {e}")
            return False
    
    def disconnect(self):
        """
        Ferme la connexion avec le broker MQTT
        """
        try:
            self.client.disconnect()
            self.client.loop_stop()
            self.connected = False
            logger.info("Déconnecté du broker MQTT")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la déconnexion: {e}")
            return False
    
    def subscribe(self, topic, callback=None, qos=1):
        """
        S'abonne à un topic MQTT
        
        Args:
            topic (str): Le topic à écouter
            callback (callable, optional): Fonction de rappel pour traiter les messages
            qos (int, optional): Qualité de service (0, 1 ou 2)
            
        Returns:
            bool: True si l'abonnement a réussi, False sinon
        """
        with self.lock:
            if not self.connected:
                if not self.connect():
                    return False
            
            try:
                # Enregistrer le callback si fourni
                if callback:
                    self.topic_callbacks[topic] = callback
                
                # S'abonner au topic
                result, mid = self.client.subscribe(topic, qos)
                
                if result == mqtt.MQTT_ERR_SUCCESS:
                    self.subscribed_topics.add(topic)
                    logger.info(f"Abonné au topic {topic} avec QoS {qos}")
                    return True
                else:
                    logger.error(f"Échec de l'abonnement au topic {topic}: {result}")
                    return False
                    
            except Exception as e:
                logger.error(f"Erreur lors de l'abonnement au topic {topic}: {e}")
                return False
    
    def unsubscribe(self, topic):
        """
        Se désabonne d'un topic MQTT
        
        Args:
            topic (str): Le topic à quitter
            
        Returns:
            bool: True si le désabonnement a réussi, False sinon
        """
        with self.lock:
            if not self.connected:
                logger.warning("Tentative de désabonnement sans connexion active")
                return False
            
            try:
                # Se désabonner du topic
                result, mid = self.client.unsubscribe(topic)
                
                if result == mqtt.MQTT_ERR_SUCCESS:
                    self.subscribed_topics.discard(topic)
                    self.topic_callbacks.pop(topic, None)  # Supprimer le callback
                    logger.info(f"Désabonné du topic {topic}")
                    return True
                else:
                    logger.error(f"Échec du désabonnement du topic {topic}: {result}")
                    return False
                    
            except Exception as e:
                logger.error(f"Erreur lors du désabonnement du topic {topic}: {e}")
                return False
    
    def publish(self, topic, payload, qos=1, retain=False):
        """
        Publie un message sur un topic MQTT
        
        Args:
            topic (str): Le topic de publication
            payload (str/dict): Le contenu du message (sera converti en JSON si c'est un dict)
            qos (int, optional): Qualité de service (0, 1 ou 2)
            retain (bool, optional): Si True, le message est conservé par le broker
            
        Returns:
            bool: True si la publication a réussi, False sinon
        """
        with self.lock:
            if not self.connected:
                if not self.connect():
                    return False
            
            try:
                # Préparer le payload
                if isinstance(payload, dict):
                    # Ajouter des métadonnées standard
                    if 'timestamp' not in payload:
                        payload['timestamp'] = datetime.now().isoformat()
                    if 'sender' not in payload:
                        payload['sender'] = {
                            'type': 'manager',
                            'id': self.client_id
                        }
                    
                    message = json.dumps(payload)
                else:
                    message = payload
                
                # Publier le message
                result = self.client.publish(topic, message, qos, retain)
                
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    logger.info(f"Message publié sur le topic {topic}")
                    return True
                else:
                    logger.error(f"Échec de la publication sur le topic {topic}: {result.rc}")
                    return False
                    
            except Exception as e:
                logger.error(f"Erreur lors de la publication sur le topic {topic}: {e}")
                return False
    
    def publish_task_assignment(self, task, volunteer_id):
        """
        Publie l'attribution d'une tâche à un volontaire
        
        Args:
            task (Task): La tâche attribuée
            volunteer_id (str): ID du volontaire
            
        Returns:
            bool: True si la publication a réussi, False sinon
        """
        payload = {
            "task_id": str(task.id),
            "task_name": task.name,
            "command": task.command,
            "parameters": task.parameters,
            "workflow_id": str(task.workflow.id),
            "workflow_type": task.workflow.workflow_type,
            "resources": task.required_resources,
            "volunteer_id": volunteer_id,
            "docker_image": task.docker_image
        }
        
        # Ajouter les informations sur les blocs de matrices si présentes
        if None not in [task.matrix_block_row_start, task.matrix_block_row_end,
                      task.matrix_block_col_start, task.matrix_block_col_end]:
            payload["matrix_block"] = {
                "row_start": task.matrix_block_row_start,
                "row_end": task.matrix_block_row_end,
                "col_start": task.matrix_block_col_start,
                "col_end": task.matrix_block_col_end
            }
            
            # Ajouter les informations spécifiques à la multiplication si disponibles
            if hasattr(task, 'matrix_data') and task.matrix_data:
                payload["matrix_data"] = task.matrix_data
        
        # Publier sur le topic spécifique au volontaire
        return self.publish(f"tasks/assignment/{volunteer_id}", payload)
    
    def publish_workflow_status(self, workflow):
        """
        Publie l'état actuel d'un workflow
        
        Args:
            workflow (Workflow): Le workflow à publier
            
        Returns:
            bool: True si la publication a réussi, False sinon
        """
        payload = {
            "workflow_id": str(workflow.id),
            "workflow_name": workflow.name,
            "workflow_type": workflow.workflow_type,
            "status": workflow.status,
            "progress": workflow.progress if hasattr(workflow, 'progress') else 0
        }
        
        # Publier sur le topic des workflows
        return self.publish(f"workflows/status/{workflow.id}", payload)
    
    def publish_matrix_result(self, workflow, result_file):
        """
        Publie les informations sur le résultat d'un calcul matriciel
        
        Args:
            workflow (Workflow): Le workflow concerné
            result_file (str): Chemin vers le fichier de résultat
            
        Returns:
            bool: True si la publication a réussi, False sinon
        """
        payload = {
            "workflow_id": str(workflow.id),
            "workflow_type": workflow.workflow_type,
            "result_file": result_file,
            "completion_time": datetime.now().isoformat()
        }
        
        # Publier sur le topic des résultats
        return self.publish(f"workflows/results/{workflow.id}", payload)
    
    def listen_for_task_status_updates(self, callback):
        """
        Écoute les mises à jour de statut des tâches
        
        Args:
            callback (callable): Fonction de rappel pour traiter les messages
            
        Returns:
            bool: True si l'abonnement a réussi, False sinon
        """
        return self.subscribe("tasks/status/#", callback)
    
    def listen_for_task_results(self, callback):
        """
        Écoute les notifications de résultats des tâches
        
        Args:
            callback (callable): Fonction de rappel pour traiter les messages
            
        Returns:
            bool: True si l'abonnement a réussi, False sinon
        """
        return self.subscribe("tasks/results/#", callback)
    
    def listen_for_volunteer_status(self, callback):
        """
        Écoute les mises à jour de statut des volontaires
        
        Args:
            callback (callable): Fonction de rappel pour traiter les messages
            
        Returns:
            bool: True si l'abonnement a réussi, False sinon
        """
        return self.subscribe("volunteers/status/#", callback)
    
    # Callbacks internes pour la gestion de la connexion MQTT
    
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Callback appelé lors de la connexion au broker"""
        if rc == 0:
            self.connected = True
            logger.info(f"Connecté au broker MQTT avec le code {rc}")
            
            # Réabonnement aux topics précédents
            for topic in self.subscribed_topics:
                self.client.subscribe(topic)
        else:
            self.connected = False
            logger.error(f"Échec de la connexion avec le code {rc}")
    
    def _on_disconnect(self, client, userdata, rc, properties=None):
        """Callback appelé lors de la déconnexion du broker"""
        self.connected = False
        if rc == 0:
            logger.info("Déconnexion normale du broker MQTT")
        else:
            logger.warning(f"Déconnexion inattendue du broker MQTT avec le code {rc}")
    
    def _on_message(self, client, userdata, message):
        """Callback appelé à la réception d'un message"""
        try:
            topic = message.topic
            payload = message.payload.decode('utf-8')
            
            logger.debug(f"Message reçu sur le topic {topic}: {payload}")
            
            # Parser le payload JSON
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                data = payload
            
            # Rechercher un callback spécifique pour ce topic
            callback = None
            for registered_topic, cb in self.topic_callbacks.items():
                # Gestion des wildcards (#, +)
                if mqtt.topic_matches_sub(registered_topic, topic):
                    callback = cb
                    break
            
            # Exécuter le callback si trouvé
            if callback:
                callback(topic, data)
                
        except Exception as e:
            logger.error(f"Erreur lors du traitement du message: {e}")
    
    def _on_subscribe(self, client, userdata, mid, granted_qos, properties=None):
        """Callback appelé après un abonnement"""
        logger.debug(f"Abonnement confirmé avec QoS {granted_qos}")
    
    def _on_publish(self, client, userdata, mid, properties=None):
        """Callback appelé après une publication"""
        logger.debug(f"Publication confirmée, message ID: {mid}")