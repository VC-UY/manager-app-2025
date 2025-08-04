"""
Client Redis universel pour la communication entre les composants du système.
"""

import json
import logging
import threading
import time
import uuid
from typing import Dict, Callable, Any, List, Optional
import redis
from django.conf import settings

from .message import Message, MessageType
from .exceptions import ChannelError, ConnectionError

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

class RedisClient:
    """
    Client Redis universel pour la communication entre les composants du système.
    Implémente le pattern Singleton pour garantir une instance unique.
    """
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls, config=None):
        """
        Récupère l'instance unique du client Redis ou en crée une nouvelle.
        
        Args:
            config: Configuration optionnelle pour surcharger les paramètres par défaut
            
        Returns:
            RedisClient: L'instance unique du client
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config)
            return cls._instance
    
    def __init__(self, config=None):
        """
        Initialise le client Redis avec la configuration fournie ou les paramètres par défaut.
        
        Args:
            config: Configuration optionnelle (host, port, db, etc.)
        """
        if RedisClient._instance is not None:
            raise RuntimeError("Utilisez RedisClient.get_instance() pour obtenir l'instance")
        
        self.config = config or {}
        self.client_type = self.config.get('client_type', 'manager')
        from redis_communication.utils import get_manager_id
        try:
            self.client_id = get_manager_id()
        except Exception as e:
            self.client_id = str(uuid.uuid4())
        
        # Paramètres de connexion
        self.host = self.config.get('host', getattr(settings, 'REDIS_PROXY_HOST', 'localhost'))
        self.port = self.config.get('port', getattr(settings, 'REDIS_PROXY_PORT', 6380))
        self.db = self.config.get('db', getattr(settings, 'REDIS_DB', 0))
        
        # Client Redis
        self.redis = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            decode_responses=True
        )
        
        # PubSub pour les abonnements
        self.pubsub = self.redis.pubsub(ignore_subscribe_messages=True)
        
        # Gestionnaires d'événements par canal
        self.handlers: Dict[str, List[Callable]] = {}
        
        # Thread d'écoute
        self.listen_thread = None
        self.running = False
        
        # Statistiques
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'last_activity': time.time(),
            'start_time': time.time()
        }
        
        logger.info(f"Client Redis initialisé: {self.client_type}:{self.client_id} @ {self.host}:{self.port}")
    
    def start(self):
        """
        Démarre le thread d'écoute des messages.
        """
        if self.running:
            logger.warning("Le client est déjà en cours d'exécution")
            return
        
        self.running = True
        self.listen_thread = threading.Thread(target=self._listen_loop)
        self.listen_thread.daemon = True
        self.listen_thread.start()
        
        logger.info(f"Client Redis démarré: {self.client_type}:{self.client_id}")
        return True
    
    def stop(self):
        """
        Arrête le thread d'écoute et ferme les connexions.
        """
        if not self.running:
            return
        
        self.running = False
        if self.listen_thread:
            self.listen_thread.join(timeout=2.0)
        
        # Désabonnement de tous les canaux
        self.pubsub.unsubscribe()
        self.pubsub.close()
        
        logger.info(f"Client Redis arrêté: {self.client_type}:{self.client_id}")
    
    def subscribe(self, channel: str, handler: Callable[[str, Any], None]):
        """
        S'abonne à un canal et associe un gestionnaire d'événements.
        
        Args:
            channel: Nom du canal
            handler: Fonction de rappel qui sera appelée avec (channel, message)
        """
        # Ajouter le gestionnaire
        if channel not in self.handlers:
            self.handlers[channel] = []
        self.handlers[channel].append(handler)
        
        # S'abonner au canal Redis
        self.pubsub.subscribe(channel)
        
        logger.info(f"Abonné au canal: {channel}")
        return True
    
    def unsubscribe(self, channel: str, handler: Optional[Callable] = None):
        """
        Se désabonne d'un canal et supprime le(s) gestionnaire(s) associé(s).
        
        Args:
            channel: Nom du canal
            handler: Gestionnaire spécifique à supprimer (si None, tous les gestionnaires sont supprimés)
        """
        if channel in self.handlers:
            if handler is None:
                # Supprimer tous les gestionnaires
                self.handlers[channel] = []
            else:
                # Supprimer un gestionnaire spécifique
                self.handlers[channel] = [h for h in self.handlers[channel] if h != handler]
            
            # Si plus aucun gestionnaire, se désabonner du canal
            if not self.handlers[channel]:
                self.pubsub.unsubscribe(channel)
                del self.handlers[channel]
                
        logger.info(f"Désabonné du canal: {channel}")
        return True
    
    def publish(self, channel: str, message_data: Any, request_id: str = None, token: str = None, message_type: str = None):
        """
        Publie un message sur un canal.
        
        Args:
            channel: Nom du canal
            message_data: Données du message
            request_id: ID de requête optionnel (généré automatiquement si non fourni)
            token: Token JWT pour l'authentification (optionnel)
            message_type: Type de message (request ou response)
            
        Returns:
            str: ID de la requête
        """
        # Déterminer le type de message en fonction du canal si non spécifié
        if message_type is None:
            if '_response' in channel:
                message_type = "response"
            else:
                message_type = "request"
                
        logger.info(f"Publication d'un message de type '{message_type}' sur le canal {channel}")
        
        # Créer un message standardisé
        message = Message(
            request_id=request_id or str(uuid.uuid4()),
            sender={
                'type': self.client_type,
                'id': self.client_id
            },
            message_type=message_type,
            data=message_data
        )
        
        # Ajouter le token JWT si fourni
        if token:
            message.token = token
            logger.info(f"Token JWT ajouté au message pour le canal {channel}")
        
        # Publier le message
        try:
            # Log pour déboguer le problème de sérialisation
            json_message = message.to_json()
            logger.info(f"Message sérialisé pour {channel}: {json_message}")
            
            # Augmenter la taille maximale du message si nécessaire
            if len(json_message) > int(self.redis.config_get('proto-max-bulk-len').get('proto-max-bulk-len', 512 * 1024)):
                self.redis.config_set('proto-max-bulk-len', len(json_message) + 9097152)
            
            self.redis.publish(channel, json_message)
            self.stats['messages_sent'] += 1
            self.stats['last_activity'] = time.time()
            
            logger.debug(f"Message publié sur {channel}: {message.request_id}")
            return message.request_id
        except Exception as e:
            logger.error(f"Erreur lors de la publication sur {channel}: {e}")
            raise ChannelError(f"Erreur de publication: {e}")
    
    def _listen_loop(self):
        """
        Boucle d'écoute des messages dans un thread séparé.
        """
        while self.running:
            try:
                message = self.pubsub.get_message(timeout=0.1)
                if message and message['type'] == 'message':
                    channel = message['channel']
                    data = message['data']
                    
                    try:
                        # Décoder le message
                        logger.warning(f"Message reçu sur le canal {channel}: {data}")
                        msg_obj = Message.from_json(data)
                        # logger.info(f"Message désérialisé sur {channel}: request_id={msg_obj.request_id}, data={msg_obj.data}")
                        
                        # Mettre à jour les statistiques
                        self.stats['messages_received'] += 1
                        self.stats['last_activity'] = time.time()
                        
                        # Appeler les gestionnaires pour ce canal
                        if channel in self.handlers:
                            logger.info(f"Gestionnaires trouvés pour le canal {channel}: {len(self.handlers[channel])}")
                            for handler in self.handlers[channel]:
                                try:
                                    logger.info(f"Appel du gestionnaire {handler.__name__ if hasattr(handler, '__name__') else 'anonyme'} pour {channel}")
                                    handler(channel, msg_obj)
                                except Exception as e:
                                    import traceback
                                    logger.error(f"Erreur dans le gestionnaire pour {channel}: {e}")
                                    logger.error(traceback.format_exc())
                        else:
                            logger.warning(f"Aucun gestionnaire pour le canal {channel}")
                    except json.JSONDecodeError as je:
                        logger.error(f"Message non JSON sur {channel}: {data}")
                        logger.error(f"Erreur de décodage JSON: {je}")
                    except Exception as e:
                        import traceback
                        logger.error(f"Erreur lors du traitement du message: {e}")
                        logger.error(traceback.format_exc())
                elif message is not None:
                    logger.error(f"Message recu inconne donc ignoré. Message: {message}")
                # Petite pause pour éviter de surcharger le CPU
                time.sleep(0.01)
                
            except redis.RedisError as e:
                logger.error(f"Erreur Redis: {e}")
                time.sleep(1.0)  # Attendre avant de réessayer
            except Exception as e:
                logger.error(f"Erreur inattendue: {e}")
                time.sleep(1.0)
    
    def get_stats(self):
        """
        Récupère les statistiques du client.
        
        Returns:
            dict: Statistiques d'utilisation
        """
        return {
            **self.stats,
            'subscribed_channels': list(self.handlers.keys()),
            'uptime': time.time() - self.stats.get('start_time', time.time())
        }
