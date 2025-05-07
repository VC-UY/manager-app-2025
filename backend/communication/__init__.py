# Module de communication pour le Manager de Workflow
# Gère les communications entre le Manager, le Coordinateur et les Volontaires

from .mqtt_client import MQTTClient
from .coordinator_api import CoordinatorAPI
from .matrix_message_handler import MatrixMessageHandler

# Initialiser le gestionnaire de messages à l'importation du module
matrix_handler = MatrixMessageHandler()

# Fonctions utilitaires pour utiliser le gestionnaire de messages

def start_message_handler():
    """
    Démarre le gestionnaire de messages matriciels
    
    Returns:
        bool: True si le démarrage a réussi, False sinon
    """
    return matrix_handler.start()

def stop_message_handler():
    """
    Arrête le gestionnaire de messages matriciels
    
    Returns:
        bool: True si l'arrêt a réussi, False sinon
    """
    return matrix_handler.stop()

def get_matrix_handler():
    """
    Récupère l'instance du gestionnaire de messages matriciels
    
    Returns:
        MatrixMessageHandler: L'instance du gestionnaire
    """
    return matrix_handler

def get_mqtt_client():
    """
    Récupère une nouvelle instance du client MQTT
    
    Returns:
        MQTTClient: Une nouvelle instance du client MQTT
    """
    return MQTTClient()

def get_coordinator_api():
    """
    Récupère une nouvelle instance de l'API du coordinateur
    
    Returns:
        CoordinatorAPI: Une nouvelle instance de l'API du coordinateur
    """
    return CoordinatorAPI()