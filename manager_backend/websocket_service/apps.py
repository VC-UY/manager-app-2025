"""
Configuration de l'application WebSocket.
"""

from django.apps import AppConfig

class WebsocketServiceConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'websocket_service'
    verbose_name = 'Service WebSocket'
    
    def ready(self):
        """Importer les signaux lors du démarrage de l'application."""
        import websocket_service.signals