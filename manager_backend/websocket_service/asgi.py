# manager_backend/asgi.py

"""
Configuration ASGI pour manager_backend.
Support des WebSockets et HTTP.
"""

import os
import django
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.security.websocket import AllowedHostsOriginValidator

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'manager_backend.settings')
django.setup()

# Import des routes WebSocket et middleware
from websocket_service.routing import websocket_urlpatterns
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("ASGI: Initialisation de l'application Django")
from websocket_service.middleware import TokenAuthMiddleware

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AllowedHostsOriginValidator(
        TokenAuthMiddleware(
            URLRouter(websocket_urlpatterns)
        )
    ),
})

