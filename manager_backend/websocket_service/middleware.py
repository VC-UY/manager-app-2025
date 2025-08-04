# websocket_service/middleware.py


"""
Middleware d'authentification pour WebSocket.
"""

from channels.middleware import BaseMiddleware
from channels.db import database_sync_to_async
from django.contrib.auth.models import AnonymousUser
from django.contrib.auth import get_user_model
from rest_framework.authtoken.models import Token
from urllib.parse import parse_qs

User = get_user_model()

@database_sync_to_async
def get_user_from_token(token_key):
    """Récupère l'utilisateur à partir du token."""
    try:
        token = Token.objects.get(key=token_key)
        return token.user
    except Token.DoesNotExist:
        return AnonymousUser()

class TokenAuthMiddleware(BaseMiddleware):
    """Middleware pour l'authentification par token dans WebSocket."""
    
    async def __call__(self, scope, receive, send):
        # Récupérer le token depuis les query parameters
        query_string = scope.get('query_string', b'').decode()
        query_params = parse_qs(query_string)
        token_key = query_params.get('token', [None])[0]
        
        if token_key:
            scope['user'] = await get_user_from_token(token_key)
        else:
            scope['user'] = AnonymousUser()
        
        return await super().__call__(scope, receive, send)