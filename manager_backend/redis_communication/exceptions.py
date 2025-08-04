"""
Exceptions personnalisées pour le module de communication Redis.
"""

class RedisCommError(Exception):
    """Classe de base pour les erreurs du module de communication Redis."""
    pass

class ConnectionError(RedisCommError):
    """Erreur de connexion au serveur Redis."""
    pass

class ChannelError(RedisCommError):
    """Erreur liée aux canaux (publication, abonnement, etc.)."""
    pass

class MessageError(RedisCommError):
    """Erreur liée aux messages (format, encodage, etc.)."""
    pass

class AuthenticationError(RedisCommError):
    """Erreur d'authentification."""
    pass

class NoLoginError(RedisCommError):
    """Erreur de connexion."""
    pass
