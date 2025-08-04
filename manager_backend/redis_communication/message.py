"""
Classes pour la standardisation des messages échangés dans le système.
"""

import json
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

class MessageType(Enum):
    """Types de messages supportés par le système."""
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    HEARTBEAT = "heartbeat"
    ERROR = "error"

@dataclass
class Message:
    """
    Format standardisé pour tous les messages échangés dans le système.
    """
    request_id: str
    sender: Dict[str, str]
    data: Any
    message_type: str = "request"
    timestamp: float = field(default_factory=time.time)
    token: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit le message en dictionnaire."""
        result = {
            'request_id': self.request_id,
            'sender': self.sender,
            'message_type': self.message_type,
            'timestamp': self.timestamp,
            'data': self.data
        }
        
        # Ajouter le token s'il existe
        if self.token:
            result['token'] = self.token
            
        return result
    
    def to_json(self) -> str:
        """Convertit le message en chaîne JSON."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Crée un message à partir d'un dictionnaire."""
        message_type = data.get('message_type', 'request')
        token = data.get('token', None)
        
        return cls(
            request_id=data.get('request_id', ''),
            sender=data.get('sender', {}),
            message_type=message_type,
            timestamp=data.get('timestamp', time.time()),
            data=data.get('data', {}),
            token=token
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        """Crée un message à partir d'une chaîne JSON."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    @classmethod
    def create_response(cls, request: 'Message', data: Any, status: str = 'success') -> 'Message':
        """
        Crée un message de réponse à partir d'une requête.
        
        Args:
            request: Message de requête d'origine
            data: Données de la réponse
            status: Statut de la réponse (success, error, etc.)
            
        Returns:
            Message: Message de réponse
        """
        response_data = {
            'status': status,
            **data
        }
        
        return cls(
            request_id=request.request_id,
            sender=request.sender,  # Même expéditeur que la requête
            message_type=MessageType.RESPONSE,
            data=response_data
        )
