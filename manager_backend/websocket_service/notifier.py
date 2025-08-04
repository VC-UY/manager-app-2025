"""
Service de notification WebSocket pour diffuser les mises à jour.
"""

import logging
import time
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class WebSocketNotifier:
    """Service pour envoyer des notifications WebSocket."""
    
    def __init__(self):
        self.channel_layer = get_channel_layer()
    
    def _send_to_group(self, group_name: str, message: Dict[str, Any]):
        """Envoie un message à un groupe WebSocket."""
        if not self.channel_layer:
            logger.warning("Channel layer non disponible")
            return False
        
        try:
            async_to_sync(self.channel_layer.group_send)(group_name, message)
            logger.debug(f"Message envoyé au groupe {group_name}: {message.get('type', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'envoi au groupe {group_name}: {e}")
            return False
    
    # Notifications pour les workflows
    def notify_workflow_created(self, workflow_data: Dict[str, Any]):
        """Notifie la création d'un workflow."""
        message = {
            'type': 'workflow_update',
            'workflow': workflow_data,
            'action': 'created',
            'timestamp': time.time()
        }
        
        self._send_to_group('workflow_updates', message)
        logger.info(f"Notification WebSocket: workflow {workflow_data.get('id')} créé")
    
    def notify_workflow_updated(self, workflow_data: Dict[str, Any]):
        """Notifie la mise à jour d'un workflow."""
        message = {
            'type': 'workflow_update',
            'workflow': workflow_data,
            'action': 'updated',
            'timestamp': time.time()
        }
        
        self._send_to_group('workflow_updates', message)
        self._send_to_group(f"workflow_{workflow_data.get('id')}", message)
        logger.info(f"Notification WebSocket: workflow {workflow_data.get('id')} mis à jour")
    
    def notify_workflow_submitted(self, workflow_data: Dict[str, Any]):
        """Notifie la soumission d'un workflow."""
        message = {
            'type': 'workflow_update',
            'workflow': workflow_data,
            'action': 'submitted',
            'timestamp': time.time()
        }
        
        self._send_to_group('workflow_updates', message)
        self._send_to_group(f"workflow_{workflow_data.get('id')}", message)
        logger.info(f"Notification WebSocket: workflow {workflow_data.get('id')} soumis")
    
    def notify_workflow_status_change(self, workflow_id: str, status: str, message_text: str = None):
        """Notifie un changement de statut de workflow."""
        message = {
            'type': 'workflow_status_change',
            'workflow_id': workflow_id,
            'status': status,
            'message': message_text or f'Statut changé vers {status}',
            'timestamp': time.time()
        }
        
        self._send_to_group('workflow_updates', message)
        self._send_to_group(f"workflow_{workflow_id}", message)
        logger.info(f"Notification WebSocket: workflow {workflow_id} statut -> {status}")
    
    # Notifications pour les tâches
    def notify_task_created(self, task_data: Dict[str, Any]):
        """Notifie la création d'une tâche."""
        message = {
            'type': 'task_update',
            'task': task_data,
            'action': 'created',
            'timestamp': time.time()
        }
        
        self._send_to_group('workflow_updates', message)
        self._send_to_group(f"task_{task_data.get('id')}", message)
        
        # Notifier aussi le workflow parent
        if task_data.get('workflow_id'):
            self._send_to_group(f"workflow_{task_data.get('workflow_id')}", message)
        
        logger.info(f"Notification WebSocket: tâche {task_data.get('id')} créée")
    
    def notify_task_updated(self, task_data: Dict[str, Any]):
        """Notifie la mise à jour d'une tâche."""
        message = {
            'type': 'task_update',
            'task': task_data,
            'action': 'updated',
            'timestamp': time.time()
        }
        
        self._send_to_group('workflow_updates', message)
        self._send_to_group(f"task_{task_data.get('id')}", message)
        
        # Notifier aussi le workflow parent
        if task_data.get('workflow_id'):
            self._send_to_group(f"workflow_{task_data.get('workflow_id')}", message)
        
        logger.info(f"Notification WebSocket: tâche {task_data.get('id')} mise à jour")

    def notify_task_assigned(self, task_data: Dict[str, Any], volunteer_id: Optional[str] = None):
        """Notifie l'assignation d'une tâche."""
        message = {
            'type': 'task_update',
            'task': task_data,
            'action': 'assigned',
            'volunteer_id': volunteer_id,
            'timestamp': time.time()
        }
        
        self._send_to_group('workflow_updates', message)
        self._send_to_group(f"task_{task_data.get('id')}", message)
        
        if volunteer_id:
            self._send_to_group(f"volunteer_{volunteer_id}", message)
        
        if task_data.get('workflow_id'):
            self._send_to_group(f"workflow_{task_data.get('workflow_id')}", message)
        
        logger.info(f"Notification WebSocket: tâche {task_data.get('id')} assignée au volontaire {volunteer_id}")
    
    def notify_task_progress(self, task_id: str, progress: float, volunteer_id: Optional[str] = None):
        """Notifie la progression d'une tâche."""
        message = {
            'type': 'task_progress',
            'task_id': task_id,
            'volunteer_id': volunteer_id,
            'progress': progress,
            'timestamp': time.time()
        }
        
        self._send_to_group('workflow_updates', message)
        self._send_to_group(f"task_{task_id}", message)
        
        if volunteer_id:
            self._send_to_group(f"volunteer_{volunteer_id}", message)
        
        logger.info(f"Notification WebSocket: progression tâche {task_id} -> {progress}%")
    
    def notify_task_completed(self, task_data: Dict[str, Any], volunteer_id: Optional[str] = None):
        """Notifie la completion d'une tâche."""
        message = {
            'type': 'task_update',
            'task': task_data,
            'action': 'completed',
            'volunteer_id': volunteer_id,
            'timestamp': time.time()
        }
        
        self._send_to_group('workflow_updates', message)
        self._send_to_group(f"task_{task_data.get('id')}", message)
        
        if volunteer_id:
            self._send_to_group(f"volunteer_{volunteer_id}", message)
        
        if task_data.get('workflow_id'):
            self._send_to_group(f"workflow_{task_data.get('workflow_id')}", message)
        
        logger.info(f"Notification WebSocket: tâche {task_data.get('id')} complétée")
    
    def notify_task_failed(self, task_data: Dict[str, Any], error_message: str, volunteer_id: Optional[str] = None):
        """Notifie l'échec d'une tâche."""
        message = {
            'type': 'task_update',
            'task': {**task_data, 'error': error_message},
            'action': 'failed',
            'volunteer_id': volunteer_id,
            'timestamp': time.time()
        }
        
        self._send_to_group('workflow_updates', message)
        self._send_to_group(f"task_{task_data.get('id')}", message)
        
        if volunteer_id:
            self._send_to_group(f"volunteer_{volunteer_id}", message)
        
        if task_data.get('workflow_id'):
            self._send_to_group(f"workflow_{task_data.get('workflow_id')}", message)
        
        logger.info(f"Notification WebSocket: tâche {task_data.get('id')} échouée")
    
    def notify_task_started(self, task_data: Dict[str, Any], volunteer_id: Optional[str] = None):
        """Notifie le démarrage d'une tâche."""
        message = {
            'type': 'task_update',
            'task': task_data,
            'action': 'started',
            'volunteer_id': volunteer_id,
            'timestamp': time.time()
        }
        
        self._send_to_group('workflow_updates', message)
        self._send_to_group(f"task_{task_data.get('id')}", message)
        
        if volunteer_id:
            self._send_to_group(f"volunteer_{volunteer_id}", message)
        
        if task_data.get('workflow_id'):
            self._send_to_group(f"workflow_{task_data.get('workflow_id')}", message)
        
        logger.info(f"Notification WebSocket: tâche {task_data.get('id')} démarrée")
    
    # Notifications pour les volontaires
    def notify_volunteer_registered(self, volunteer_data: Dict[str, Any]):
        """Notifie l'enregistrement d'un volontaire."""
        message = {
            'type': 'volunteer_update',
            'volunteer': volunteer_data,
            'action': 'registered',
            'timestamp': time.time()
        }
        
        self._send_to_group('workflow_updates', message)
        self._send_to_group(f"volunteer_{volunteer_data.get('id')}", message)
        logger.info(f"Notification WebSocket: volontaire {volunteer_data.get('id')} enregistré")
    
    def notify_volunteer_status_change(self, volunteer_id: str, status: str, available: bool = None):
        """Notifie un changement de statut de volontaire."""
        message = {
            'type': 'volunteer_status',
            'volunteer_id': volunteer_id,
            'status': status,
            'available': available,
            'timestamp': time.time()
        }
        
        self._send_to_group('workflow_updates', message)
        self._send_to_group(f"volunteer_{volunteer_id}", message)
        logger.info(f"Notification WebSocket: volontaire {volunteer_id} statut -> {status}")
    
    def notify_volunteer_availability_change(self, volunteer_id: str, available: bool):
        """Notifie un changement de disponibilité de volontaire."""
        message = {
            'type': 'volunteer_status',
            'volunteer_id': volunteer_id,
            'available': available,
            'timestamp': time.time()
        }
        
        self._send_to_group('workflow_updates', message)
        self._send_to_group(f"volunteer_{volunteer_id}", message)
        logger.info(f"Notification WebSocket: volontaire {volunteer_id} disponibilité -> {available}")
    
    def notify_volunteer_updated(self, volunteer_data: Dict[str, Any]):
        """Notifie la mise à jour d'un volontaire."""
        message = {
            'type': 'volunteer_update',
            'volunteer': volunteer_data,
            'action': 'updated',
            'timestamp': time.time()
        }
        
        self._send_to_group('workflow_updates', message)
        self._send_to_group(f"volunteer_{volunteer_data.get('id')}", message)
        logger.info(f"Notification WebSocket: volontaire {volunteer_data.get('id')} mis à jour")
    
    # Notifications génériques
    def notify_custom_event(self, event_type: str, data: Dict[str, Any], groups: list = None):
        """Notifie un événement personnalisé (toujours sur 'workflow_updates')."""
        # On aplatit le contenu pour que le frontend reçoive les clés attendues directement
        message = {
            'type': event_type,
            **data,
            'timestamp': time.time()
        }
        self._send_to_group('workflow_updates', message)
        logger.info(f"Notification WebSocket: événement personnalisé {event_type}")

# Instance globale du notifier
notifier = WebSocketNotifier()

# Fonction helper pour notifier les événements depuis les vues
def notify_event(event_type: str, data: Dict[str, Any], groups: list = None):
    """Fonction helper pour notifier des événements depuis n'importe où dans l'application."""
    notifier.notify_custom_event(event_type, data, groups)