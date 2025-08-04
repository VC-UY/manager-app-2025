"""
Module client pour faciliter l'envoi de notifications WebSocket depuis les vues Django.
"""

import logging
from typing import Dict, Any, Optional, List
from .notifier import notifier

logger = logging.getLogger(__name__)

def notify_event(event_type: str, data: Dict[str, Any], groups: Optional[List[str]] = None):
    """
    Fonction helper pour envoyer des notifications WebSocket depuis n'importe où dans l'application.
    Toutes les notifications sont envoyées sur le groupe 'workflow_updates'.
    """
    try:
        notifier.notify_custom_event(event_type, data, ['workflow_updates'])
        logger.debug(f"Notification WebSocket envoyée: {event_type}")
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi de la notification WebSocket: {e}")

def notify_workflow_status_change(workflow_id: str, status: str, message: str = None):
    """Notifie un changement de statut de workflow."""
    notifier.notify_workflow_status_change(workflow_id, status, message)

def notify_task_progress_update(task_id: str, progress: float, volunteer_id: str = None):
    """Notifie une mise à jour de progression de tâche."""
    notifier.notify_task_progress(task_id, progress, volunteer_id)

def notify_volunteer_status_update(volunteer_id: str, status: str, available: bool = None):
    """Notifie un changement de statut de volontaire."""
    notifier.notify_volunteer_status_change(volunteer_id, status, available)