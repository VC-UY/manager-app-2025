import json
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

from workflows.models import Workflow
from tasks.models import Task

# Obtenir la couche de canaux pour les WebSockets
channel_layer = get_channel_layer()

@receiver(post_save, sender=Workflow)
def workflow_saved(sender, instance, created, **kwargs):
    """Signal émis lorsqu'un workflow est sauvegardé."""
    # Préparer les données à envoyer
    workflow_data = {
        'id': str(instance.id),
        'name': instance.name,
        'status': instance.status,
        'updated_at': instance.updated_at.isoformat() if instance.updated_at else None
    }
    
    # Envoyer la mise à jour à tous les clients connectés
    async_to_sync(channel_layer.group_send)(
        'workflow_updates',
        {
            'type': 'workflow_update',
            'workflow_id': str(instance.id),
            'created': created,
            'status': workflow_data
        }
    )

@receiver(post_save, sender=Task)
def task_saved(sender, instance, created, **kwargs):
    """Signal émis lorsqu'une tâche est sauvegardée."""
    # Préparer les données à envoyer
    task_data = {
        'id': str(instance.id),
        'name': instance.name,
        'status': instance.status,
        'progress': instance.progress,
        'assigned_to': instance.assigned_to,
        'workflow_id': str(instance.workflow.id) if instance.workflow else None
    }
    
    # Envoyer la mise à jour à tous les clients connectés
    async_to_sync(channel_layer.group_send)(
        'workflow_updates',
        {
            'type': 'task_update',
            'task_id': str(instance.id),
            'created': created,
            'status': task_data
        }
    )
    
    # Si le statut de la tâche change, mettre à jour le statut du workflow
    if not created and 'status' in kwargs.get('update_fields', []):
        instance.workflow.save(update_fields=['updated_at'])

# Fonction pour diffuser une mise à jour du Coordinateur
def broadcast_coordinator_status(connected=True, authenticated=False, token_expiry=None):
    """Diffuse le statut du Coordinateur à tous les clients connectés."""
    try:
        async_to_sync(channel_layer.group_send)(
            'workflow_updates',
            {
                'type': 'coordinator_update',
                'status': {
                    'connected': connected,
                    'authenticated': authenticated,
                    'token_expiry': token_expiry
                }
            }
        )
    except Exception as e:
        print(f"Erreur lors de la diffusion du statut du Coordinateur: {e}")

# Fonction pour diffuser une mise à jour des volontaires
def broadcast_volunteers_update(volunteers):
    """Diffuse la liste des volontaires à tous les clients connectés."""
    try:
        async_to_sync(channel_layer.group_send)(
            'workflow_updates',
            {
                'type': 'volunteers_update',
                'count': len(volunteers),
                'volunteers': volunteers
            }
        )
    except Exception as e:
        print(f"Erreur lors de la diffusion de la liste des volontaires: {e}")