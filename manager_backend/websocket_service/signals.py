# websocket_service/signals.py




"""
Signaux Django pour déclencher les notifications WebSocket.
"""

import logging
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from workflows.models import Workflow
from tasks.models import Task
from volunteers.models import Volunteer, VolunteerTask
from .notifier import notifier

logger = logging.getLogger(__name__)

@receiver(post_save, sender=Workflow)
def workflow_saved(sender, instance, created, **kwargs):
    """Signal déclenché lors de la sauvegarde d'un workflow."""
    try:
        workflow_data = {
            'id': str(instance.id),
            'name': instance.name,
            'status': instance.status,
            'workflow_type': instance.workflow_type,
            'owner': instance.owner.username if instance.owner else None,
            'created_at': instance.created_at.isoformat() if instance.created_at else None,
            'updated_at': instance.updated_at.isoformat() if instance.updated_at else None,
        }
        
        if created:
            notifier.notify_workflow_created(workflow_data)
            logger.info(f"Notification WebSocket: workflow {instance.id} créé")
        else:
            notifier.notify_workflow_updated(workflow_data)
            logger.info(f"Notification WebSocket: workflow {instance.id} mis à jour")
            
    except Exception as e:
        logger.error(f"Erreur lors de la notification workflow: {e}")

@receiver(post_save, sender=Task)
def task_saved(sender, instance, created, **kwargs):
    """Signal déclenché lors de la sauvegarde d'une tâche."""
    try:
        task_data = {
            'id': str(instance.id),
            'name': instance.name,
            'status': instance.status,
            'workflow_id': str(instance.workflow.id) if instance.workflow else None,
            'progress': instance.progress,
            'created_at': instance.created_at.isoformat() if instance.created_at else None,
            'start_time': instance.start_time.isoformat() if instance.start_time else None,
            'end_time': instance.end_time.isoformat() if instance.end_time else None,
        }
        
        if created:
            notifier.notify_task_created(task_data)
            logger.info(f"Notification WebSocket: tâche {instance.id} créée")
        else:
            # Vérifier si c'est un changement de statut important
            if hasattr(instance, '_previous_status'):
                if instance._previous_status != instance.status:
                    if instance.status == 'COMPLETED':
                        notifier.notify_task_completed(task_data)
                    elif instance.status == 'FAILED':
                        notifier.notify_task_failed(task_data, "Tâche échouée")
                    else:
                        notifier.notify_task_updated(task_data)
            else:
                notifier.notify_task_updated(task_data)
            logger.info(f"Notification WebSocket: tâche {instance.id} mise à jour")
            
    except Exception as e:
        logger.error(f"Erreur lors de la notification tâche: {e}")

@receiver(post_save, sender=Volunteer)
def volunteer_saved(sender, instance, created, **kwargs):
    """Signal déclenché lors de la sauvegarde d'un volontaire."""
    try:
        volunteer_data = {
            'id': str(instance.id),
            'name': instance.name,
            'hostname': instance.hostname,
            'ip_address': instance.ip_address,
            'status': instance.status,
            'available': instance.available,
            'cpu_cores': instance.cpu_cores,
            'ram_mb': instance.ram_mb,
            'last_seen': instance.last_seen.isoformat() if instance.last_seen else None,
        }
        
        if created:
            notifier.notify_volunteer_registered(volunteer_data)
            logger.info(f"Notification WebSocket: volontaire {instance.id} enregistré")
        else:
            # Vérifier les changements de statut ou de disponibilité
            notifier.notify_volunteer_status_change(
                str(instance.id), 
                instance.status, 
                instance.available
            )
            logger.info(f"Notification WebSocket: volontaire {instance.id} mis à jour")
            
    except Exception as e:
        logger.error(f"Erreur lors de la notification volontaire: {e}")

@receiver(post_save, sender=VolunteerTask)
def volunteer_task_saved(sender, instance, created, **kwargs):
    """Signal déclenché lors de la sauvegarde d'une assignation."""
    try:
        if created:
            # Nouvelle assignation
            task_data = {
                'id': str(instance.task.id),
                'name': instance.task.name,
                'status': instance.task.status,
                'workflow_id': str(instance.task.workflow.id) if instance.task.workflow else None,
                'progress': instance.progress,
            }
            notifier.notify_task_assigned(task_data, str(instance.volunteer.id))
            logger.info(f"Notification WebSocket: tâche {instance.task.id} assignée au volontaire {instance.volunteer.id}")
        else:
            # Mise à jour de progression
            if hasattr(instance, '_previous_progress') and instance._previous_progress != instance.progress:
                notifier.notify_task_progress(
                    str(instance.task.id), 
                    instance.progress, 
                    str(instance.volunteer.id)
                )
                logger.info(f"Notification WebSocket: progression tâche {instance.task.id} mise à jour à {instance.progress}%")
            
    except Exception as e:
        logger.error(f"Erreur lors de la notification assignation: {e}")

# Fonction pour sauvegarder l'état précédent avant modification
def save_previous_state(sender, instance, **kwargs):
    """Sauvegarde l'état précédent pour détecter les changements."""
    try:
        if instance.pk:
            old_instance = sender.objects.get(pk=instance.pk)
            if sender == Task:
                instance._previous_status = old_instance.status
            elif sender == VolunteerTask:
                instance._previous_progress = old_instance.progress
    except sender.DoesNotExist:
        pass

# Connecter les signaux pre_save pour sauvegarder l'état précédent
from django.db.models.signals import pre_save
pre_save.connect(save_previous_state, sender=Task)
pre_save.connect(save_previous_state, sender=VolunteerTask)