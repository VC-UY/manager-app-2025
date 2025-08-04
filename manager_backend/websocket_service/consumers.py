# websocket_service/consumers.py



"""
WebSocket Consumer pour la communication en temps réel.
"""

import json
import logging
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.contrib.auth import get_user_model
from asgiref.sync import sync_to_async
from rest_framework.authtoken.models import Token

logger = logging.getLogger(__name__)
User = get_user_model()

class WorkflowConsumer(AsyncWebsocketConsumer):
    """
    Consumer WebSocket pour les mises à jour de workflows, tâches et volontaires.
    """
    
    async def connect(self):
        """Connexion WebSocket."""
        # Authentification par token depuis les query parameters
        token_key = None
        query_string = self.scope.get('query_string', b'').decode()
        
        if query_string:
            params = dict(param.split('=') for param in query_string.split('&') if '=' in param)
            token_key = params.get('token')
        
        if token_key:
            try:
                token = await database_sync_to_async(Token.objects.get)(key=token_key)
                user = await database_sync_to_async(lambda t: t.user)(token)
                self.user = user
            except Token.DoesNotExist:
                logger.warning(f"Token WebSocket invalide: {token_key}")
                await self.close()
                return
        else:
            # Fallback sur l'utilisateur du scope
            self.user = self.scope.get("user")
            
        if not self.user or not self.user.is_authenticated:
            logger.warning("Tentative de connexion WebSocket non authentifiée")
            await self.close()
            return
        
        # Groupes par défaut pour cet utilisateur
        self.user_group = f"user_{self.user.id}"
        self.general_group = "workflow_updates"
        
        # Rejoindre les groupes
        await self.channel_layer.group_add(self.user_group, self.channel_name)
        await self.channel_layer.group_add(self.general_group, self.channel_name)
        
        await self.accept()
        
        # Envoyer confirmation de connexion
        await self.send(text_data=json.dumps({
            'type': 'connection_established',
            'user_id': str(self.user.id),
            'username': str(self.user.username),
            'message': 'Connexion WebSocket établie'
        }))
        
        logger.info(f"WebSocket connecté pour l'utilisateur {self.user.username}")
    
    async def disconnect(self, close_code):
        """Déconnexion WebSocket."""
        if hasattr(self, 'user_group'):
            await self.channel_layer.group_discard(self.user_group, self.channel_name)
        if hasattr(self, 'general_group'):
            await self.channel_layer.group_discard(self.general_group, self.channel_name)
        
        logger.info(f"WebSocket déconnecté (code: {close_code})")
    
    async def receive(self, text_data):
        """Réception de messages du client."""
        try:
            data = json.loads(text_data)
            message_type = data.get('type')
            
            # Traiter différents types de messages
            if message_type == 'subscribe_workflow':
                await self.subscribe_to_workflow(data.get('workflow_id'))
            elif message_type == 'subscribe_task':
                await self.subscribe_to_task(data.get('task_id'))
            elif message_type == 'subscribe_volunteer':
                await self.subscribe_to_volunteer(data.get('volunteer_id'))
            elif message_type == 'ping':
                await self.send(text_data=json.dumps({'type': 'pong'}))
            else:
                logger.warning(f"Type de message non reconnu: {message_type}")
                
        except json.JSONDecodeError:
            logger.error("Données JSON invalides reçues")
        except Exception as e:
            logger.error(f"Erreur lors du traitement du message: {e}")
    
    async def subscribe_to_workflow(self, workflow_id):
        """S'abonner aux mises à jour d'un workflow spécifique."""
        if workflow_id:
            group_name = f"workflow_{workflow_id}"
            await self.channel_layer.group_add(group_name, self.channel_name)
            
            await self.send(text_data=json.dumps({
                'type': 'subscription_confirmed',
                'subject': 'workflow',
                'id': workflow_id
            }))
            
            logger.info(f"Abonnement au workflow {workflow_id} pour {self.user.username}")
    
    async def subscribe_to_task(self, task_id):
        """S'abonner aux mises à jour d'une tâche spécifique."""
        if task_id:
            group_name = f"task_{task_id}"
            await self.channel_layer.group_add(group_name, self.channel_name)
            
            await self.send(text_data=json.dumps({
                'type': 'subscription_confirmed',
                'subject': 'task',
                'id': task_id
            }))
            
            logger.info(f"Abonnement à la tâche {task_id} pour {self.user.username}")
    
    async def subscribe_to_volunteer(self, volunteer_id):
        """S'abonner aux mises à jour d'un volontaire spécifique."""
        if volunteer_id:
            group_name = f"volunteer_{volunteer_id}"
            await self.channel_layer.group_add(group_name, self.channel_name)
            
            await self.send(text_data=json.dumps({
                'type': 'subscription_confirmed',
                'subject': 'volunteer',
                'id': volunteer_id
            }))
            
            logger.info(f"Abonnement au volontaire {volunteer_id} pour {self.user.username}")
    
    # Méthodes pour recevoir les mises à jour des groupes
    async def workflow_update(self, event):
        """Diffuser une mise à jour de workflow."""
        await self.send(text_data=json.dumps({
            'type': 'workflow_update',
            'workflow': event['workflow'],
            'action': event.get('action', 'updated'),
            'timestamp': event.get('timestamp')
        }))

    async def workflow_status_change(self, event):
        """Diffuser un changement de statut de workflow (handler manquant)."""
        await self.send(text_data=json.dumps({
            'type': 'workflow_status_change',
            'workflow_id': event.get('workflow_id'),
            'status': event.get('status'),
            'message': event.get('message'),
            'timestamp': event.get('timestamp')
        }))
    
    async def task_update(self, event):
        """Diffuser une mise à jour de tâche."""
        await self.send(text_data=json.dumps({
            'type': 'task_update',
            'task': event['task'],
            'action': event.get('action', 'updated'),
            'timestamp': event.get('timestamp')
        }))
    
    async def volunteer_update(self, event):
        """Diffuser une mise à jour de volontaire."""
        await self.send(text_data=json.dumps({
            'type': 'volunteer_update',
            'volunteer': event['volunteer'],
            'action': event.get('action', 'updated'),
            'timestamp': event.get('timestamp')
        }))
    
    async def task_progress(self, event):
        """Diffuser une mise à jour de progression de tâche."""
        await self.send(text_data=json.dumps({
            'type': 'task_progress',
            'task_id': event.get('task_id'),
            'volunteer': event.get('volunteer'),
            'message': event.get('message'),
            'workflow_id': event.get('workflow_id'),
            'progress': event.get('progress'),
            'timestamp': event.get('timestamp')
        }))
    
    async def task_status_change(self, event):
        """Diffuser une mise à jour de statut de tâche."""
        await self.send(text_data=json.dumps({
            'type': 'task_status_change',
            'workflow_id': event.get('workflow_id'),
            'task_id': event.get('task_id'),
            'status': event.get('status'),
            'volunteer': event.get('volunteer'),
            'message': event.get('message'),
            'timestamp': event.get('timestamp')
        }))

    async def task_status(self, event):
        """Diffuser une mise à jour de statut de tâche."""
        await self.send(text_data=json.dumps({
            'type': 'task_status_change',
            'workflow_id': event.get('workflow_id'),
            'task_id': event.get('task_id'),
            'status': event.get('status'),
            'volunteer': event.get('volunteer'),
            'message': event.get('message'),
            'timestamp': event.get('timestamp')
        }))
    
    async def volunteer_status(self, event):
        """Diffuser une mise à jour de statut de volontaire."""
        await self.send(text_data=json.dumps({
            'type': 'volunteer_status',
            'volunteer_id': event.get('volunteer_id'),
            'status': event.get('status'),
            'message': event.get('message'),
            'timestamp': event.get('timestamp')
        }))