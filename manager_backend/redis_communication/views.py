"""
Vues Django pour l'application redis_communication.
Expose des API pour interagir avec le client Redis.
"""

from django.http import JsonResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status

from .client import RedisClient

@api_view(['GET'])
def get_stats(request):
    """
    Récupère les statistiques du client Redis.
    """
    client = RedisClient.get_instance()
    stats = client.get_stats()
    
    return Response(stats)

@api_view(['GET'])
def get_channels(request):
    """
    Récupère la liste des canaux auxquels le client est abonné.
    """
    client = RedisClient.get_instance()
    channels = list(client.handlers.keys())
    
    return Response({
        'channels': channels,
        'count': len(channels)
    })

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def publish_message(request):
    """
    Publie un message sur un canal.
    
    Requiert:
    - channel: Nom du canal
    - message: Données du message
    - request_id: (optionnel) ID de la requête
    """
    channel = request.data.get('channel')
    message = request.data.get('message')
    request_id = request.data.get('request_id')
    
    if not channel or not message:
        return Response({
            'error': 'Les paramètres channel et message sont requis'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        client = RedisClient.get_instance()
        req_id = client.publish(channel, message, request_id)
        
        return Response({
            'success': True,
            'request_id': req_id
        })
    except Exception as e:
        return Response({
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
