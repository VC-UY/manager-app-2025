# backend/results/views.py
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from utils.mongodb import get_collection
from bson.objectid import ObjectId
from datetime import datetime
import json
import logging
from .result_aggregator import ResultAggregator
from communication.coordinator_client import CoordinatorClient

logger = logging.getLogger(__name__)

class TaskResultView(APIView):
    """API pour soumettre les résultats d'une sous-tâche"""
    permission_classes = []  # Permettre l'accès sans authentification pour les volontaires
    
    def post(self, request, subtask_id):
        tasks, client = get_collection('tasks')
        subtask = tasks.find_one({'_id': ObjectId(subtask_id)})
        
        if not subtask:
            client.close()
            return Response({"error": "Subtask not found"}, status=status.HTTP_404_NOT_FOUND)
        
        # Vérifier que la sous-tâche est bien assignée et en cours d'exécution
        if subtask.get('status') not in ['ASSIGNED', 'RUNNING']:
            client.close()
            return Response({"error": "Subtask not in a valid state for result submission"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Récupérer les données du résultat
        result_data = request.data.get('result', {})
        success = request.data.get('success', False)
        error_message = request.data.get('error_message', '')
        
        # Mettre à jour la sous-tâche avec le résultat
        if success:
            update_data = {
                'status': 'COMPLETED',
                'results': result_data,
                'completed_at': datetime.now(),
                'updated_at': datetime.now(),
                'progress': 100
            }
        else:
            update_data = {
                'status': 'FAILED',
                'error_details': {
                    'message': error_message,
                    'timestamp': datetime.now()
                },
                'updated_at': datetime.now(),
                'progress': 0
            }
        
        tasks.update_one(
            {'_id': ObjectId(subtask_id)},
            {'$set': update_data}
        )
        
        # Notifier le changement de statut
        parent_task_id = subtask.get('parent_task_id')
        workflow_id = subtask.get('workflow_id')
        
        coordinator = CoordinatorClient()
        coordinator.broadcast_task_update(
            task_id=subtask_id,
            workflow_id=workflow_id,
            status=update_data['status'],
            progress=update_data.get('progress', 0),
            message=f"Subtask {update_data['status'].lower()}"
        )
        
        # Vérifier si toutes les sous-tâches sont terminées et agréger les résultats
        if parent_task_id:
            ResultAggregator.check_task_completion(parent_task_id)
        
        client.close()
        return Response({"message": "Result submitted successfully"})

class TaskAggregationView(APIView):
    """API pour déclencher l'agrégation des résultats d'une tâche"""
    permission_classes = [IsAuthenticated]
    
    def post(self, request, task_id):
        success = ResultAggregator.aggregate_results(task_id)
        
        if success:
            return Response({"message": "Results aggregated successfully"})
        else:
            return Response({"error": "Failed to aggregate results"}, status=status.HTTP_400_BAD_REQUEST)

class WorkflowResultsView(APIView):
    """API pour récupérer les résultats agrégés d'un workflow"""
    permission_classes = [IsAuthenticated]
    
    def get(self, request, workflow_id):
        workflows, client = get_collection('workflows')
        workflow = workflows.find_one({
            '_id': ObjectId(workflow_id),
            'owner': str(request.user.id)
        })
        
        if not workflow:
            client.close()
            return Response({"error": "Workflow not found"}, status=status.HTTP_404_NOT_FOUND)
        
        # Récupérer toutes les tâches du workflow avec leurs résultats
        tasks, _ = get_collection('tasks')
        workflow_tasks = list(tasks.find({
            'workflow_id': workflow_id,
            'is_subtask': False  # Seulement les tâches principales
        }))
        
        results = []
        for task in workflow_tasks:
            if task.get('status') == 'COMPLETED':
                task_result = {
                    'task_id': str(task['_id']),
                    'task_name': task.get('name', ''),
                    'completed_at': task.get('completed_at'),
                    'results': task.get('results', {})
                }
                results.append(task_result)
        
        client.close()
        
        return Response({
            'workflow_id': workflow_id,
            'workflow_name': workflow.get('name', ''),
            'status': workflow.get('status', 'UNKNOWN'),
            'task_results': results
        })