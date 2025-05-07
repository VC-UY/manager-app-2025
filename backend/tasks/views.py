# backend/tasks/views.py
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import AllowAny

from .models import Task, TaskStatus
from .serializers import TaskSerializer, TaskDetailSerializer, MatrixTaskSerializer
from .services.task_assigner import assign_task
from .services.task_aggregator import aggregate_matrix_results

import logging

logger = logging.getLogger(__name__)

class TaskViewSet(viewsets.ModelViewSet):
    """
    API endpoint pour gérer les tâches matricielles
    """
    queryset = Task.objects.all().order_by('-created_at')
    serializer_class = TaskSerializer
    permission_classes = [AllowAny]
    
    def get_serializer_class(self):
        if self.action == 'retrieve':
            return TaskDetailSerializer
        if self.action in ['matrix_info', 'subtasks']:
            return MatrixTaskSerializer
        return TaskSerializer
    
    @action(detail=True, methods=['post'])
    def assign(self, request, pk=None):
        """Attribue la tâche à un volontaire"""
        task = self.get_object()
        
        volunteer_id = request.data.get('volunteer_id')
        
        success = assign_task(task, volunteer_id)
        
        if success:
            return Response({
                "status": "success",
                "message": f"Tâche attribuée avec succès à {task.assigned_to}"
            })
        else:
            return Response({
                "status": "error",
                "message": "Échec de l'attribution de la tâche"
            }, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=True, methods=['post'])
    def start_execution(self, request, pk=None):
        """Démarre l'exécution de la tâche"""
        task = self.get_object()
        
        if task.status != TaskStatus.ASSIGNED:
            return Response({
                "error": "La tâche doit être attribuée avant de démarrer"
            }, status=status.HTTP_400_BAD_REQUEST)
        
        task.set_status(TaskStatus.RUNNING)
        
        return Response({
            "status": "success",
            "message": "Exécution démarrée"
        })
    
    @action(detail=True, methods=['post'])
    def update_progress(self, request, pk=None):
        """Met à jour la progression de la tâche"""
        task = self.get_object()
        
        if task.status != TaskStatus.RUNNING:
            return Response({
                "error": "Seule une tâche en cours d'exécution peut être mise à jour"
            }, status=status.HTTP_400_BAD_REQUEST)
        
        progress = request.data.get('progress', 0)
        try:
            progress = float(progress)
            if not (0 <= progress <= 100):
                raise ValueError("La progression doit être entre 0 et 100")
        except (ValueError, TypeError):
            return Response({
                "error": "Valeur de progression invalide"
            }, status=status.HTTP_400_BAD_REQUEST)
        
        task.progress = progress
        task.save(update_fields=['progress'])
        
        return Response({
            "status": "success",
            "progress": progress
        })
    
    @action(detail=True, methods=['post'])
    def complete(self, request, pk=None):
        """Marque la tâche comme terminée avec des résultats"""
        task = self.get_object()
        
        if task.status not in [TaskStatus.RUNNING, TaskStatus.ASSIGNED]:
            return Response({
                "error": "Seule une tâche en cours d'exécution ou assignée peut être terminée"
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Récupérer les résultats
        results = request.data.get('results', {})
        
        task.results = results
        task.progress = 100
        task.set_status(TaskStatus.COMPLETED)
        
        return Response({
            "status": "success",
            "message": "Tâche terminée avec succès"
        })
    
    @action(detail=True, methods=['post'])
    def abort(self, request, pk=None):
        """Abandonne l'exécution de la tâche"""
        task = self.get_object()
        
        if task.status not in [TaskStatus.RUNNING, TaskStatus.ASSIGNED]:
            return Response({
                "error": "Seule une tâche en cours d'exécution ou assignée peut être abandonnée"
            }, status=status.HTTP_400_BAD_REQUEST)
        
        error_details = request.data.get('error_details', {})
        
        task.error_details = error_details
        task.set_status(TaskStatus.FAILED)
        
        return Response({
            "status": "success",
            "message": "Tâche abandonnée"
        })
    
    @action(detail=True, methods=['post'])
    def reassign(self, request, pk=None):
        """Réattribue une tâche à un autre volontaire"""
        task = self.get_object()
        
        if task.status not in [TaskStatus.ASSIGNED, TaskStatus.FAILED]:
            return Response({
                "error": "Seule une tâche attribuée ou en échec peut être réattribuée"
            }, status=status.HTTP_400_BAD_REQUEST)
        
        volunteer_id = request.data.get('volunteer_id')
        
        # Réinitialiser la tâche
        task.assigned_to = None
        task.progress = 0
        task.error_details = {}
        task.status = TaskStatus.PENDING
        task.save()
        
        # Réattribuer la tâche
        success = assign_task(task, volunteer_id)
        
        if success:
            return Response({
                "status": "success",
                "message": f"Tâche réattribuée avec succès à {task.assigned_to}"
            })
        else:
            return Response({
                "status": "error",
                "message": "Échec de la réattribution de la tâche"
            }, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=True, methods=['post'])
    def aggregate_results(self, request, pk=None):
        """Agrège les résultats des sous-tâches matricielles"""
        task = self.get_object()
        
        if task.is_subtask:
            return Response({
                "error": "L'agrégation ne peut être effectuée que sur des tâches principales"
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            result = aggregate_matrix_results(task)
            
            return Response({
                "status": "success",
                "message": "Résultats agrégés avec succès",
                "result": result
            })
        except Exception as e:
            logger.error(f"Erreur lors de l'agrégation des résultats: {str(e)}")
            return Response({
                "error": f"Erreur lors de l'agrégation des résultats: {str(e)}"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=True, methods=['get'])
    def matrix_info(self, request, pk=None):
        """Récupère les informations spécifiques à la matrice"""
        task = self.get_object()
        
        # Vérifier si c'est une tâche matricielle
        if None in [task.matrix_block_row_start, task.matrix_block_row_end,
                   task.matrix_block_col_start, task.matrix_block_col_end]:
            return Response({
                "error": "Cette tâche n'est pas une tâche matricielle"
            }, status=status.HTTP_400_BAD_REQUEST)
        
        serializer = MatrixTaskSerializer(task)
        return Response(serializer.data)
    
    @action(detail=True, methods=['get'])
    def subtasks(self, request, pk=None):
        """Récupère la liste des sous-tâches"""
        task = self.get_object()
        
        if task.is_subtask:
            return Response({
                "error": "Cette tâche est déjà une sous-tâche"
            }, status=status.HTTP_400_BAD_REQUEST)
        
        subtasks = Task.objects.filter(parent_task=task)
        
        serializer = TaskSerializer(subtasks, many=True)
        
        # Ajouter des statistiques
        stats = {
            "total": subtasks.count(),
            "completed": subtasks.filter(status=TaskStatus.COMPLETED).count(),
            "failed": subtasks.filter(status=TaskStatus.FAILED).count(),
            "running": subtasks.filter(status=TaskStatus.RUNNING).count(),
            "pending": subtasks.filter(status=TaskStatus.PENDING).count(),
            "assigned": subtasks.filter(status=TaskStatus.ASSIGNED).count()
        }
        
        return Response({
            "subtasks": serializer.data,
            "stats": stats
        })
    
    @action(detail=True, methods=['post'])
    def retry_failed_subtasks(self, request, pk=None):
        """Réessaye les sous-tâches en échec"""
        task = self.get_object()
        
        if task.is_subtask:
            return Response({
                "error": "Cette opération ne peut être effectuée que sur des tâches principales"
            }, status=status.HTTP_400_BAD_REQUEST)
        
        failed_subtasks = Task.objects.filter(parent_task=task, status=TaskStatus.FAILED)
        
        if not failed_subtasks.exists():
            return Response({
                "message": "Aucune sous-tâche en échec à réessayer"
            })
        
        # Réinitialiser les sous-tâches en échec
        count = 0
        for subtask in failed_subtasks:
            subtask.status = TaskStatus.PENDING
            subtask.progress = 0
            subtask.error_details = {}
            subtask.assigned_to = None
            subtask.results = {}
            subtask.save()
            count += 1
        
        return Response({
            "status": "success",
            "message": f"{count} sous-tâches réinitialisées pour réessai"
        })