# views.py
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from django.db.models import Count
from .models import Volunteer, VolunteerTask
from tasks.models import Task
from workflows.models import Workflow
from .serializers import (
    VolunteerSerializer,
    VolunteerTaskSerializer,
    VolunteerDetailSerializer,
    TaskWithVolunteerCountSerializer,
    TaskSerializer,
)

class VolunteerViewSet(viewsets.ModelViewSet):
    """
    ViewSet pour les opérations CRUD sur les volontaires.
    Permet de créer, lire, mettre à jour et supprimer des volontaires.
    """
    queryset = Volunteer.objects.all()
    serializer_class = VolunteerSerializer
    permission_classes = [AllowAny]

    def get_serializer_class(self):
        if self.action == 'retrieve':
            return VolunteerDetailSerializer
        return VolunteerSerializer

    @action(detail=True, methods=['get'])
    def tasks(self, request, pk=None):
        """
        Récupérer toutes les tâches assignées à ce volontaire.
        """
        volunteer = self.get_object()
        volunteer_tasks = VolunteerTask.objects.filter(volunteer=volunteer)
        serializer = VolunteerTaskSerializer(volunteer_tasks, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['post'])
    def assign_task(self, request, pk=None):
        """
        Assigner une tâche à ce volontaire.
        """
        volunteer = self.get_object()
        task_id = request.data.get('task_id')
        
        if not task_id:
            return Response(
                {"error": "Task ID is required"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            task = Task.objects.get(id=task_id)
        except Task.DoesNotExist:
            return Response(
                {"error": "Task not found"}, 
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Vérifier si la tâche est déjà assignée à ce volontaire
        if VolunteerTask.objects.filter(task=task, volunteer=volunteer).exists():
            return Response(
                {"error": "Task already assigned to this volunteer"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Créer l'assignation
        volunteer_task = VolunteerTask.objects.create(
            task=task,
            volunteer=volunteer,
            status="ASSIGNED"
        )
        
        # Mettre à jour le statut de la tâche
        task.status = "ASSIGNED"
        task.save()
        
        return Response(
            {"message": f"Task {task.name} assigned to volunteer"}, 
            status=status.HTTP_201_CREATED
        )

    @action(detail=False, methods=['get'])
    def by_workflow(self, request):
        """
        Filtrer les volontaires par workflow.
        """
        workflow_id = request.query_params.get('workflow_id')
        if not workflow_id:
            return Response(
                {"error": "Workflow ID is required"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        volunteers = Volunteer.objects.filter(
            assigned_tasks__task__workflow__id=workflow_id
        ).distinct()
        
        serializer = self.get_serializer(volunteers, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def by_status(self, request):
        """
        Filtrer les volontaires par statut.
        """
        status_param = request.query_params.get('status')
        if not status_param:
            return Response(
                {"error": "Status parameter is required"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        volunteers = Volunteer.objects.filter(status=status_param)
        serializer = self.get_serializer(volunteers, many=True)
        
        return Response(serializer.data)

class VolunteerTaskViewSet(viewsets.ModelViewSet):
    """
    ViewSet pour les opérations CRUD sur les assignations de tâches aux volontaires.
    """
    queryset = VolunteerTask.objects.all()
    serializer_class = VolunteerTaskSerializer
    permission_classes = [AllowAny]

    @action(detail=False, methods=['get'])
    def by_task(self, request):
        """
        Filtrer les assignations par tâche.
        """
        task_id = request.query_params.get('task_id')
        if not task_id:
            return Response(
                {"error": "Task ID is required"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        volunteer_tasks = VolunteerTask.objects.filter(task__id=task_id)
        serializer = self.get_serializer(volunteer_tasks, many=True)
        
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def by_volunteer(self, request):
        """
        Filtrer les assignations par volontaire.
        """
        volunteer_id = request.query_params.get('volunteer_id')
        if not volunteer_id:
            return Response(
                {"error": "Volunteer ID is required"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        volunteer_tasks = VolunteerTask.objects.filter(volunteer__id=volunteer_id)
        serializer = self.get_serializer(volunteer_tasks, many=True)
        
        return Response(serializer.data)

    @action(detail=True, methods=['post'])
    def update_progress(self, request, pk=None):
        """
        Mettre à jour la progression d'une assignation.
        """
        volunteer_task = self.get_object()
        progress = request.data.get('progress')
        
        if progress is None:
            return Response(
                {"error": "Progress parameter is required"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            progress = float(progress)
            if progress < 0 or progress > 100:
                raise ValueError("Progress must be between 0 and 100")
        except ValueError:
            return Response(
                {"error": "Invalid progress value"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        volunteer_task.progress = progress
        
        # Si la progression est à 100%, marquer comme terminée
        if progress == 100:
            volunteer_task.status = "COMPLETED"
            volunteer_task.completed_at = timezone.now()
            
            # Mettre à jour la tâche si toutes les assignations sont terminées
            task = volunteer_task.task
            all_completed = all(
                vt.status == "COMPLETED" 
                for vt in VolunteerTask.objects.filter(task=task)
            )
            
            if all_completed:
                task.status = "COMPLETED"
                task.end_time = timezone.now()
                task.save()
        
        volunteer_task.save()
        
        return Response(
            {"message": f"Progress updated to {progress}%"}, 
            status=status.HTTP_200_OK
        )

class TaskViewSet(viewsets.ModelViewSet):
    """
    ViewSet pour les opérations CRUD sur les tâches.
    """
    queryset = Task.objects.all()
    serializer_class = TaskSerializer
    permission_classes = [AllowAny]

    @action(detail=False, methods=['get'])
    def by_workflow(self, request):
        """
        Filtrer les tâches par workflow.
        """
        workflow_id = request.query_params.get('workflow_id')
        if not workflow_id:
            return Response(
                {"error": "Workflow ID is required"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        tasks = Task.objects.filter(workflow__id=workflow_id)
        serializer = self.get_serializer(tasks, many=True)
        
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def ordered_by_volunteer_count(self, request):
        """
        Récupérer les tâches d'un workflow triées par nombre de volontaires assignés.
        """
        workflow_id = request.query_params.get('workflow_id')
        if not workflow_id:
            return Response(
                {"error": "Workflow ID is required"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        tasks = Task.objects.filter(workflow__id=workflow_id).annotate(
            volunteer_count=Count('volunteer_tasks')
        ).order_by('-volunteer_count')
        
        serializer = TaskWithVolunteerCountSerializer(tasks, many=True)
        
        return Response(serializer.data)
