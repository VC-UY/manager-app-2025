from django.shortcuts import get_object_or_404
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from .models import Task
from workflows.models import Workflow
from volunteers.models import Volunteer, VolunteerTask
from .serializers import TaskSerializer, TaskDetailSerializer

class TaskViewSet(viewsets.ModelViewSet):
    """
    ViewSet pour les opérations CRUD sur les tâches.
    Permet de créer, lire, mettre à jour et supprimer des tâches.
    """
    queryset = Task.objects.all()
    serializer_class = TaskSerializer
    permission_classes = [AllowAny]

    def get_serializer_class(self):
        if self.action == 'retrieve':
            return TaskDetailSerializer
        return TaskSerializer

    def perform_create(self, serializer):
        # Récupérer le workflow associé
        workflow_id = self.request.data.get('workflow')
        workflow = get_object_or_404(Workflow, id=workflow_id)
        serializer.save(workflow=workflow)

    @action(detail=True, methods=['post'])
    def assign(self, request, pk=None):
        """
        Assigner une tâche à un volontaire spécifique.
        """
        task = self.get_object()
        volunteer_id = request.data.get('volunteer_id')
        
        if not volunteer_id:
            return Response(
                {"error": "Volunteer ID is required"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            volunteer = Volunteer.objects.get(id=volunteer_id)
        except Volunteer.DoesNotExist:
            return Response(
                {"error": "Volunteer not found"}, 
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
            {"message": f"Task assigned to volunteer {volunteer.name}"}, 
            status=status.HTTP_201_CREATED
        )

    @action(detail=True, methods=['get'])
    def volunteers(self, request, pk=None):
        """
        Récupérer tous les volontaires assignés à cette tâche.
        """
        task = self.get_object()
        volunteer_tasks = VolunteerTask.objects.filter(task=task)
        volunteers = [vt.volunteer for vt in volunteer_tasks]
        
        from volunteers.serializers import VolunteerSerializer
        serializer = VolunteerSerializer(volunteers, many=True)
        
        return Response(serializer.data)

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

        # Vérifier si le workflow existe
        workflow = Workflow.objects.filter(id=workflow_id).first()
        if not workflow:
            return Response(
                {"error": "Workflow not found"}, 
                status=status.HTTP_404_NOT_FOUND
            )

        # Récupérer les tâches associées
        tasks = Task.objects.filter(workflow=workflow)
        serializer = self.get_serializer(tasks, many=True)

        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def by_status(self, request):
        """
        Filtrer les tâches par statut.
        """
        status_param = request.query_params.get('status')
        if not status_param:
            return Response(
                {"error": "Status parameter is required"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        tasks = Task.objects.filter(status=status_param)
        serializer = self.get_serializer(tasks, many=True)
        
        return Response(serializer.data)
