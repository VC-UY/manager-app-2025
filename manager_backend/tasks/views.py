from django.shortcuts import get_object_or_404
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import Task
from workflows.models import Workflow
from volunteers.models import Volunteer, VolunteerTask
from .serializers import TaskSerializer, TaskDetailSerializer

class TaskViewSet(viewsets.ModelViewSet):
    """
    ViewSet pour les opérations CRUD sur les tâches.
    """
    queryset = Task.objects.all()
    serializer_class = TaskSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Task.objects.filter(workflow__owner=self.request.user).order_by('-created_at')

    def get_serializer_class(self):
        if self.action == 'retrieve':
            return TaskDetailSerializer
        return TaskSerializer

    def perform_create(self, serializer):
        workflow_id = self.request.data.get('workflow')
        workflow = get_object_or_404(Workflow, id=workflow_id, owner=self.request.user)
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
        workflow = get_object_or_404(Workflow, id=workflow_id, owner=request.user)

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
        
        tasks = Task.objects.filter(status=status_param, workflow__owner=request.user)
        serializer = self.get_serializer(tasks, many=True)
        
        return Response(serializer.data)

    @action(detail=True, methods=['post'])
    def retry(self, request, pk=None):
        """
        Relance une tâche échouée (ou assignée expirée) vers un volontaire en ligne.
        """
        from tasks.models import TaskStatus
        from tasks.assignment import assign_and_publish
        from tasks.recovery import recover_pending_and_failed_work
        from volunteers.models import VolunteerTask as VT
        from volunteers.presence import get_online_volunteers_data

        task = self.get_object()
        workflow = task.workflow

        if task.status not in (TaskStatus.FAILED, TaskStatus.ASSIGNED, TaskStatus.CREATED):
            return Response(
                {"error": f"Impossible de relancer une tâche au statut {task.status}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        max_retries = workflow.retry_count or 3
        if task.status == TaskStatus.FAILED and (task.attempts or 0) >= max_retries:
            return Response(
                {
                    "error": (
                        f"Nombre max de tentatives atteint ({task.attempts}/{max_retries}). "
                        "Resoumettez le workflow entier."
                    )
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        if task.status == TaskStatus.FAILED:
            VT.objects.filter(task=task).update(status="FAILED")
            task.status = TaskStatus.CREATED
            task.progress = 0
            task.end_time = None
            details = dict(task.error_details or {})
            details.pop("attempts_counted", None)
            task.error_details = details
            task.save()
        elif task.status == TaskStatus.ASSIGNED:
            VT.objects.filter(task=task, accepted_at__isnull=True).update(status="EXPIRED")
            task.status = TaskStatus.CREATED
            task.save(update_fields=["status"])

        online = get_online_volunteers_data()
        if not online:
            return Response(
                {
                    "success": False,
                    "error": "Aucun volontaire en ligne. La tâche est prête et sera assignée dès qu'un volontaire se reconnecte.",
                    "task_id": str(task.id),
                    "task_status": task.status,
                },
                status=status.HTTP_202_ACCEPTED,
            )

        result = assign_and_publish(workflow, online)
        if result.get("assigned", 0) == 0:
            result = recover_pending_and_failed_work(online)

        task.refresh_from_db()
        return Response(
            {
                "success": True,
                "task_id": str(task.id),
                "task_status": task.status,
                "workflow_id": str(workflow.id),
                "result": result,
            }
        )
