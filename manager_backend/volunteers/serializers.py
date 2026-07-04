# serializers.py
from rest_framework import serializers
from .models import Volunteer, VolunteerTask
from tasks.models import Task
from workflows.models import Workflow

class VolunteerSerializer(serializers.ModelSerializer):
    """
    Sérialiseur de base pour le modèle Volunteer.
    """
    is_online = serializers.SerializerMethodField()

    class Meta:
        model = Volunteer
        fields = [
            'id', 'name', 'hostname', 'ip_address', 'last_ip_address',
            'cpu_cores', 'ram_mb', 'disk_gb', 'gpu', 'available', 'status',
            'last_seen', 'tags', 'is_online',
        ]

    def get_is_online(self, obj):
        from datetime import timedelta
        from django.utils import timezone
        from volunteers.presence import ONLINE_TTL_SECONDS

        if obj.status == "offline" or not obj.available:
            return False
        if not obj.last_seen:
            return False
        return obj.last_seen >= timezone.now() - timedelta(seconds=ONLINE_TTL_SECONDS)

    def to_representation(self, instance):
        data = super().to_representation(instance)
        # Statut affiché = offline si pas de heartbeat récent
        if not data.get("is_online"):
            data["status"] = "offline"
            data["available"] = False
        return data

class VolunteerDetailSerializer(serializers.ModelSerializer):
    """
    Sérialiseur détaillé pour le modèle Volunteer avec les tâches assignées.
    """
    assigned_tasks_count = serializers.SerializerMethodField()
    
    class Meta:
        model = Volunteer
        fields = '__all__'
    
    def get_assigned_tasks_count(self, obj):
        return obj.assigned_tasks.count()

class VolunteerTaskSerializer(serializers.ModelSerializer):
    """
    Sérialiseur pour le modèle VolunteerTask avec les détails du volontaire et de la tâche.
    """
    volunteer_name = serializers.SerializerMethodField()
    task_name = serializers.SerializerMethodField()
    
    class Meta:
        model = VolunteerTask
        fields = [
            'id', 'volunteer', 'volunteer_name', 'task', 'task_name',
            'assigned_at', 'started_at', 'completed_at', 'status',
            'progress', 'result', 'error'
        ]
    
    def get_volunteer_name(self, obj):
        return obj.volunteer.name if obj.volunteer else None
    
    def get_task_name(self, obj):
        return obj.task.name if obj.task else None

class TaskWithVolunteerCountSerializer(serializers.ModelSerializer):
    """
    Sérialiseur pour le modèle Task avec le nombre de volontaires assignés.
    """
    volunteer_count = serializers.SerializerMethodField()
    workflow_name = serializers.SerializerMethodField()

    class Meta:
        model = Task
        fields = ['id', 'name', 'workflow', 'workflow_name', 'status', 'volunteer_count', 'progress']

    def get_volunteer_count(self, obj):
        return obj.volunteer_tasks.count()
    
    def get_workflow_name(self, obj):
        return obj.workflow.name if obj.workflow else None

class TaskSerializer(serializers.ModelSerializer):
    """
    Sérialiseur pour le modèle Task.
    """
    class Meta:
        model = Task
        fields = '__all__'

class VolunteerTaskDetailSerializer(serializers.ModelSerializer):
    """
    Sérialiseur détaillé pour VolunteerTask avec toutes les informations de la tâche.
    Utilisé pour permettre aux volontaires de récupérer leurs tâches via HTTP.
    """
    task_id = serializers.CharField(source='task.id', read_only=True)
    task_name = serializers.CharField(source='task.name', read_only=True)
    workflow_id = serializers.CharField(source='task.workflow.id', read_only=True)
    workflow_name = serializers.CharField(source='task.workflow.name', read_only=True)
    parameters = serializers.JSONField(source='task.parameters', read_only=True)
    dependencies = serializers.SerializerMethodField()
    task_status = serializers.CharField(source='task.status', read_only=True)

    class Meta:
        model = VolunteerTask
        fields = [
            'id', 'task_id', 'task_name', 'workflow_id', 'workflow_name',
            'parameters', 'dependencies', 'status', 'task_status',
            'assigned_at', 'started_at', 'completed_at', 'progress',
            'result', 'error'
        ]

    def get_dependencies(self, obj):
        """Retourne les IDs des tâches dont cette tâche dépend."""
        if obj.task and hasattr(obj.task, 'dependencies'):
            return list(obj.task.dependencies.values_list('id', flat=True))
        return []
