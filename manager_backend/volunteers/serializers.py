# serializers.py
from rest_framework import serializers
from .models import Volunteer, VolunteerTask
from tasks.models import Task
from workflows.models import Workflow

class VolunteerSerializer(serializers.ModelSerializer):
    """
    Sérialiseur de base pour le modèle Volunteer.
    """
    class Meta:
        model = Volunteer
        fields = [
            'id', 'name', 'hostname', 'ip_address', 'last_ip_address',
            'cpu_cores', 'ram_mb', 'disk_gb', 'gpu', 'available', 'status',
            'last_seen', 'tags'
        ]

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
