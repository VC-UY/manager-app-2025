from rest_framework import serializers
from .models import Task
from workflows.models import Workflow

class TaskSerializer(serializers.ModelSerializer):
    """
    Sérialiseur pour le modèle Task avec les champs de base.
    """
    workflow_name = serializers.SerializerMethodField()
    
    class Meta:
        model = Task
        fields = [
            'id', 'name', 'description', 'status', 'workflow', 'workflow_name',
            'command', 'parameters', 'progress', 'created_at', 'start_time', 'end_time',
            'required_resources', 'estimated_max_time'
        ]
        read_only_fields = ['id', 'created_at', 'workflow_name']
    
    def get_workflow_name(self, obj):
        return obj.workflow.name if obj.workflow else None

class TaskDetailSerializer(serializers.ModelSerializer):
    """
    Sérialiseur détaillé pour le modèle Task avec tous les champs.
    """
    workflow_name = serializers.SerializerMethodField()
    subtasks = serializers.SerializerMethodField()
    
    class Meta:
        model = Task
        fields = '__all__'
        read_only_fields = ['id', 'created_at', 'workflow_name', 'subtasks']
    
    def get_workflow_name(self, obj):
        return obj.workflow.name if obj.workflow else None
    
    def get_subtasks(self, obj):
        subtasks = obj.subtasks.all()
        return TaskSerializer(subtasks, many=True).data
