# workflows/serializers.py
from rest_framework import serializers
from .models import Workflow, WorkflowStatus, WorkflowType, TaskStatus
from django.contrib.auth.models import User

class MatrixInfoSerializer(serializers.Serializer):
    """Sérialiseur pour les informations de matrices"""
    format = serializers.CharField(required=True)
    dimensions = serializers.ListField(child=serializers.IntegerField(), required=True)
    file_path = serializers.CharField(required=False, allow_null=True)
    storage_type = serializers.ChoiceField(choices=['file', 'embedded'], default='file')
    block_size = serializers.IntegerField(required=False, default=1000)

class WorkflowInputSerializer(serializers.Serializer):
    """Sérialiseur pour les entrées spécifiques aux workflows matriciels"""
    matrix_a = MatrixInfoSerializer(required=True)
    matrix_b = MatrixInfoSerializer(required=True)
    output_format = serializers.CharField(required=False, default='csv')
    algorithm = serializers.ChoiceField(
        choices=['standard', 'strassen', 'cannon', 'fox'], 
        required=False, 
        default='standard'
    )
    precision = serializers.ChoiceField(
        choices=['single', 'double', 'extended'], 
        required=False, 
        default='double'
    )
    block_size = serializers.IntegerField(required=False)

class WorkflowSerializer(serializers.ModelSerializer):
    input_data = WorkflowInputSerializer(required=False, write_only=True)
    
    class Meta:
        model = Workflow
        fields = [
            'id', 'name', 'description', 'workflow_type', 'volunteer_preferences', 
            'min_volunteers', 'max_volunteers', 'status', 'created_at', 
            'updated_at', 'submitted_at', 'completed_at', 'priority', 
            'estimated_resources', 'tags', 'metadata', 'max_execution_time', 
            'retry_count', 'input_data'
        ]
        read_only_fields = ('id', 'created_at', 'updated_at', 'owner')
    
    def create(self, validated_data):
        # Extraire les données d'entrée si présentes
        input_data = validated_data.pop('input_data', None)
        
        # Utiliser l'utilisateur par défaut pour toutes les tâches
        default_user, _ = User.objects.get_or_create(
            username='workflow_manager',
            defaults={
                'email': 'workflow_manager@system.local',
                'is_staff': True
            }
        )
        validated_data['owner'] = default_user
        
        # Créer le workflow
        workflow = super().create(validated_data)
        
        # Si des données d'entrée sont présentes, les stocker dans metadata
        if input_data:
            if not workflow.metadata:
                workflow.metadata = {}
            
            workflow.metadata['input_data'] = input_data
            workflow.save(update_fields=['metadata'])
        
        return workflow
    
    def validate(self, data):
        # Validation spécifique pour les types de workflow
        workflow_type = data.get('workflow_type')
        input_data = data.get('input_data')
        
        if workflow_type and input_data:
            # Vérifications pour les matrices
            matrix_a = input_data.get('matrix_a', {})
            matrix_b = input_data.get('matrix_b', {})
            
            # Vérifier les dimensions des matrices
            if workflow_type == WorkflowType.MATRIX_ADDITION:
                # Pour l'addition, les dimensions doivent être identiques
                if matrix_a.get('dimensions') != matrix_b.get('dimensions'):
                    raise serializers.ValidationError(
                        "Les dimensions des matrices doivent être identiques pour l'addition."
                    )
            
            elif workflow_type == WorkflowType.MATRIX_MULTIPLICATION:
                # Pour la multiplication, le nombre de colonnes de A doit être égal
                # au nombre de lignes de B
                dim_a = matrix_a.get('dimensions', [])
                dim_b = matrix_b.get('dimensions', [])
                
                if len(dim_a) < 2 or len(dim_b) < 2:
                    raise serializers.ValidationError(
                        "Les dimensions des matrices doivent être spécifiées pour la multiplication."
                    )
                
                if dim_a[1] != dim_b[0]:
                    raise serializers.ValidationError(
                        "Pour la multiplication, le nombre de colonnes de A doit être égal au nombre de lignes de B."
                    )
        
        return data

class WorkflowDetailSerializer(serializers.ModelSerializer):
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    type_display = serializers.CharField(source='get_workflow_type_display', read_only=True)
    progress = serializers.SerializerMethodField()
    matrix_dimensions = serializers.SerializerMethodField()
    remaining_time = serializers.SerializerMethodField()
    
    class Meta:
        model = Workflow
        exclude = ['owner']  # Exclure le champ owner puisqu'il est toujours le même
    
    def get_progress(self, obj):
        """Calcule la progression générale du workflow basée sur les tâches"""
        tasks = obj.tasks.all()
        if not tasks:
            return 0
        
        completed = sum(1 for task in tasks if task.status == TaskStatus.COMPLETED)
        running = sum(task.progress for task in tasks if task.status == TaskStatus.RUNNING)
        
        total_progress = completed + (running / 100) if running else completed
        return round((total_progress / len(tasks)) * 100, 2)
    
    def get_matrix_dimensions(self, obj):
        """Récupère les dimensions des matrices depuis les métadonnées"""
        if obj.metadata and 'input_data' in obj.metadata:
            input_data = obj.metadata.get('input_data', {})
            matrix_a = input_data.get('matrix_a', {})
            matrix_b = input_data.get('matrix_b', {})
            
            result = {
                'matrix_a': matrix_a.get('dimensions'),
                'matrix_b': matrix_b.get('dimensions')
            }
            
            # Calcul des dimensions de la matrice résultante
            if obj.workflow_type == WorkflowType.MATRIX_ADDITION:
                result['result'] = matrix_a.get('dimensions')
            elif obj.workflow_type == WorkflowType.MATRIX_MULTIPLICATION:
                dim_a = matrix_a.get('dimensions', [])
                dim_b = matrix_b.get('dimensions', [])
                if len(dim_a) >= 2 and len(dim_b) >= 2:
                    result['result'] = [dim_a[0], dim_b[1]]
            
            return result
        return None
    
    def get_remaining_time(self, obj):
        """Estimation du temps restant basée sur la progression et le temps écoulé"""
        if obj.status != WorkflowStatus.RUNNING or not obj.submitted_at:
            return None
            
        import datetime
        
        now = datetime.datetime.now(datetime.timezone.utc)
        elapsed = (now - obj.submitted_at).total_seconds()
        progress = self.get_progress(obj) / 100
        
        if progress <= 0:
            return None
            
        estimated_total = elapsed / progress
        remaining_seconds = estimated_total - elapsed
        
        if remaining_seconds <= 0:
            return "Finalisation en cours"
            
        # Formatter le temps restant
        hours, remainder = divmod(remaining_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{int(hours)}h {int(minutes)}m"
        elif minutes > 0:
            return f"{int(minutes)}m {int(seconds)}s"
        else:
            return f"{int(seconds)}s"