#backend/tasks/serializers.py

from rest_framework import serializers
from .models import Task, TaskStatus
from workflows.serializers import WorkflowSerializer

class TaskSerializer(serializers.ModelSerializer):
    workflow_name = serializers.CharField(source='workflow.name', read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    workflow_type = serializers.CharField(source='workflow.workflow_type', read_only=True)
    matrix_info = serializers.SerializerMethodField()
    
    class Meta:
        model = Task
        fields = [
            'id', 'workflow', 'workflow_name', 'workflow_type', 'name', 
            'description', 'command', 'parameters', 'dependencies', 
            'status', 'status_display', 'is_subtask', 'progress', 
            'created_at', 'start_time', 'end_time', 'required_resources', 
            'assigned_to', 'attempts', 'matrix_block_row_start', 
            'matrix_block_row_end', 'matrix_block_col_start', 
            'matrix_block_col_end', 'matrix_info', 'volunteer_preference',
            'parent_task'
        ]
        read_only_fields = ('id', 'created_at')
    
    def get_matrix_info(self, obj):
        """Renvoie des informations sur les blocs matriciels si applicable"""
        if None not in [obj.matrix_block_row_start, obj.matrix_block_row_end, 
                       obj.matrix_block_col_start, obj.matrix_block_col_end]:
            
            # Calcul de la taille du bloc
            rows = obj.matrix_block_row_end - obj.matrix_block_row_start + 1
            cols = obj.matrix_block_col_end - obj.matrix_block_col_start + 1
            
            return {
                'block_position': f"[{obj.matrix_block_row_start}:{obj.matrix_block_row_end}, {obj.matrix_block_col_start}:{obj.matrix_block_col_end}]",
                'block_dimensions': [rows, cols],
                'elements_count': rows * cols
            }
        return None

class TaskDetailSerializer(serializers.ModelSerializer):
    workflow = WorkflowSerializer(read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    matrix_block_info = serializers.SerializerMethodField()
    execution_time = serializers.SerializerMethodField()
    complexity_estimate = serializers.SerializerMethodField()
    parent_task_info = serializers.SerializerMethodField()
    subtasks = serializers.SerializerMethodField()
    
    class Meta:
        model = Task
        fields = '__all__'
    
    def get_matrix_block_info(self, obj):
        """Renvoie les informations détaillées sur le bloc matriciel"""
        if None not in [obj.matrix_block_row_start, obj.matrix_block_row_end, 
                       obj.matrix_block_col_start, obj.matrix_block_col_end]:
            
            # Calcul des dimensions du bloc
            rows = obj.matrix_block_row_end - obj.matrix_block_row_start + 1
            cols = obj.matrix_block_col_end - obj.matrix_block_col_start + 1
            
            # Estimation de la taille en mémoire
            bytes_per_element = 8  # Double précision (8 octets)
            size_bytes = rows * cols * bytes_per_element
            
            # Format plus lisible pour la taille
            if size_bytes < 1024:
                size_str = f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                size_str = f"{size_bytes/1024:.2f} KB"
            else:
                size_str = f"{size_bytes/(1024*1024):.2f} MB"
            
            return {
                'position': f"Bloc [{obj.matrix_block_row_start}:{obj.matrix_block_row_end}, "
                          f"{obj.matrix_block_col_start}:{obj.matrix_block_col_end}]",
                'dimensions': [rows, cols],
                'size_formatted': size_str,
                'elements_count': rows * cols
            }
        return None
    
    def get_execution_time(self, obj):
        """Calcule le temps d'exécution si la tâche est terminée"""
        if obj.start_time and obj.end_time:
            duration = (obj.end_time - obj.start_time).total_seconds()
            
            # Formatter pour lisibilité
            if duration < 60:
                return f"{duration:.2f} secondes"
            elif duration < 3600:
                minutes, seconds = divmod(duration, 60)
                return f"{int(minutes)} min {int(seconds)} sec"
            else:
                hours, remainder = divmod(duration, 3600)
                minutes, seconds = divmod(remainder, 60)
                return f"{int(hours)} h {int(minutes)} min"
        return None
    
    def get_complexity_estimate(self, obj):
        """Estime la complexité de calcul de la tâche"""
        if obj.workflow.workflow_type in ['MATRIX_ADDITION', 'Addition de matrices']:
            # Pour l'addition: O(n*m) où n,m sont les dimensions du bloc
            if None not in [obj.matrix_block_row_start, obj.matrix_block_row_end, 
                            obj.matrix_block_col_start, obj.matrix_block_col_end]:
                rows = obj.matrix_block_row_end - obj.matrix_block_row_start + 1
                cols = obj.matrix_block_col_end - obj.matrix_block_col_start + 1
                operations = rows * cols
                return {
                    'operations': operations,
                    'complexity': 'O(n*m)',
                    'description': f"Addition de {operations} éléments"
                }
        
        elif obj.workflow.workflow_type in ['MATRIX_MULTIPLICATION', 'Multiplication de matrices']:
            # Pour la multiplication standard: O(n*m*k)
            matrix_data = obj.matrix_data or {}
            k = matrix_data.get('k_dimension', 0)
            
            if None not in [obj.matrix_block_row_start, obj.matrix_block_row_end, 
                            obj.matrix_block_col_start, obj.matrix_block_col_end]:
                rows = obj.matrix_block_row_end - obj.matrix_block_row_start + 1
                cols = obj.matrix_block_col_end - obj.matrix_block_col_start + 1
                
                operations = rows * cols * k
                
                # Ajuster selon l'algorithme utilisé
                algorithm = next((p['value'] for p in obj.parameters if p['name'] == 'algorithm'), 'standard')
                
                if algorithm == 'standard':
                    return {
                        'operations': operations,
                        'complexity': 'O(n*m*k)',
                        'description': f"Multiplication standard avec {operations} opérations flottantes"
                    }
                elif algorithm == 'strassen':
                    # Strassen a une complexité d'environ O(n^2.8) au lieu de O(n^3)
                    operations_estimation = int(operations * 0.8)  # Approximation simplifiée
                    return {
                        'operations': operations_estimation,
                        'complexity': 'O(n^2.8)',
                        'description': f"Algorithme de Strassen avec environ {operations_estimation} opérations"
                    }
                
        return None
    
    def get_parent_task_info(self, obj):
        """Renvoie des informations sur la tâche parente si c'est une sous-tâche"""
        if obj.parent_task:
            return {
                'id': obj.parent_task.id,
                'name': obj.parent_task.name,
                'status': obj.parent_task.status,
                'status_display': obj.parent_task.get_status_display()
            }
        return None
    
    def get_subtasks(self, obj):
        """Renvoie la liste des sous-tâches si c'est une tâche principale"""
        subtasks = Task.objects.filter(parent_task=obj)
        if subtasks.exists():
            return [
                {
                    'id': subtask.id,
                    'name': subtask.name,
                    'status': subtask.status,
                    'status_display': subtask.get_status_display(),
                    'progress': subtask.progress,
                    'matrix_position': f"[{subtask.matrix_block_row_start}:{subtask.matrix_block_row_end}, {subtask.matrix_block_col_start}:{subtask.matrix_block_col_end}]" 
                    if None not in [subtask.matrix_block_row_start, subtask.matrix_block_row_end, 
                                   subtask.matrix_block_col_start, subtask.matrix_block_col_end] else None
                }
                for subtask in subtasks
            ]
        return []

class MatrixTaskSerializer(serializers.ModelSerializer):
    """Sérialiseur spécialisé pour les tâches matricielles avec des informations supplémentaires"""
    workflow_name = serializers.CharField(source='workflow.name', read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    matrix_dimensions = serializers.SerializerMethodField()
    block_dimensions = serializers.SerializerMethodField()
    computation_intensity = serializers.SerializerMethodField()
    
    class Meta:
        model = Task
        fields = [
            'id', 'workflow_name', 'name', 'description', 'status', 'status_display',
            'progress', 'matrix_block_row_start', 'matrix_block_row_end',
            'matrix_block_col_start', 'matrix_block_col_end', 'matrix_dimensions',
            'block_dimensions', 'computation_intensity', 'required_resources',
            'assigned_to', 'start_time', 'end_time', 'execution_time'
        ]
    
    def get_matrix_dimensions(self, obj):
        """Extrait les dimensions de la matrice à partir des paramètres"""
        if not obj.parameters:
            return None
            
        for param in obj.parameters:
            if param.get('name') == 'matrix_a_dimensions':
                return param.get('value')
        
        return None
    
    def get_block_dimensions(self, obj):
        """Calcule les dimensions du bloc"""
        if None not in [obj.matrix_block_row_start, obj.matrix_block_row_end, 
                       obj.matrix_block_col_start, obj.matrix_block_col_end]:
            rows = obj.matrix_block_row_end - obj.matrix_block_row_start + 1
            cols = obj.matrix_block_col_end - obj.matrix_block_col_start + 1
            return [rows, cols]
        return None
    
    def get_computation_intensity(self, obj):
        """Évalue l'intensité de calcul requise pour cette tâche"""
        if None in [obj.matrix_block_row_start, obj.matrix_block_row_end, 
                   obj.matrix_block_col_start, obj.matrix_block_col_end]:
            return None
            
        rows = obj.matrix_block_row_end - obj.matrix_block_row_start + 1
        cols = obj.matrix_block_col_end - obj.matrix_block_col_start + 1
        
        if obj.workflow.workflow_type in ['MATRIX_ADDITION', 'Addition de matrices']:
            # Pour l'addition, l'intensité est proportionnelle au nombre d'éléments
            elements = rows * cols
            if elements < 10000:
                return "Faible"
            elif elements < 1000000:
                return "Moyenne"
            else:
                return "Élevée"
        
        elif obj.workflow.workflow_type in ['MATRIX_MULTIPLICATION', 'Multiplication de matrices']:
            # Pour la multiplication, l'intensité dépend aussi de la dimension commune k
            matrix_data = obj.matrix_data or {}
            k = matrix_data.get('k_dimension', 0)
            
            operations = rows * cols * k
            if operations < 100000:
                return "Faible"
            elif operations < 10000000:
                return "Moyenne"
            else:
                return "Élevée"
                
        return None