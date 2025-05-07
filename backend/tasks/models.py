#backend/tasks/models.py

from django.db import models
from workflows.models import Workflow, TaskStatus, VolunteerPreferenceType
from django.utils import timezone
import uuid

class Task(models.Model):
    """
    Modèle pour les tâches individuelles dans un workflow.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    workflow = models.ForeignKey(Workflow, on_delete=models.CASCADE, related_name='tasks')
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    command = models.CharField(max_length=500)
    parameters = models.JSONField(default=list)
    dependencies = models.JSONField(default=list)
    status = models.CharField(
        max_length=20, 
        choices=TaskStatus.choices, 
        default=TaskStatus.PENDING
    )
    parent_task = models.ForeignKey('self', null=True, blank=True, on_delete=models.CASCADE, related_name='subtasks')
    is_subtask = models.BooleanField(default=False)
    progress = models.FloatField(default=0)
    created_at = models.DateTimeField(default=timezone.now)
    start_time = models.DateTimeField(null=True, blank=True)
    end_time = models.DateTimeField(null=True, blank=True)
    required_resources = models.JSONField(default=dict)
    assigned_to = models.CharField(max_length=255, blank=True, null=True)
    attempts = models.IntegerField(default=0)
    results = models.JSONField(default=dict, blank=True, null=True)
    error_details = models.JSONField(default=dict, blank=True, null=True)
    docker_image = models.CharField(max_length=255, blank=True, null=True)
    
    # Préférences spécifiques pour cette tâche
    volunteer_preference = models.CharField(
        max_length=20,
        choices=VolunteerPreferenceType.choices,
        default=VolunteerPreferenceType.ANY
    )
    
    # Informations spécifiques pour les tâches matricielles
    matrix_block_row_start = models.IntegerField(null=True, blank=True)
    matrix_block_row_end = models.IntegerField(null=True, blank=True)
    matrix_block_col_start = models.IntegerField(null=True, blank=True)
    matrix_block_col_end = models.IntegerField(null=True, blank=True)
    
    # Pour stocker les données matricielles ou les références
    matrix_data = models.JSONField(default=dict, blank=True, null=True)
    
    # Informations spécifiques selon le type de workflow
    input_files = models.JSONField(default=list)  # Liste des fichiers d'entrée
    output_files = models.JSONField(default=list)  # Liste des fichiers de sortie attendus
    
    def __str__(self):
        return f"{self.name} ({self.workflow.name})"
    
    class Meta:
        ordering = ['-is_subtask', 'created_at']
        verbose_name = 'Tâche'
        verbose_name_plural = 'Tâches'
    
    def increment_attempts(self):
        """Incrémente le compteur de tentatives"""
        self.attempts += 1
        self.save(update_fields=['attempts'])
        return self.attempts
    
    def set_status(self, status):
        """Met à jour le statut avec horodatage approprié"""
        self.status = status
        
        if status == TaskStatus.RUNNING and not self.start_time:
            self.start_time = timezone.now()
        elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            self.end_time = timezone.now()
            
        self.save(update_fields=['status', 'start_time', 'end_time'])