# backend/workflows/models.py# backend/workflows/models.py
from django.db import models
from django.contrib.auth.models import User
import uuid


def get_default_owner():
    default_user, _ = User.objects.get_or_create(
        username='workflow_manager',
        defaults={
            'email': 'workflow_manager@system.local',
            'is_staff': True
        }
    )
    return default_user.id 

class WorkflowType(models.TextChoices):
    MATRIX_ADDITION = 'MATRIX_ADDITION', 'Addition de matrices de grande taille'
    MATRIX_MULTIPLICATION = 'MATRIX_MULTIPLICATION', 'Multiplication de matrices de grande taille'

class WorkflowStatus(models.TextChoices):
    CREATED = 'CREATED', 'Créé'
    VALIDATED = 'VALIDATED', 'Validé'
    SUBMITTED = 'SUBMITTED', 'Soumis'
    SPLITTING = 'SPLITTING', 'En découpage'
    ASSIGNING = 'ASSIGNING', 'En attribution'
    PENDING = 'PENDING', 'En attente'
    RUNNING = 'RUNNING', 'En exécution'
    PAUSED = 'PAUSED', 'En pause'
    PARTIAL_FAILURE = 'PARTIAL_FAILURE', 'Échec partiel'
    REASSIGNING = 'REASSIGNING', 'Réattribution'
    AGGREGATING = 'AGGREGATING', 'Agrégation'
    COMPLETED = 'COMPLETED', 'Terminé'
    FAILED = 'FAILED', 'Échoué'

class TaskStatus(models.TextChoices):
    PENDING = 'PENDING', 'En attente'
    ASSIGNED = 'ASSIGNED', 'Assigné'
    RUNNING = 'RUNNING', 'En exécution'
    COMPLETED = 'COMPLETED', 'Terminé'
    FAILED = 'FAILED', 'Échoué'

class VolunteerPreferenceType(models.TextChoices):
    ANY = 'ANY', 'Tous types'
    CPU_INTENSIVE = 'CPU_INTENSIVE', 'Calcul intensif CPU'
    GPU_REQUIRED = 'GPU_REQUIRED', 'GPU nécessaire'
    NETWORK_INTENSIVE = 'NETWORK_INTENSIVE', 'Transfert intensif réseau'
    LOW_RESOURCE = 'LOW_RESOURCE', 'Ressources limitées'
    HIGH_MEMORY = 'HIGH_MEMORY', 'Mémoire importante'

class Workflow(models.Model):
    """
    Modèle principal pour les workflows de calcul distribué.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    workflow_type = models.CharField(
        max_length=30, 
        choices=WorkflowType.choices,
        default=WorkflowType.MATRIX_ADDITION
    )
    
    # Préférences pour les types de volontaires
    volunteer_preferences = models.JSONField(default=list)  # Liste des types de volontaires préférés
    min_volunteers = models.IntegerField(default=1)  # Nombre minimal de volontaires requis
    max_volunteers = models.IntegerField(default=100)  # Nombre maximal de volontaires
    
    owner = models.ForeignKey(
            User, 
            on_delete=models.CASCADE,
            default=get_default_owner  # Utiliser la fonction externe
    )
    status = models.CharField(
        max_length=20, 
        choices=WorkflowStatus.choices, 
        default=WorkflowStatus.CREATED
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    submitted_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    priority = models.IntegerField(default=1)
    
    # Champs pour stocker des informations structurées sous forme JSON
    estimated_resources = models.JSONField(default=dict)
    tags = models.JSONField(default=list)
    metadata = models.JSONField(default=dict)
    
    # Paramètres d'exécution
    max_execution_time = models.IntegerField(default=3600)  # Temps maximal d'exécution en secondes
    retry_count = models.IntegerField(default=3)  # Nombre de tentatives en cas d'échec
    
    def __str__(self):
        return self.name
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Workflow'
        verbose_name_plural = 'Workflows'
    
    def get_volunteer_preferences_display(self):
        """Retourne les préférences de volontaires en format lisible"""
        if not self.volunteer_preferences:
            return "Tous types"
        
        prefs = []
        for pref in self.volunteer_preferences:
            for choice in VolunteerPreferenceType.choices:
                if pref == choice[0]:
                    prefs.append(choice[1])
        
        return ", ".join(prefs) if prefs else "Tous types"
    



