from django.db import models
from workflows.models import Workflow
from django.utils import timezone
import uuid

class TaskStatus(models.TextChoices):
    CREATED = "CREATED", "Créée"
    PENDING = "PENDING", "En attente"
    ASSIGNED = "ASSIGNED", "Assignée"
    RUNNING = "RUNNING", "En cours"
    COMPLETED = "COMPLETED", "Terminée"
    FAILED = "FAILED", "Échouée"
    RETRYING = "RETRYING", "En réessai"
    CANCELLED = "CANCELLED", "Annulée"

class VolunteerPreferenceType(models.TextChoices):
    ANY = "ANY", "N'importe lequel"
    TRUSTED = "TRUSTED", "Volontaires de confiance"
    HIGH_MEMORY = "HIGH_MEMORY", "Machines à haute mémoire"
    LOW_LATENCY = "LOW_LATENCY", "Faible latence"

def get_default_workflow():
    """Retourne l'ID du workflow par défaut ou None si aucun workflow n'existe."""
    from workflows.models import Workflow
    default_workflow = Workflow.objects.first()
    if default_workflow:
        return default_workflow.id
    return None

class Task(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    workflow = models.ForeignKey(
        Workflow, 
        on_delete=models.CASCADE, 
        related_name='tasks',
        null=True,  # Permettre les valeurs nulles pour faciliter les migrations
        blank=True
    )

    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)

    command = models.CharField(max_length=500)  # Commande à exécuter (dans le conteneur Docker)
    parameters = models.JSONField(default=list)  # Paramètres à passer à la commande
    dependencies = models.JSONField(default=list)  # Liste d’IDs de tâches dont dépend celle-ci

    status = models.CharField(
        max_length=20, 
        choices=TaskStatus.choices, 
        default=TaskStatus.PENDING
    )

    parent_task = models.ForeignKey(
        'self', null=True, blank=True, 
        on_delete=models.CASCADE, related_name='subtasks'
    )   
    is_subtask = models.BooleanField(default=False)

    progress = models.FloatField(default=0)
    created_at = models.DateTimeField(default=timezone.now)
    start_time = models.DateTimeField(null=True, blank=True)
    end_time = models.DateTimeField(null=True, blank=True)

    # Informations pour planification intelligente
    required_resources = models.JSONField(default=dict)  # Ex: {"cpu": 2, "memory_mb": 512}
    estimated_max_time = models.FloatField(help_text="Durée estimée en secondes", default=0)

    # Tentatives et échecs
    attempts = models.IntegerField(default=0)
    results = models.JSONField(default=dict, blank=True, null=True)
    error_details = models.JSONField(default=dict, blank=True, null=True)

    # Docker
    docker_info = models.JSONField(default=dict)

    # Préférences de volontaires
    volunteer_preference = models.CharField(
        max_length=20,
        choices=VolunteerPreferenceType.choices,
        default=VolunteerPreferenceType.ANY
    )

    # Gestion des fichiers
    input_files = models.JSONField(default=list)   # ex: ["data_0.pkl", "meta.json"]
    input_size = models.IntegerField(default=0, help_text="Taille des fichiers d'entrée en Mo")
    output_files = models.JSONField(default=list)  # ex: ["model_0.pt"]

    # Tracking pour monitoring et logs
    last_updated = models.DateTimeField(auto_now=True)
    assigned_to = models.JSONField(max_length=255, blank=True, null=True, help_text="Identifiant du volontaire assigné")

    def __str__(self):
        return f"{self.name} ({self.workflow.name})"
    
    class Meta:
        ordering = ['-is_subtask', 'created_at']
        verbose_name = 'Tâche'
        verbose_name_plural = 'Tâches'
    
    def increment_attempts(self):
        self.attempts += 1
        self.save(update_fields=['attempts'])
        return self.attempts
