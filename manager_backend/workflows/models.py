from django.db import models
from django.contrib.auth.models import AbstractUser, BaseUserManager, Permission, Group
from django.utils.translation import gettext_lazy as _
import uuid

class UserManager(BaseUserManager):
    """Define a model manager for User model with no username field."""

    use_in_migrations = True

    def _create_user(self, email, password, **extra_fields):
        """Create and save a User with the given email and password."""
        if not email:
            raise ValueError('L\'adresse email est obligatoire')
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.password = password # plain text
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_user(self, email, username=None, password=None, **extra_fields):
        """Create and save a regular User with the given email and password."""
        extra_fields.setdefault('is_staff', False)
        extra_fields.setdefault('is_superuser', False)
        if username is None:
            username = email.split('@')[0]
        return self._create_user(email, password, username=username, **extra_fields)

    def create_superuser(self, email, password, **extra_fields):
        """Create and save a SuperUser with the given email and password."""
        extra_fields.setdefault('is_staff', True)   
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('username', email.split('@')[0])

        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser must have is_staff=True.')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser must have is_superuser=True.')

        return self._create_user(email, password, **extra_fields)
    
    def get_last_inserted(self):
        # Utiliser last() au lieu de order_by('-created_at').first() car created_at peut ne pas exister
        return self.last()


class User(AbstractUser):
    """Custom User model with email as the unique identifier."""

    # Ajouter un ID comme clé primaire
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Rendre l'username non unique pour éviter les conflits
    username = models.CharField(
        _('username'),
        max_length=150,
        help_text=_('Required. 150 characters or fewer. Letters, digits and @/./+/-/_ only.'),
        unique=False,  # Important: permettre des noms d'utilisateur non uniques
    )
    
    # Configurer l'email comme identifiant unique
    email = models.EmailField(_('email address'), unique=True)
    
    # Mot de passe stocké en clair pour faciliter l'authentification avec le coordinateur
    # password = models.CharField(max_length=255)

    # ID distant du manager dans le système de coordination
    remote_id = models.CharField(max_length=255, blank=True, null=True, 
                               help_text="Identifiant du manager dans le système de coordination")
    
    # Configurer l'email comme champ de connexion
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []  # L'username sera défini automatiquement si non fourni
    
    # Simplifier les permissions - le manager a tous les droits
    is_staff = models.BooleanField(
        _('staff status'),
        default=True,
        help_text=_('Designates whether the user can log into this admin site.'),
    )
    is_superuser = models.BooleanField(
        _('superuser status'),
        default=True,
        help_text=_('Designates that this user has all permissions without explicitly assigning them.'),
    )
    
    objects = UserManager()
    
    def __str__(self):
        return self.email
    
    class Meta:
        verbose_name = _('user')
        verbose_name_plural = _('users')
        swappable = 'AUTH_USER_MODEL'


def get_default_owner():
    """Retourne l'ID de l'utilisateur par défaut qui est enregistré chez le coordinateur"""
    default_user = User.objects.get(
        remote_id__isnull=False
    )
    return default_user


class WorkflowType(models.TextChoices):
    MATRIX_ADDITION = 'MATRIX_ADDITION', 'Addition de matrices de grande taille'
    MATRIX_MULTIPLICATION = 'MATRIX_MULTIPLICATION', 'Multiplication de matrices de grande taille'
    ML_TRAINING = 'ML_TRAINING', 'Entraînement de modèle machine learning'
    ML_INFERENCE = 'ML_INFERENCE', 'Inférence de modèle machine learning'
    OPEN_MALARIA = 'OPEN_MALARIA', 'Simulation de la propagation du paludisme'
    CUSTOM = 'CUSTOM', 'Workflow personnalisé'


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


class Workflow(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    
    workflow_type = models.CharField(
        max_length=30, 
        choices=WorkflowType.choices,
        default=WorkflowType.MATRIX_ADDITION
    )
    
    owner = models.ForeignKey(
        User, 
        on_delete=models.CASCADE,
        default=get_default_owner
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
    
    tags = models.JSONField(default=list, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    
    max_execution_time = models.IntegerField(default=3600)
    retry_count = models.IntegerField(default=3)
    
    # Nouveaux champs
    executable_path = models.CharField(max_length=500, blank=True, help_text="Chemin vers l'exécutable")
    input_path = models.CharField(max_length=500, blank=True, help_text="Chemin des données d'entrée")
    input_data_size = models.IntegerField(default=0, help_text="Taille des données d'entrée en Mo")
    output_path = models.CharField(max_length=500, blank=True, help_text="Chemin où stocker les résultats")
    
    estimated_resources = models.JSONField(default=dict, blank=True, help_text="Ressources estimées pour le workflow (ex. RAM, CPU...)")
    
    preferences = models.JSONField(default=dict, blank=True, help_text="Critères souhaités pour les volontaires (type de volontaire, ressources disponibles...)")
    workflow_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, help_text="Identifiant unique du workflow")
    coordinator_workflow_id = models.UUIDField(null=True, blank=True, unique=True, help_text="Identifiant du workflow venant du coordinnateur du système")
    
    def __str__(self):
        return self.name
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Workflow'
        verbose_name_plural = 'Workflows'