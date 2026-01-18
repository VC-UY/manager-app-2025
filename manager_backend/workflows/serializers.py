# backend/workflows/serializers.py
from rest_framework import serializers
from .models import Workflow, User
from django.contrib.auth.password_validation import validate_password

from django.contrib.auth import get_user_model
import traceback
import logging

from redis_communication.auth_client import register_manager

logger = logging.getLogger(__name__)

User = get_user_model()


class WorkflowSerializer(serializers.ModelSerializer):
    class Meta:
        model = Workflow
        fields = '__all__'
        read_only_fields = ('id', 'created_at', 'updated_at', 'submitted_at', 'completed_at')



from workflows.examples.distributed_training_demo.estimate_resources import estimate_resources

class WorkflowSerializer(serializers.ModelSerializer):
    class Meta:
        model = Workflow
        fields = '__all__'

    def create(self, validated_data):

        # Verfier le type de workflow
        workflow_type = validated_data.get("workflow_type")
        if workflow_type not in [choice[0] for choice in Workflow.WorkflowType.choices]:
            raise serializers.ValidationError("Invalid workflow type.")
        
        if workflow_type == Workflow.WorkflowType.ML_TRAINING:
            # Vérifier les champs requis pour le type de workflow ML_TRAINING
            if not validated_data.get("executable_path"):
                raise serializers.ValidationError("executable_path is required for ML_TRAINING workflow type.")
            if not validated_data.get("inputs_path"):
                raise serializers.ValidationError("inputs_path is required for ML_TRAINING workflow type.")
            # Estimer les ressources pour le type de workflow ML_TRAINING
            executable_path = validated_data.get("executable_path")
            inputs_path = validated_data.get("inputs_path")

            if inputs_path:
                try:
                    resources = estimate_resources(inputs_path)
                    validated_data["estimated_resources"] = resources
                except Exception as e:
                    raise serializers.ValidationError(f"Resource estimation failed: {e}")

            return super().create(validated_data)


class WorkflowSerializer(serializers.ModelSerializer):
    owner_username = serializers.SerializerMethodField()
    
    class Meta:
        model = Workflow
        fields = '__all__'
        read_only_fields = ('id', 'created_at', 'updated_at', 'submitted_at', 'completed_at', 'owner', 'owner_username')
    
    def get_owner_username(self, obj):
        return obj.owner.username if obj.owner else None
    

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'first_name', 'last_name', 'password')
        read_only_fields = ('id',)

class RegisterSerializer(serializers.ModelSerializer):
    password2 = serializers.CharField(write_only=True, required=True)
    
    class Meta:
        model = User
        fields = ('email', 'first_name', 'last_name', 'password', 'password2')
        extra_kwargs = {
            'password': {'write_only': True},
            'email': {'required': True}
        }
    
    def validate_email(self, value):
        """
        Valide que l'email n'est pas déjà utilisé.
        """
        if User.objects.filter(email=value).exists():
            raise serializers.ValidationError("Un utilisateur avec cet email existe déjà.")
        return value
    
    
    def validate(self, attrs):
        """
        Validation de tous les champs ensemble.
        """
        # Vérification des mots de passe
        if attrs.get('password') != attrs.get('password2'):
            raise serializers.ValidationError({"password": "Les mots de passe ne correspondent pas."})
        
        # Si l'username n'est pas fourni, générer à partir de l'email
        if not attrs.get('username'):
            email_prefix = attrs.get('email', '').split('@')[0]
            base_username = email_prefix[:30]  # Tronquer si nécessaire
            
            # Vérifier si ce nom d'utilisateur existe déjà
            username = base_username
            counter = 1
            
            while User.objects.filter(username=username).exists():
                # Ajouter un compteur jusqu'à trouver un nom disponible
                username = f"{base_username}{counter}"
                counter += 1
                
                if counter > 100:  # Éviter une boucle infinie
                    raise serializers.ValidationError(
                        {"username": "Impossible de générer un nom d'utilisateur unique."}
                    )
            
            attrs['username'] = username
            print(f"[DEBUG] Nom d'utilisateur généré: {username}")
        
        return attrs
    
    def create(self, validated_data):
        """
        Crée un nouvel utilisateur avec les données validées et le synchronise avec le coordinateur.
        """
        try:
            # Supprimer le champ de confirmation du mot de passe
            validated_data.pop('password2', None)

            logger.info(f"Création d'utilisateur avec email: {validated_data.get('email')}, "
                  f"username: {validated_data.get('username')}")

            # Créer l'utilisateur avec create_user pour hasher le mot de passe
            user = User.objects.create_user(
                email=validated_data['email'],
                username=validated_data['username'],
                password=validated_data['password'],
                first_name=validated_data.get('first_name', ''),
                last_name=validated_data.get('last_name', '')
            )

            logger.info(f"Utilisateur créé localement avec succès, ID: {user.id}")

            # Synchroniser avec le coordinateur
            try:
                logger.info(f"Synchronisation de l'utilisateur {user.username} avec le coordinateur...")
                success, response = register_manager(
                    username=user.username,
                    email=user.email,
                    password=validated_data['password'],  # Mot de passe en clair pour le coordinateur
                    first_name=validated_data.get('first_name', ''),
                    last_name=validated_data.get('last_name', ''),
                    timeout=30
                )

                if success:
                    # Stocker le remote_id (manager_id) retourné par le coordinateur
                    remote_id = response.get('manager_id')
                    if remote_id:
                        user.remote_id = remote_id
                        user.save()
                        logger.info(f"Utilisateur synchronisé avec le coordinateur, remote_id: {remote_id}")
                    else:
                        logger.warning(f"Enregistrement réussi mais pas de manager_id dans la réponse: {response}")
                else:
                    logger.warning(f"Échec de la synchronisation avec le coordinateur: {response.get('message', 'Erreur inconnue')}")
                    # L'utilisateur est créé localement même si la synchro échoue
                    # Il pourra être synchronisé plus tard

            except Exception as sync_error:
                logger.error(f"Erreur lors de la synchronisation avec le coordinateur: {sync_error}")
                # Ne pas échouer la création de l'utilisateur si la synchro échoue
                # L'utilisateur peut être synchronisé manuellement plus tard

            return user

        except Exception as e:
            logger.error(f"Erreur lors de la création de l'utilisateur: {str(e)}")
            logger.error(traceback.format_exc())
            raise serializers.ValidationError(f"Erreur lors de la création de l'utilisateur: {str(e)}")