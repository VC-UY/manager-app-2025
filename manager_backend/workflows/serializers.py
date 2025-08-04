# backend/workflows/serializers.py
from rest_framework import serializers
from .models import Workflow, User
from django.contrib.auth.password_validation import validate_password

from django.contrib.auth import get_user_model
import traceback

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
        Crée un nouvel utilisateur avec les données validées.
        """
        try:
            # Supprimer le champ de confirmation du mot de passe
            validated_data.pop('password2', None)
            
            print(f"[DEBUG] Création d'utilisateur avec email: {validated_data.get('email')}, "
                  f"username: {validated_data.get('username')}")
            
            # Créer l'utilisateur avec create_user pour hasher le mot de passe
            user = User.objects.create_user(
                email=validated_data['email'],
                username=validated_data['username'],
                password=validated_data['password']
            )
            
            print(f"[DEBUG] Utilisateur créé avec succès, ID: {user.id}")
            return user
            
        except Exception as e:
            print(f"[ERROR] Erreur lors de la création de l'utilisateur: {str(e)}")
            print(traceback.format_exc())
            raise serializers.ValidationError(f"Erreur lors de la création de l'utilisateur: {str(e)}")