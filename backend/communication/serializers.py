from rest_framework import serializers

class VolunteerSerializer(serializers.Serializer):
    """Serializer pour les données d'un volontaire."""
    id = serializers.CharField(read_only=True)
    name = serializers.CharField(read_only=True, allow_null=True, allow_blank=True)
    status = serializers.CharField(read_only=True)
    
    # Ressources du volontaire
    resources = serializers.DictField(
        child=serializers.JSONField(),
        read_only=True,
        allow_null=True
    )
    
    # Champs supplémentaires potentiels
    lastSeen = serializers.DateTimeField(read_only=True, allow_null=True)
    performanceHistory = serializers.DictField(
        child=serializers.JSONField(),
        read_only=True,
        allow_null=True
    )

class TaskStatusSerializer(serializers.Serializer):
    """Serializer pour les données de statut d'une tâche."""
    taskId = serializers.CharField(read_only=True)
    volunteerId = serializers.CharField(read_only=True, allow_null=True)
    status = serializers.CharField(read_only=True)
    timestamp = serializers.DateTimeField(read_only=True, allow_null=True)
    progress = serializers.FloatField(read_only=True, allow_null=True)
    details = serializers.DictField(
        child=serializers.JSONField(),
        read_only=True,
        allow_null=True
    )

class TaskResultSerializer(serializers.Serializer):
    """Serializer pour les données de résultat d'une tâche."""
    taskId = serializers.CharField(read_only=True)
    status = serializers.CharField(read_only=True)
    completionTime = serializers.DateTimeField(read_only=True, allow_null=True)
    resultPath = serializers.CharField(read_only=True, allow_null=True)
    files = serializers.ListField(
        child=serializers.CharField(),
        read_only=True,
        allow_null=True
    )
    failedFiles = serializers.ListField(
        child=serializers.CharField(),
        read_only=True,
        allow_null=True
    )