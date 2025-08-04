from django.utils import timezone
from django.db import models
import uuid

from tasks.models import Task

class Volunteer(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    coordinator_volunteer_id = models.CharField(max_length=255, unique=True)
    name = models.CharField(max_length=255)
    hostname = models.CharField(max_length=255)
    last_ip_address = models.GenericIPAddressField(null=True, blank=True)
    cpu_cores = models.IntegerField()
    ram_mb = models.IntegerField()
    gpu = models.BooleanField(default=False)
    available = models.BooleanField(default=True)
    status = models.CharField(max_length=20, default="available")  # Ex: "available", "busy", "offline"
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    last_seen = models.DateTimeField(auto_now=True)
    disk_gb = models.IntegerField()
    tags = models.JSONField(default=list, blank=True)  # Ex: ["arm64", "nvidia", "raspberry", "gpu"]
    meta_info = models.JSONField(default=dict, blank=True)  

    def __str__(self):
        return f"{self.hostname} ({self.ip_address})"
    class Meta:
        ordering = ['hostname']
        verbose_name = 'Volontaire'
        verbose_name_plural = 'Volontaires'



class VolunteerTask(models.Model):
    task = models.ForeignKey(Task, on_delete=models.CASCADE, related_name="volunteer_tasks")
    volunteer = models.ForeignKey("Volunteer", on_delete=models.CASCADE, related_name="assigned_tasks")

    assigned_at = models.DateTimeField(default=timezone.now)
    progress = models.FloatField(default=0)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    STATUS_CHOICES = [
        ("ASSIGNED", "Assigned"),
        ("ACCEPTED", "Accepted"),
        ("STARTED", "Started"),
        ("COMPLETED", "Completed"),
        ("FAILED", "Failed"),
        ("EXPIRED", "Expired"),
        ("CANCEL", "Cancel"),
        ("RUNNING", "Running"),
        ("PAUSED", "Paused")
    ]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="ASSIGNED")
    progress = models.FloatField(default=0) 
    accepted_at = models.DateTimeField(null=True, blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    result = models.JSONField(null=True, blank=True)
    error = models.TextField(null=True, blank=True)

    class Meta:
        unique_together = ('task', 'volunteer')  # Une tâche ne peut pas être assignée plusieurs fois à un même volontaire

    def __str__(self):
        return f"{self.volunteer} - {self.task} ({self.status})"