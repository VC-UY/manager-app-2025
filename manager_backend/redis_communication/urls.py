"""
Configuration des URLs pour l'application redis_communication.
"""

from django.urls import path
from . import views

app_name = 'redis_communication'

urlpatterns = [
    path('stats/', views.get_stats, name='stats'),
    path('channels/', views.get_channels, name='channels'),
    path('publish/', views.publish_message, name='publish'),
]
