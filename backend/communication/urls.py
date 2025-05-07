from django.urls import path
from . import views

app_name = 'communication'

urlpatterns = [
    # Point d'entrée pour les notifications du coordinateur
    path('notifications/coordinator/', views.coordinator_notification, name='coordinator_notification'),
    
    # Point d'entrée pour les notifications des volontaires
    path('notifications/volunteer/', views.volunteer_notification, name='volunteer_notification'),
    
    # Point d'entrée pour les mises à jour de statut
    path('status/task/<str:task_id>/', views.task_status_update, name='task_status_update'),
    path('status/workflow/<str:workflow_id>/', views.workflow_status_update, name='workflow_status_update'),
    
    # Point d'entrée pour les résultats de tâches
    path('results/task/<str:task_id>/', views.task_result_notification, name='task_result_notification'),
    
    # Point d'entrée pour les contrôles du gestionnaire de messages
    path('matrix/start/', views.start_matrix_handler, name='start_matrix_handler'),
    path('matrix/stop/', views.stop_matrix_handler, name='stop_matrix_handler'),
    path('matrix/status/', views.matrix_handler_status, name='matrix_handler_status'),
]