# backend/workflows/urls.py - Configuration des URLs corrigée

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    WorkflowViewSet, submit_workflow_view, RegisterView, LoginView, LogoutView,
    get_workflow_outputs, download_workflow_output, download_workflow_outputs_zip
)
from .openmalaria_views import submit_openmalaria_workflow_view
# Router pour les opérations CRUD sur les workflows
router = DefaultRouter()
router.register(r'', WorkflowViewSet)


# URLs pour l'application workflows
urlpatterns = [
    # Routes d'authentification avec des chemins clairs et non ambigus
    path('auth/register/', RegisterView.as_view(), name='user-register'),
    path('auth/login/', LoginView.as_view(), name='user-login'),
    path('auth/logout/', LogoutView.as_view(), name='user-logout'),

    # Routes pour les résultats/sorties des workflows
    path('<str:workflow_id>/outputs/', get_workflow_outputs, name='workflow-outputs'),
    path('<str:workflow_id>/outputs/download/<path:file_path>', download_workflow_output, name='workflow-output-download'),
    path('<str:workflow_id>/outputs/download-zip/', download_workflow_outputs_zip, name='workflow-outputs-zip'),

    # Routes pour les workflows
    path('', include(router.urls)),
    path('<str:workflow_id>/submit/', submit_openmalaria_workflow_view, name='submit-workflow'),
]

