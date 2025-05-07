from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'', views.WorkflowViewSet)

urlpatterns = [
    # Endpoints pour les opérations de base sur les workflows
    path('', include(router.urls)),
    
    # Endpoints spécifiques au workflow matriciel 
    path('<uuid:pk>/submit/', views.WorkflowViewSet.as_view({'post': 'submit'}), name='workflow-submit'),
    path('<uuid:pk>/containerize/', views.WorkflowViewSet.as_view({'post': 'containerize'}), name='workflow-containerize'),
    path('<uuid:pk>/docker-images/', views.WorkflowViewSet.as_view({'get': 'docker_images'}), name='workflow-docker-images'),
    path('<uuid:pk>/push-images/', views.WorkflowViewSet.as_view({'post': 'push_images'}), name='workflow-push-images'),
    path('<uuid:pk>/tasks/', views.WorkflowViewSet.as_view({'get': 'tasks'}), name='workflow-tasks'),
    path('<uuid:pk>/aggregate-results/', views.WorkflowViewSet.as_view({'post': 'aggregate_results'}), name='workflow-aggregate'),
    path('<uuid:pk>/result-info/', views.WorkflowViewSet.as_view({'get': 'result_info'}), name='workflow-result-info'),
    
    # Contrôle d'exécution du workflow
    path('<uuid:pk>/pause/', views.WorkflowViewSet.as_view({'post': 'pause'}), name='workflow-pause'),
    path('<uuid:pk>/resume/', views.WorkflowViewSet.as_view({'post': 'resume'}), name='workflow-resume'),
    path('<uuid:pk>/cancel/', views.WorkflowViewSet.as_view({'post': 'cancel'}), name='workflow-cancel'),
]