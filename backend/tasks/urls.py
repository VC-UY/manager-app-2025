from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'', views.TaskViewSet)

urlpatterns = [
    # Routes de base via le routeur
    path('', include(router.urls)),
    
    # Endpoints spécifiques aux tâches matricielles
    path('<uuid:pk>/assign/', views.TaskViewSet.as_view({'post': 'assign'}), name='task-assign'),
    path('<uuid:pk>/start/', views.TaskViewSet.as_view({'post': 'start_execution'}), name='task-start'),
    path('<uuid:pk>/update-progress/', views.TaskViewSet.as_view({'post': 'update_progress'}), name='task-progress'),
    path('<uuid:pk>/complete/', views.TaskViewSet.as_view({'post': 'complete'}), name='task-complete'),
    path('<uuid:pk>/abort/', views.TaskViewSet.as_view({'post': 'abort'}), name='task-abort'),
    path('<uuid:pk>/reassign/', views.TaskViewSet.as_view({'post': 'reassign'}), name='task-reassign'),
    
    # Endpoints d'agrégation pour les tâches matricielles
    path('<uuid:pk>/aggregate-results/', views.TaskViewSet.as_view({'post': 'aggregate_results'}), name='task-aggregate'),
    path('<uuid:pk>/matrix-info/', views.TaskViewSet.as_view({'get': 'matrix_info'}), name='task-matrix-info'),
    
    # Endpoints de gestion des sous-tâches
    path('<uuid:pk>/subtasks/', views.TaskViewSet.as_view({'get': 'subtasks'}), name='task-subtasks'),
    path('<uuid:pk>/retry-failed-subtasks/', views.TaskViewSet.as_view({'post': 'retry_failed_subtasks'}), name='task-retry-subtasks'),
]