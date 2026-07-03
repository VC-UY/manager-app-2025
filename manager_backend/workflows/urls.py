from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    WorkflowViewSet,
    RegisterView,
    LoginView,
    LogoutView,
    get_workflow_outputs,
    download_workflow_output,
    download_workflow_outputs_zip,
)
from .submit_dispatcher import submit_workflow_dispatcher

router = DefaultRouter()
router.register(r'', WorkflowViewSet, basename='workflow')

urlpatterns = [
    path('auth/register/', RegisterView.as_view(), name='user-register'),
    path('auth/login/', LoginView.as_view(), name='user-login'),
    path('auth/logout/', LogoutView.as_view(), name='user-logout'),
    path('<str:workflow_id>/outputs/download-zip/', download_workflow_outputs_zip, name='workflow-outputs-zip'),
    path('<str:workflow_id>/outputs/download/<path:file_path>', download_workflow_output, name='workflow-output-download'),
    path('<str:workflow_id>/outputs/', get_workflow_outputs, name='workflow-outputs'),
    path('<str:workflow_id>/submit/', submit_workflow_dispatcher, name='submit-workflow'),
    path('', include(router.urls)),
]
