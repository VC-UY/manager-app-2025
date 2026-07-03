"""
URL configuration for manager_backend project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from manager_backend.health import health_check

from tasks.file_transfer import serve_workflow_input_file, upload_task_outputs

urlpatterns = [
    path('health/', health_check, name='health'),
    path('admin/', admin.site.urls),
    path(
        'workflow-files/<uuid:workflow_id>/<path:file_path>',
        serve_workflow_input_file,
        name='workflow-input-file',
    ),
    path(
        'tasks/<uuid:task_id>/outputs/',
        upload_task_outputs,
        name='task-output-upload',
    ),
    path('workflows/', include('workflows.urls')),
    path('tasks/', include('tasks.urls')),
    path('volunteers/', include('volunteers.urls')),
]
