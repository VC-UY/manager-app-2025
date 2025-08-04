from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

app_name = 'tasks'

router = DefaultRouter()
router.register(r'', views.TaskViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
