from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'', views.VolunteerViewSet, basename='volunteer')
router.register(r'tasks', views.VolunteerTaskViewSet, basename='volunteer-task')

urlpatterns = [
    path('', include(router.urls)),
]
