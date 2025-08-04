from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/manager/$', consumers.WorkflowConsumer.as_asgi()),
]