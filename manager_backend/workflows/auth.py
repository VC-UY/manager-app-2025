from django.contrib.auth.backends import ModelBackend
from django.contrib.auth import get_user_model
from django.db.models import Q

User = get_user_model()

class EmailBackend(ModelBackend):
    def authenticate(self, request, email=None, username=None, password=None, **kwargs):
        try:
            if email:
                user = User.objects.get(email=email)
            elif username:
                user = User.objects.get(Q(username=username) | Q(email=username))
            else:
                return None
                
            if user.check_password(password):
                return user
        except User.DoesNotExist:
            return None
        
        return None