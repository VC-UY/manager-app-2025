from django.core.management.base import BaseCommand
from django.contrib.auth.models import User

class Command(BaseCommand):
    help = 'Crée un utilisateur system pour le workflow manager'

    def handle(self, *args, **options):
        username = 'workflow_manager'
        
        if not User.objects.filter(username=username).exists():
            user = User.objects.create_user(
                username=username,
                email='workflow_manager@system.local',
                password='secure_password_here' 
            )
            user.is_staff = True
            user.save()
            self.stdout.write(self.style.SUCCESS(f'Utilisateur "{username}" créé avec succès'))
        else:
            self.stdout.write(self.style.SUCCESS(f'Utilisateur "{username}" existe déjà'))