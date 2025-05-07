from django.core.management.base import BaseCommand
import sys
import time
import logging
from django.conf import settings
from communication import get_auth_service, get_coordinator_service

logger = logging.getLogger('workflow_manager.test')

class Command(BaseCommand):
    help = 'Teste la connexion et l\'authentification avec le Coordinateur'

    def add_arguments(self, parser):
        parser.add_argument(
            '--action',
            type=str,
            default='authenticate',
            help='Action à exécuter (authenticate, volunteers, heartbeat)'
        )

    def handle(self, *args, **options):
        # Configuration du logging
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        action = options['action']
        
        self.stdout.write(self.style.SUCCESS(f'Exécution de l\'action: {action}'))
        
        # Authentification
        auth_service = get_auth_service()
        
        if action == 'authenticate':
            self.stdout.write('Tentative d\'authentification...')
            token = auth_service.authenticate()
            
            if token:
                self.stdout.write(self.style.SUCCESS(f'Authentification réussie ! Token: {token[:10]}...'))
                self.stdout.write(f'Expiration: {time.ctime(auth_service.token_expiry)}')
            else:
                self.stdout.write(self.style.ERROR('Échec de l\'authentification'))
        
        # Récupération des volontaires
        elif action == 'volunteers':
            self.stdout.write('Récupération des volontaires disponibles...')
            coordinator_service = get_coordinator_service()
            volunteers = coordinator_service.get_available_volunteers()
            
            if volunteers:
                self.stdout.write(self.style.SUCCESS(f'Récupération réussie ! {len(volunteers)} volontaires disponibles'))
                for idx, volunteer in enumerate(volunteers, 1):
                    self.stdout.write(f'{idx}. ID: {volunteer.get("id")}, Status: {volunteer.get("status")}')
                    self.stdout.write(f'   Resources: {volunteer.get("resources", {})}')
            else:
                self.stdout.write(self.style.WARNING('Aucun volontaire disponible ou échec de la récupération'))
        
        # Vérification de l'état du système
        elif action == 'heartbeat':
            self.stdout.write('Vérification de l\'état du système...')
            coordinator_service = get_coordinator_service()
            status = coordinator_service.check_system_status()
            
            if status and status.get('status') == 'success':
                self.stdout.write(self.style.SUCCESS('Système opérationnel !'))
                self.stdout.write(f'Version: {status.get("version", "N/A")}')
                self.stdout.write(f'Uptime: {status.get("uptime", "N/A")} secondes')
                self.stdout.write(f'Active volunteers: {status.get("activeVolunteers", 0)}')
                self.stdout.write(f'Pending tasks: {status.get("pendingTasks", 0)}')
            else:
                self.stdout.write(self.style.ERROR('Système non disponible'))
        
        else:
            self.stdout.write(self.style.ERROR(f'Action inconnue: {action}'))
            self.stdout.write('Actions disponibles: authenticate, volunteers, heartbeat')