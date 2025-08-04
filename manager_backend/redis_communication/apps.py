"""
Configuration de l'application Django pour le module de communication Redis.
"""
from django.apps import AppConfig
import logging
import sys
import json
import os

# Configuration du logging pour afficher les messages dans la console
logger = logging.getLogger('redis_communication')
logger.setLevel(logging.DEBUG)

# Ajouter un gestionnaire de console si aucun n'existe déjà
if not logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

class RedisCommunicationConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'redis_communication'
    verbose_name = 'Communication Redis Universelle'
    
    def ready(self):
        """
        Méthode appelée lorsque l'application est prête.
        Initialise et démarre le client Redis.
        """
        # Ne pas exécuter en mode commande (sauf pour runserver)
        if 'runserver' not in sys.argv and 'daphne' not in sys.argv[0]:
            return
            
        logger.info("Initialisation du service de communication Redis...")
        
        try:
            # Importer ici pour éviter les importations circulaires
            from .client import RedisClient
            from .handlers import DEFAULT_HANDLERS
            
            # Récupérer ou créer l'instance du client
            client = RedisClient.get_instance()

            
            # Enregistrer les gestionnaires par défaut
            for channel, handler in DEFAULT_HANDLERS.items():
                client.subscribe(channel, handler)
            
            if not client.running:
                client.start()
            
            # Lancer l'enregistrement
            from .auth_client import register_manager, login_manager
            from workflows.models import User

            user = User.objects.get_last_inserted()

            if user and not user.remote_id:
                sucess, data = register_manager(username=user.username,
                                    email=user.email,
                                    password=user.password,
                                    first_name=user.first_name,
                                    last_name=user.last_name
                                    )
                
                if sucess:
                    # logger.warning(data)
                    remote_id = data.get('manager_id')
                    user.remote_id = remote_id
                    user.save()

                    # Ecrire les informations dans .manager/manager_info.json
                    if not os.path.exists('.manager'):
                        os.makedirs('.manager')
                    with open('.manager/manager_info.json', 'w') as f:
                        json.dump({
                            'remote_id': remote_id,
                            'username': user.username,
                            'email': user.email,
                            'password': user.password
                        }, f)
                    logger.debug("Manager enregistré avec succès")


                    # Lancer le login
                    sucess, data = login_manager(username=user.username,
                                    password=user.password)
                    
                    if sucess:
                        token = data.get('token')
                        refresh_token = data.get('refresh_token')
                        manager_id = data.get('manager_id')

                        # Ecrire les informations dans .manager/manager_login_info.json
                        if not os.path.exists('.manager'):
                            os.makedirs('.manager')
                        with open('.manager/manager_login_info.json', 'w') as f:
                            json.dump({
                                'token': token,
                                'refresh_token': refresh_token,
                                'manager_id': manager_id
                            }, f)
                        logger.debug("Manager connecté avec succès")
                    else:
                        logger.error(f"Erreur lors de la connexion du manager, {data}")
                else:
                    logger.error(f"Erreur lors de l'enregistrement du manager, {data}")
            elif user and user.remote_id:
                logger.debug("Manager deja enregistré")
                sucess, data = login_manager(user.username, user.password)
                if sucess:
                    # logger.warning(data)
                    token = data.get('token')
                    refresh_token = data.get('refresh_token')
                    manager_id = user.remote_id

                    # Ecrire les informations dans .manager/manager_login_info.json
                    if not os.path.exists('.manager'):
                        os.makedirs('.manager')
                    with open('.manager/manager_login_info.json', 'w') as f:
                        json.dump({
                            'token': token,
                            'refresh_token': refresh_token,
                            'manager_id': manager_id
                        }, f)
                    logger.debug("Manager connecté avec succès")
                else:
                    logger.error(f"Erreur lors de la connexion du manager, {data}")
            else:
                logger.error("Le client Redis n'est pas en cours d'execution")
                
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du client Redis: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        logger.debug("Service de communication Redis démarré")
