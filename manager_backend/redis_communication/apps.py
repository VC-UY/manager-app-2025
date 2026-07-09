"""
Configuration de l'application Django pour le module de communication Redis.
"""
from django.apps import AppConfig
import logging
import sys
import json
import os
import threading

logger = logging.getLogger('redis_communication')
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)


def _bootstrap_redis():
    """Initialise Redis et la synchro manager hors du chemin de demarrage HTTP."""
    try:
        from .client import RedisClient
        from .handlers import DEFAULT_HANDLERS
        from .auth_client import register_manager, login_manager
        from workflows.models import User

        client = RedisClient.get_instance()
        if not client.running:
            client.start()

        for channel, handler in DEFAULT_HANDLERS.items():
            try:
                client.subscribe(channel, handler)
            except Exception as sub_err:
                logger.error(f"Abonnement canal {channel}: {sub_err}")

        user = User.objects.get_last_inserted()
        if not user:
            logger.warning("Aucun manager local a synchroniser.")
            return

        if not user.remote_id:
            logger.info("Synchronisation manager avec le coordinateur en arriere-plan...")
            success, data = register_manager(
                username=user.username,
                email=user.email,
                password=user.password,
                first_name=user.first_name,
                last_name=user.last_name,
                timeout=8,
            )
            if success:
                remote_id = data.get('manager_id')
                if remote_id:
                    user.remote_id = remote_id
                    user.save(update_fields=['remote_id'])
                os.makedirs('.manager', exist_ok=True)
                with open('.manager/manager_info.json', 'w') as f:
                    json.dump({
                        'remote_id': remote_id,
                        'username': user.username,
                        'email': user.email,
                    }, f)
            else:
                logger.error(f"Erreur enregistrement manager: {data}")
                return

        success, data = login_manager(user.username, user.password, timeout=8)
        if success:
            os.makedirs('.manager', exist_ok=True)
            with open('.manager/manager_login_info.json', 'w') as f:
                json.dump({
                    'token': data.get('token'),
                    'refresh_token': data.get('refresh_token'),
                    'manager_id': data.get('manager_id') or user.remote_id,
                }, f)
            if data.get('token'):
                user.coordinator_token = data.get('token')
                user.save(update_fields=['coordinator_token'])
            logger.debug("Manager connecte avec succes")
        else:
            logger.error(f"Erreur connexion manager: {data}")
    except Exception as exc:
        logger.error(f"Bootstrap Redis en arriere-plan: {exc}")
        import traceback
        logger.error(traceback.format_exc())


def _pending_assignment_loop():
    """Présence + reprise périodique (CREATED, ASSIGNED expirées, FAILED retryables)."""
    import time
    import random

    time.sleep(10)
    while True:
        try:
            from tasks.recovery import recover_pending_and_failed_work

            result = recover_pending_and_failed_work()
            if any(result.get(k) for k in ("released", "prepared_failed", "assigned", "online")):
                logger.info("Boucle recovery: %s", result)
        except Exception as exc:
            logger.warning("Boucle d'assignation en attente: %s", exc)
        # Intervalle aléatoire pour lisser la charge (5s à 20s)
        time.sleep(random.randint(5, 20))


class RedisCommunicationConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'redis_communication'
    verbose_name = 'Communication Redis Universelle'

    def ready(self):
        if 'runserver' not in sys.argv and 'daphne' not in sys.argv[0]:
            return

        logger.info("Initialisation du service de communication Redis...")
        threading.Thread(target=_bootstrap_redis, daemon=True).start()
        threading.Thread(target=_pending_assignment_loop, daemon=True).start()
        logger.debug("Service de communication Redis demarre en arriere-plan")
