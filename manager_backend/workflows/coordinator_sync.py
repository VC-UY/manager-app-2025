"""Synchronisation des managers vers le coordinateur via HTTP interne."""

import logging
import os

import requests

logger = logging.getLogger(__name__)


def coordinator_base_url() -> str:
    return os.environ.get(
        'COORDINATOR_API_URL',
        os.environ.get('COORDINATOR_PUBLIC_URL', 'http://coordinator-api:8001'),
    ).rstrip('/')


def sync_manager_to_coordinator(
    *,
    username: str,
    email: str,
    password: str = '',
    first_name: str = '',
    last_name: str = '',
) -> tuple[bool, dict]:
    token = os.environ.get('COORDINATOR_INTERNAL_TOKEN', '').strip()
    if not token:
        logger.warning('COORDINATOR_INTERNAL_TOKEN manquant, synchro ignoree')
        return False, {'message': 'token interne manquant'}

    url = f"{coordinator_base_url()}/api/internal/managers/"
    try:
        response = requests.post(
            url,
            json={
                'username': username,
                'email': email,
                'password': password,
                'first_name': first_name,
                'last_name': last_name,
            },
            headers={'X-Internal-Token': token, 'Content-Type': 'application/json'},
            timeout=10,
        )
        data = response.json() if response.content else {}
        if response.status_code in (200, 201) and data.get('manager_id'):
            return True, data
        logger.warning('Synchro coordinateur echouee (%s): %s', response.status_code, data)
        return False, data
    except Exception as exc:
        logger.error('Erreur synchro coordinateur: %s', exc)
        return False, {'message': str(exc)}
