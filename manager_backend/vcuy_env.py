"""Chargement des variables d'environnement partagées par le Manager."""

import os
from pathlib import Path


def _load_dotenv():
    env_path = Path(__file__).resolve().parent.parent / '.env'
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        pass


_load_dotenv()


def env_bool(key: str, default: bool = False) -> bool:
    return os.environ.get(key, str(default)).lower() in ('true', '1', 'yes')


def env_list(key: str, default: str = '') -> list:
    raw = os.environ.get(key, default)
    return [item.strip() for item in raw.split(',') if item.strip()]
