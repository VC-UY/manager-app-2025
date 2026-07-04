"""
RPC request/response fiable via le proxy Redis.

Utilise des connexions fraiches et separees (subscribe vs publish),
ce qui evite les timeouts du client singleton partage.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, Optional, Tuple

import redis
from django.conf import settings

logger = logging.getLogger(__name__)


def _redis_kwargs() -> dict:
    return dict(
        host=getattr(settings, "REDIS_PROXY_HOST", "coordinator-proxy"),
        port=int(getattr(settings, "REDIS_PROXY_PORT", 6380)),
        db=int(getattr(settings, "REDIS_DB", 0)),
        decode_responses=True,
        protocol=2,
        lib_name=None,
        lib_version=None,
        socket_connect_timeout=10,
        socket_timeout=15,
    )


def proxy_publish(
    channel: str,
    data: Dict[str, Any],
    *,
    token: Optional[str] = None,
    sender_id: str = "manager",
    message_type: str = "request",
    request_id: Optional[str] = None,
    to_volunteers: bool = False,
) -> str:
    """Publie un message sans attendre de reponse.

    to_volunteers=True: publie via coordinator-proxy (6380) pour atteindre
    les volontaires externes. Sinon publie sur Redis interne.
    """
    rid = request_id or str(uuid.uuid4())
    payload = {
        "request_id": rid,
        "sender": {"type": "manager", "id": sender_id},
        "message_type": message_type,
        "timestamp": time.time(),
        "data": data,
    }
    if token:
        payload["token"] = token
    kwargs = _redis_kwargs()
    if to_volunteers:
        # Les volontaires sont abonnes via le proxy externe
        kwargs["host"] = "coordinator-proxy"
        kwargs["port"] = 6380
        kwargs["protocol"] = 2
        kwargs["lib_name"] = None
        kwargs["lib_version"] = None
    client = redis.Redis(**kwargs)
    try:
        client.publish(channel, json.dumps(payload))
        return rid
    finally:
        try:
            client.close()
        except Exception:
            pass


def proxy_request_response(
    request_channel: str,
    response_channel: str,
    data: Dict[str, Any],
    *,
    token: Optional[str] = None,
    sender_id: str = "manager",
    timeout: float = 30.0,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Publie une requete et attend la reponse correspondante.

    Returns:
        (success, response_data)
    """
    request_id = str(uuid.uuid4())
    payload = {
        "request_id": request_id,
        "sender": {"type": "manager", "id": sender_id},
        "message_type": "request",
        "timestamp": time.time(),
        "data": data,
    }
    if token:
        payload["token"] = token

    kwargs = _redis_kwargs()
    sub_client = redis.Redis(**kwargs)
    pub_client = redis.Redis(**kwargs)
    pubsub = sub_client.pubsub(ignore_subscribe_messages=True)

    try:
        pubsub.subscribe(response_channel)
        # Laisser le proxy enregistrer l'abonnement
        time.sleep(0.15)
        pub_client.publish(request_channel, json.dumps(payload))
        logger.info(
            "RPC %s -> %s request_id=%s",
            request_channel,
            response_channel,
            request_id,
        )

        deadline = time.time() + timeout
        while time.time() < deadline:
            message = pubsub.get_message(timeout=1.0)
            if not message or message.get("type") != "message":
                continue
            try:
                body = json.loads(message["data"])
            except (TypeError, json.JSONDecodeError):
                continue
            if body.get("request_id") != request_id:
                continue
            response_data = body.get("data") or {}
            status = response_data.get("status")
            logger.info("RPC response request_id=%s status=%s", request_id, status)
            return status == "success", response_data

        logger.error("RPC timeout request_id=%s channel=%s", request_id, request_channel)
        return False, {
            "status": "error",
            "message": f"Timeout: aucune reponse du coordinateur apres {timeout} secondes",
        }
    except Exception as exc:
        logger.error("RPC error on %s: %s", request_channel, exc)
        return False, {"status": "error", "message": str(exc)}
    finally:
        try:
            pubsub.unsubscribe(response_channel)
            pubsub.close()
        except Exception:
            pass
        try:
            sub_client.close()
            pub_client.close()
        except Exception:
            pass
