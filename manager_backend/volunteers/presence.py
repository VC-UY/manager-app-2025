"""
Présence réelle des volontaires côté Manager.

Un volontaire n'est « en ligne » que s'il a envoyé un heartbeat récent.
Les tâches ASSIGNED non acceptées sont republicées ou libérées.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional

from django.utils import timezone

logger = logging.getLogger(__name__)

# Au-delà de ce délai sans heartbeat → offline
ONLINE_TTL_SECONDS = 90
# ASSIGNED sans acceptation au-delà → republication / réassignation
ASSIGNMENT_STALE_SECONDS = 45


def mark_online(
    coordinator_volunteer_id: str,
    *,
    name: Optional[str] = None,
    resources: Optional[Dict[str, Any]] = None,
    status: str = "available",
    preferences: Optional[Dict[str, Any]] = None,
) -> None:
    from volunteers.models import Volunteer

    if not coordinator_volunteer_id:
        return

    resources = resources or {}
    # Si hors plage horaire, le volontaire signale offline
    if status == "offline":
        mark_offline(coordinator_volunteer_id, reason="schedule")
        return

    defaults = {
        "status": status if status in ("available", "busy") else "available",
        "available": True,
        "last_seen": timezone.now(),
    }
    if name:
        defaults["name"] = name
    if resources.get("cpu_cores") is not None:
        defaults["cpu_cores"] = int(resources.get("cpu_cores") or 1)
    if resources.get("memory_mb") is not None:
        defaults["ram_mb"] = int(resources.get("memory_mb") or 1024)
    if resources.get("disk_space_mb") is not None:
        defaults["disk_gb"] = max(1, int(resources.get("disk_space_mb") or 1024) // 1024)
    if "gpu" in resources:
        defaults["gpu"] = bool(resources.get("gpu"))
    if resources.get("ip_address"):
        defaults["ip_address"] = resources.get("ip_address")

    # Valeurs minimales pour create
    defaults.setdefault("name", f"Volontaire {coordinator_volunteer_id[:8]}")
    defaults.setdefault("hostname", defaults["name"])
    defaults.setdefault("cpu_cores", 1)
    defaults.setdefault("ram_mb", 1024)
    defaults.setdefault("disk_gb", 10)

    existing = Volunteer.objects.filter(
        coordinator_volunteer_id=str(coordinator_volunteer_id)
    ).first()
    previous_status = existing.status if existing else None
    meta = dict((existing.meta_info if existing else None) or {})
    if preferences:
        meta["preferences"] = preferences
        # Aligner ressources offertes sur les préférences
        if preferences.get("max_cpu_cores"):
            defaults["cpu_cores"] = int(preferences["max_cpu_cores"])
        if preferences.get("max_ram_gb"):
            defaults["ram_mb"] = int(preferences["max_ram_gb"]) * 1024
        if preferences.get("max_disk_gb"):
            defaults["disk_gb"] = int(preferences["max_disk_gb"])
    defaults["meta_info"] = meta

    volunteer, created = Volunteer.objects.update_or_create(
        coordinator_volunteer_id=str(coordinator_volunteer_id),
        defaults=defaults,
    )
    volunteer.refresh_from_db()

    came_back = (not created) and previous_status in (None, "offline", "OFFLINE")
    status_changed = created or previous_status != volunteer.status

    if status_changed:
        _notify_presence(volunteer)

    # Volontaire de retour : relancer immédiatement les tâches en attente / échouées
    if created or came_back:
        logger.info(
            "Volontaire de retour en ligne: %s (%s) — reprise des travaux",
            volunteer.name,
            coordinator_volunteer_id,
        )
        _trigger_recovery()

    logger.debug("Volontaire en ligne: %s (%s)", volunteer.name, coordinator_volunteer_id)


def _trigger_recovery() -> None:
    """Lance la reprise des tâches en arrière-plan (évite de bloquer le heartbeat)."""
    import threading

    def _run():
        try:
            from tasks.recovery import recover_pending_and_failed_work

            result = recover_pending_and_failed_work()
            logger.info("Recovery après retour volontaire: %s", result)
        except Exception as exc:
            logger.warning("Recovery après retour volontaire échouée: %s", exc)

    threading.Thread(target=_run, name="volunteer-recovery", daemon=True).start()


def mark_offline(coordinator_volunteer_id: str, reason: str = "timeout") -> None:
    from volunteers.models import Volunteer

    volunteer = Volunteer.objects.filter(
        coordinator_volunteer_id=str(coordinator_volunteer_id)
    ).exclude(status="offline").first()
    if not volunteer:
        return
    volunteer.status = "offline"
    volunteer.available = False
    volunteer.save(update_fields=["status", "available"])
    _notify_presence(volunteer)
    logger.info("Volontaire offline (%s): %s", reason, coordinator_volunteer_id)


def _notify_presence(volunteer) -> None:
    """Diffuse le statut réel au frontend Manager (temps réel)."""
    try:
        from websocket_service.client import notify_event

        notify_event(
            "volunteer_status",
            {
                "volunteer_id": str(volunteer.id),
                "coordinator_volunteer_id": str(volunteer.coordinator_volunteer_id),
                "name": volunteer.name,
                "hostname": volunteer.hostname,
                "status": volunteer.status,
                "available": volunteer.available,
                "is_online": volunteer.status in ("available", "busy") and volunteer.available,
                "last_seen": volunteer.last_seen.isoformat() if volunteer.last_seen else None,
                "cpu_cores": volunteer.cpu_cores,
                "ram_mb": volunteer.ram_mb,
                "ip_address": volunteer.ip_address,
            },
        )
    except Exception as exc:
        logger.debug("Notification présence ignorée: %s", exc)


def sweep_stale_volunteers(ttl_seconds: int = ONLINE_TTL_SECONDS) -> int:
    """Passe en offline les volontaires sans heartbeat récent."""
    from volunteers.models import Volunteer

    cutoff = timezone.now() - timedelta(seconds=ttl_seconds)
    qs = Volunteer.objects.filter(status__in=["available", "busy"]).filter(
        last_seen__lt=cutoff
    )
    count = 0
    for volunteer in qs:
        volunteer.status = "offline"
        volunteer.available = False
        volunteer.save(update_fields=["status", "available"])
        _notify_presence(volunteer)
        count += 1
        logger.info(
            "Volontaire marqué offline (dernier signal %s): %s",
            volunteer.last_seen,
            volunteer.coordinator_volunteer_id,
        )
    return count


def get_online_volunteers_data(ttl_seconds: int = ONLINE_TTL_SECONDS) -> List[Dict[str, Any]]:
    """Liste des volontaires réellement en ligne pour l'assignation."""
    from volunteers.models import Volunteer

    sweep_stale_volunteers(ttl_seconds)
    cutoff = timezone.now() - timedelta(seconds=ttl_seconds)
    volunteers = Volunteer.objects.filter(
        status__in=["available", "busy"],
        available=True,
        last_seen__gte=cutoff,
    ).exclude(coordinator_volunteer_id="")

    data = []
    for volunteer in volunteers:
        data.append(
            {
                "volunteer_id": str(volunteer.coordinator_volunteer_id),
                "username": volunteer.name,
                "resources": {
                    "cpu_cores": volunteer.cpu_cores or 1,
                    "memory_mb": volunteer.ram_mb or 1024,
                    "disk_space_mb": (volunteer.disk_gb or 10) * 1024,
                    "gpu": bool(volunteer.gpu),
                    "ip_address": volunteer.ip_address or "0.0.0.0",
                },
            }
        )
    return data


def release_stale_assignments(stale_seconds: int = ASSIGNMENT_STALE_SECONDS) -> int:
    """
    Libère les tâches ASSIGNED non acceptées dont le volontaire est offline
    ou dont l'assignation est trop ancienne.
    """
    from tasks.models import Task, TaskStatus
    from volunteers.models import VolunteerTask

    cutoff = timezone.now() - timedelta(seconds=stale_seconds)
    online_ids = {
        v["volunteer_id"] for v in get_online_volunteers_data()
    }

    released = 0
    stale_links = VolunteerTask.objects.filter(
        status__in=["ASSIGNED", "assigned"],
        accepted_at__isnull=True,
        assigned_at__lt=cutoff,
    ).select_related("task", "volunteer")

    for link in stale_links:
        volunteer = link.volunteer
        vid = str(volunteer.coordinator_volunteer_id or "")
        volunteer_offline = (
            not vid
            or vid not in online_ids
            or volunteer.status == "offline"
            or not volunteer.available
        )
        if not volunteer_offline and link.assigned_at >= cutoff:
            continue

        task = link.task
        if task.status not in (TaskStatus.ASSIGNED, TaskStatus.PENDING, TaskStatus.CREATED):
            continue

        task.status = TaskStatus.CREATED
        task.save(update_fields=["status"])
        link.status = "EXPIRED"
        link.save(update_fields=["status"])
        released += 1
        try:
            from tasks.coordinator_sync import publish_task_status

            publish_task_status(
                task.workflow,
                task,
                clear_assignment=True,
                message="Assignation expirée — tâche remise en file d'attente",
            )
        except Exception:
            pass
        logger.info(
            "Tâche %s libérée (volontaire %s indisponible / assignation expirée)",
            task.id,
            vid or "?",
        )

    return released
