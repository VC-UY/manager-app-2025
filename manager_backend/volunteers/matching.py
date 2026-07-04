"""Filtrage des volontaires selon leurs préférences, ressources et capacité temps."""

from __future__ import annotations

from datetime import datetime, time
from typing import Any, Dict, List, Optional

from django.utils import timezone


DAY_INDEX = {
    "lundi": 0,
    "mardi": 1,
    "mercredi": 2,
    "jeudi": 3,
    "vendredi": 4,
    "samedi": 5,
    "dimanche": 6,
}

# Tâches en cours qui consomment la capacité temps du volontaire
ACTIVE_TASK_STATUSES = ("ASSIGNED", "ACCEPTED", "RUNNING", "STARTED")


def _prefs(volunteer) -> dict:
    meta = volunteer.meta_info or {}
    return meta.get("preferences") or {}


def is_within_schedule(prefs: dict, when: Optional[datetime] = None) -> bool:
    schedule = prefs.get("schedule") or []
    if not schedule:
        return True
    when = when or timezone.localtime()
    today = when.weekday()
    now_t = when.timetz().replace(tzinfo=None, second=0, microsecond=0)
    for slot in schedule:
        day = (slot.get("day") or "").strip().lower()
        if DAY_INDEX.get(day) != today:
            continue
        try:
            start = time.fromisoformat(slot.get("start", "00:00"))
            end = time.fromisoformat(slot.get("end", "23:59"))
        except ValueError:
            continue
        if start <= now_t <= end:
            return True
    return False


def volunteer_max_capacity_seconds(volunteer) -> Optional[float]:
    """
    Capacité temps max du volontaire (préférences), en secondes.
    None = pas de limite déclarée (on peut lui donner autant de tâches que ses ressources le permettent).
    """
    prefs = _prefs(volunteer)
    max_min = int(prefs.get("duree_max_execution") or 0)
    if max_min <= 0:
        return None
    return float(max_min) * 60.0


def volunteer_used_capacity_seconds(volunteer) -> float:
    """Somme des estimations des tâches déjà ASSIGNED/RUNNING pour ce volontaire."""
    from volunteers.models import VolunteerTask

    total = 0.0
    links = (
        VolunteerTask.objects.filter(volunteer=volunteer, status__in=ACTIVE_TASK_STATUSES)
        .select_related("task")
    )
    for link in links:
        task = link.task
        if not task or task.status not in ACTIVE_TASK_STATUSES:
            continue
        total += float(task.estimated_max_time or 0)
    return total


def volunteer_remaining_capacity_seconds(volunteer) -> Optional[float]:
    """
    Temps encore disponible pour de nouvelles tâches.
    None = illimité.
    """
    max_cap = volunteer_max_capacity_seconds(volunteer)
    if max_cap is None:
        return None
    return max(0.0, max_cap - volunteer_used_capacity_seconds(volunteer))


def task_estimated_seconds(task) -> float:
    """Durée estimée d'une tâche (secondes). Minimum 60s si non renseignée."""
    est = float(getattr(task, "estimated_max_time", None) or 0)
    return est if est > 0 else 60.0


def volunteer_can_run_task(
    volunteer,
    task,
    *,
    remaining_seconds: Optional[float] = None,
) -> bool:
    """
    True si le volontaire peut exécuter CETTE tâche maintenant.

    La capacité temps est par tâche (et budget restant), pas pour le workflow entier :
    un volontaire de 40 min peut prendre des tâches tant que leur somme ≤ 40 min ;
    le reste reste en file d'attente.
    """
    prefs = _prefs(volunteer)
    if not is_within_schedule(prefs):
        return False

    req = task.required_resources or {}
    req_cpu = float(req.get("cpu") or req.get("cpu_cores") or 1)
    req_ram = float(req.get("ram") or req.get("memory_mb") or 512)
    req_disk = float(req.get("disk") or req.get("disk_gb") or 1)

    max_cpu = float(prefs.get("max_cpu_cores") or volunteer.cpu_cores or 1)
    max_ram_mb = float(prefs.get("max_ram_gb") or 0) * 1024.0
    if max_ram_mb <= 0:
        max_ram_mb = float(volunteer.ram_mb or 1024)
    max_disk = float(prefs.get("max_disk_gb") or volunteer.disk_gb or 1)

    if req_cpu > max_cpu + 0.05:
        return False
    if req_ram > max_ram_mb + 1:
        return False
    if req_disk > max_disk + 0.05:
        return False

    est = task_estimated_seconds(task)
    if remaining_seconds is None:
        remaining_seconds = volunteer_remaining_capacity_seconds(volunteer)
    # Budget temps : la tâche doit tenir dans le reste disponible
    if remaining_seconds is not None and est > remaining_seconds + 1:
        return False

    types = (prefs.get("types_calcul_autorises") or "").strip()
    if types and task.workflow_id:
        allowed = {t.strip().upper() for t in types.split(",") if t.strip()}
        wf_type = (getattr(task.workflow, "workflow_type", "") or "").upper()
        if allowed and wf_type and wf_type not in allowed:
            return False

    min_prio = int(prefs.get("priorite_min_acceptee") or 0)
    wf_prio = int(getattr(task.workflow, "priority", 0) or 0) if task.workflow_id else 0
    if min_prio and wf_prio < min_prio:
        return False

    return True


def filter_volunteers_for_task(volunteers: List, task) -> List:
    return [v for v in volunteers if volunteer_can_run_task(v, task)]


def filter_volunteer_data_for_task(
    volunteers_data: List[Dict[str, Any]],
    task,
    volunteer_objs: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Filtre la liste d'assignation (dicts) selon les préférences stockées."""
    result = []
    for vdata in volunteers_data:
        vid = str(vdata.get("volunteer_id") or "")
        volunteer = volunteer_objs.get(vid)
        if volunteer is None:
            result.append(vdata)
            continue
        if volunteer_can_run_task(volunteer, task):
            result.append(vdata)
    return result
