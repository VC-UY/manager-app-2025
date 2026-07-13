"""
Transfert de fichiers via l'API publique du manager.

Les volontaires telechargent les entrees et poussent les sorties en HTTP,
sans dependre de ports aleatoires ni d'IP de conteneur Docker.
"""

import logging
import os

from django.http import FileResponse, Http404
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from tasks.models import Task, TaskStatus
from workflows.models import Workflow

logger = logging.getLogger(__name__)


def _safe_join(base_dir: str, relative_path: str) -> str:
    base = os.path.abspath(base_dir)
    target = os.path.abspath(os.path.join(base, relative_path))
    if not target.startswith(base + os.sep) and target != base:
        raise Http404("Chemin invalide")
    return target


@api_view(["GET"])
@permission_classes([AllowAny])
def serve_workflow_input_file(request, workflow_id, file_path):
    """Sert un fichier d'entree du workflow aux volontaires."""
    workflow = Workflow.objects.filter(id=workflow_id).first()
    base_dir = ""
    if workflow:
        base_dir = (workflow.input_path or "").strip()

    # Fallback: fichiers encore présents sur disque alors que la ligne DB a disparu
    if not base_dir or not os.path.isdir(base_dir):
        data_root = os.environ.get("WORKFLOW_DATA_ROOT", "/data/workflow_data")
        candidates = []
        if os.path.isdir(data_root):
            for owner in os.listdir(data_root):
                cand = os.path.join(data_root, owner, str(workflow_id), "inputs")
                if os.path.isdir(cand):
                    candidates.append(cand)
        if candidates:
            base_dir = candidates[0]
            logger.warning(
                "workflow-files fallback disque pour %s → %s",
                workflow_id,
                base_dir,
            )
        else:
            raise Http404("Repertoire d'entree introuvable")

    full_path = _safe_join(base_dir, file_path)
    if not os.path.isfile(full_path):
        # Compat: parfois le chemin demandé inclut déjà "inputs/"
        if file_path.startswith("inputs/"):
            full_path = _safe_join(base_dir, file_path[len("inputs/") :])
        if not os.path.isfile(full_path):
            raise Http404("Fichier introuvable")

    return FileResponse(open(full_path, "rb"), as_attachment=True, filename=os.path.basename(full_path))


@api_view(["POST"])
@permission_classes([AllowAny])
def upload_task_outputs(request, task_id):
    """Recoit les fichiers de sortie d'une tache depuis un volontaire."""
    task = get_object_or_404(Task, id=task_id)
    workflow = task.workflow

    if not workflow.output_path:
        workflow.output_path = os.path.join(workflow.executable_path or "/tmp", "outputs")
        workflow.save(update_fields=["output_path", "updated_at"])

    output_dir = os.path.join(workflow.output_path, str(task.id))
    os.makedirs(output_dir, exist_ok=True)

    uploaded = request.FILES.getlist("files")
    if not uploaded:
        # Compat: un seul fichier sous une cle quelconque
        uploaded = list(request.FILES.values())

    if not uploaded:
        return Response({"success": False, "error": "Aucun fichier recu"}, status=400)

    saved = []
    for uploaded_file in uploaded:
        name = os.path.basename(uploaded_file.name)
        if not name or name in (".", ".."):
            continue
        dest = os.path.join(output_dir, name)
        with open(dest, "wb") as handle:
            for chunk in uploaded_file.chunks():
                handle.write(chunk)
        saved.append(dest)
        logger.info("Sortie tache %s recue: %s", task_id, dest)

    if not saved:
        return Response({"success": False, "error": "Aucun fichier valide"}, status=400)

    task.output_files = saved
    task.status = TaskStatus.COMPLETED
    task.end_time = timezone.now()
    task.progress = 100
    task.save()

    from tasks.workflow_utils import check_and_finalize_workflow

    final_status = check_and_finalize_workflow(workflow)
    logger.info("Upload sorties tache %s OK (%s fichiers), workflow=%s", task_id, len(saved), final_status)

    return Response({
        "success": True,
        "files": [os.path.basename(path) for path in saved],
        "workflow_status": final_status,
    })
