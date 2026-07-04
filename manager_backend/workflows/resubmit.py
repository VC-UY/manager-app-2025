"""
Réinitialisation et resoumission des workflows échoués / bloqués.
"""

import logging

from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated

from workflows.models import Workflow, WorkflowStatus, WorkflowType
from workflows.views import process_workflow_submission
from workflows.openmalaria_views import process_openmalaria_submission

logger = logging.getLogger(__name__)

RESUBMITTABLE_STATUSES = {
    WorkflowStatus.FAILED,
    WorkflowStatus.PARTIAL_FAILURE,
    WorkflowStatus.PENDING,
    WorkflowStatus.PAUSED,
    'ERROR',
    'SUBMISSION_FAILED',
}

SUPPORTED_SUBMIT_TYPES = {
    WorkflowType.ML_TRAINING,
    WorkflowType.ML_INFERENCE,
    WorkflowType.MATRIX_ADDITION,
    WorkflowType.MATRIX_MULTIPLICATION,
    WorkflowType.OPEN_MALARIA,
    WorkflowType.CUSTOM,
}


def reset_workflow_for_resubmit(workflow: Workflow) -> dict:
    """
    Efface les assignations et tâches, remet le workflow en CREATED.
    Le découpage recréera des tâches propres à la prochaine soumission.
    """
    from volunteers.models import VolunteerTask

    task_qs = workflow.tasks.all()
    task_ids = list(task_qs.values_list('id', flat=True))
    vt_deleted = 0
    tasks_deleted = 0

    if task_ids:
        vt_deleted, _ = VolunteerTask.objects.filter(task_id__in=task_ids).delete()
        tasks_deleted, _ = task_qs.delete()

    previous_status = workflow.status
    workflow.status = WorkflowStatus.CREATED
    workflow.submitted_at = None
    workflow.completed_at = None
    # Conserver les chemins / metadata (inputs utilisateur)
    workflow.save(update_fields=['status', 'submitted_at', 'completed_at', 'updated_at'])

    logger.info(
        "Workflow %s réinitialisé pour resoumission (était %s, %s tâches, %s liens VT)",
        workflow.id,
        previous_status,
        tasks_deleted,
        vt_deleted,
    )

    try:
        from websocket_service.client import notify_event
        notify_event('workflow_status_change', {
            'workflow_id': str(workflow.id),
            'status': WorkflowStatus.CREATED,
            'message': 'Workflow réinitialisé, resoumission en cours…',
        })
    except Exception:
        pass

    return {
        'previous_status': previous_status,
        'tasks_deleted': tasks_deleted,
        'volunteer_links_deleted': vt_deleted,
    }


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def resubmit_workflow_view(request, workflow_id):
    """Réinitialise un workflow échoué / en attente puis le resoumet."""
    workflow = get_object_or_404(Workflow, id=workflow_id, owner=request.user)

    if workflow.status not in RESUBMITTABLE_STATUSES:
        return JsonResponse(
            {
                'success': False,
                'error': (
                    f"Impossible de resoumettre un workflow au statut « {workflow.status} ». "
                    f"Statuts autorisés : {', '.join(sorted(RESUBMITTABLE_STATUSES))}."
                ),
            },
            status=400,
        )

    if workflow.workflow_type not in SUPPORTED_SUBMIT_TYPES:
        return JsonResponse(
            {
                'success': False,
                'error': f"Type de workflow non supporté : {workflow.workflow_type}",
            },
            status=400,
        )

    try:
        reset_info = reset_workflow_for_resubmit(workflow)
    except Exception as exc:
        logger.exception("Échec de la réinitialisation du workflow %s", workflow_id)
        return JsonResponse(
            {'success': False, 'error': f"Réinitialisation impossible : {exc}"},
            status=500,
        )

    # Relancer la soumission standard
    if workflow.workflow_type == WorkflowType.OPEN_MALARIA:
        response = process_openmalaria_submission(workflow_id, request)
    else:
        response = process_workflow_submission(workflow_id)

    # Enrichir la réponse JSON si possible
    if hasattr(response, 'content'):
        try:
            import json
            payload = json.loads(response.content.decode('utf-8'))
            if isinstance(payload, dict):
                payload['resubmit'] = True
                payload['reset'] = reset_info
                return JsonResponse(payload, status=response.status_code)
        except Exception:
            pass

    return response
