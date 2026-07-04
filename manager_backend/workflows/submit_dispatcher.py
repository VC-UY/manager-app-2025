"""
Point d'entrée unique pour la soumission de workflows.
Dispatch vers le handler approprié selon le type de workflow.
"""

import logging

from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated

from workflows.models import Workflow, WorkflowStatus, WorkflowType
from workflows.views import process_workflow_submission
from workflows.openmalaria_views import process_openmalaria_submission

logger = logging.getLogger(__name__)

SUPPORTED_SUBMIT_TYPES = {
    WorkflowType.ML_TRAINING,
    WorkflowType.ML_INFERENCE,
    WorkflowType.MATRIX_ADDITION,
    WorkflowType.MATRIX_MULTIPLICATION,
    WorkflowType.OPEN_MALARIA,
    WorkflowType.CUSTOM,
}


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def submit_workflow_dispatcher(request, workflow_id):
    """Route la soumission vers le handler adapté au type de workflow."""
    workflow = get_object_or_404(Workflow, id=workflow_id, owner=request.user)

    if workflow.status != WorkflowStatus.CREATED:
        return JsonResponse(
            {
                'success': False,
                'error': (
                    f"Soumission impossible au statut « {workflow.status} ». "
                    "Utilisez /resubmit/ pour un workflow échoué ou en attente."
                ),
            },
            status=400,
        )

    if workflow.workflow_type == WorkflowType.CUSTOM:
        from workflows.custom_validation import validate_custom_metadata

        ok, err, metadata = validate_custom_metadata(workflow.metadata)
        if not ok:
            return JsonResponse({'success': False, 'error': err}, status=400)
        if metadata != (workflow.metadata or {}):
            workflow.metadata = metadata
            workflow.save(update_fields=['metadata', 'updated_at'])

    if workflow.workflow_type not in SUPPORTED_SUBMIT_TYPES:
        logger.warning(
            "Soumission refusée pour le workflow %s (type %s non supporté)",
            workflow_id,
            workflow.workflow_type,
        )
        return JsonResponse(
            {
                'success': False,
                'error': (
                    f"Type de workflow non supporté pour la soumission: "
                    f"{workflow.workflow_type}. "
                    f"Types supportés: {', '.join(SUPPORTED_SUBMIT_TYPES)}"
                ),
            },
            status=400,
        )

    if workflow.workflow_type == WorkflowType.OPEN_MALARIA:
        return process_openmalaria_submission(workflow_id, request)

    return process_workflow_submission(workflow_id)
