"""
Tests d'intégration du découpage de workflows (Phase 3).
Usage: python manage.py test_workflow_splits
"""

import logging
import os
import shutil
import tempfile

from django.core.management.base import BaseCommand

from tasks.models import Task
from workflows.models import Workflow, WorkflowStatus, WorkflowType, User
from workflows.split_workflow import split_workflow

logger = logging.getLogger(__name__)

SPLIT_CASES = [
  (WorkflowType.MATRIX_ADDITION, {'num_tasks': 2, 'matrix_size': 64}),
  (WorkflowType.MATRIX_MULTIPLICATION, {'num_tasks': 2, 'matrix_size': 64}),
  (WorkflowType.ML_INFERENCE, {'num_tasks': 2, 'samples_per_task': 32}),
  (WorkflowType.CUSTOM, {
      'tasks': [
          {'name': 'Step A', 'command': 'echo A'},
          {'name': 'Step B', 'command': 'echo B', 'dependencies': []},
      ],
  }),
]


class Command(BaseCommand):
    help = 'Teste le découpage de tous les types de workflow supportés'

    def handle(self, *args, **options):
        passed = 0
        failed = 0
        tmp_root = tempfile.mkdtemp(prefix='vcuy_split_test_')

        try:
            owner = User.objects.filter(remote_id__isnull=False).first()
            if not owner:
                owner = User.objects.create_user(
                    email='split-test@vcuy.local',
                    password='testpass123',
                    username='split_test',
                )
                owner.remote_id = 'test-manager-id'
                owner.save()

            for workflow_type, metadata in SPLIT_CASES:
                work_dir = os.path.join(tmp_root, workflow_type.lower())
                os.makedirs(work_dir, exist_ok=True)

                workflow = Workflow.objects.create(
                    name=f'Test {workflow_type}',
                    workflow_type=workflow_type,
                    status=WorkflowStatus.CREATED,
                    owner=owner,
                    executable_path=work_dir,
                    input_path=work_dir,
                    output_path=os.path.join(work_dir, 'output'),
                    metadata=metadata,
                    input_data_size=64,
                )

                try:
                    kwargs = {}
                    if workflow_type == WorkflowType.OPEN_MALARIA:
                        kwargs = {'num_tasks': 2, 'population_per_task': 100}

                    tasks = split_workflow(
                        workflow.id,
                        workflow_type,
                        logger,
                        **kwargs,
                    )

                    if not tasks:
                        raise ValueError('Aucune tâche créée')

                    task_count = Task.objects.filter(workflow=workflow).count()
                    if task_count != len(tasks):
                        raise ValueError(f'Attendu {len(tasks)} tâches, trouvé {task_count}')

                    self.stdout.write(self.style.SUCCESS(
                        f'✓ {workflow_type}: {len(tasks)} tâche(s)'
                    ))
                    passed += 1
                except Exception as exc:
                    self.stdout.write(self.style.ERROR(
                        f'✗ {workflow_type}: {exc}'
                    ))
                    failed += 1
                finally:
                    Task.objects.filter(workflow=workflow).delete()
                    workflow.delete()

        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

        self.stdout.write('')
        if failed:
            self.stdout.write(self.style.ERROR(f'Échec: {failed} test(s)'))
            raise SystemExit(1)

        self.stdout.write(self.style.SUCCESS(f'Tous les tests OK ({passed}/{passed})'))
