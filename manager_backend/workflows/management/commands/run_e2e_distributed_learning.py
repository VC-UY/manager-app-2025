"""
E2E réel : crée, split, soumet et suit un workflow DISTRIBUTED_LEARNING.

Usage:
  python manage.py run_e2e_distributed_learning
  python manage.py run_e2e_distributed_learning --wait-minutes 45
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid

from django.core.management.base import BaseCommand
from django.utils import timezone

from tasks.file_server import start_file_server
from tasks.models import Task, TaskStatus
from tasks.coordinator_sync import publish_tasks_created, publish_workflow_tasks_ready
from workflows.handlers import submit_workflow_handler
from workflows.models import Workflow, WorkflowStatus, WorkflowType, User
from workflows.split_workflow import split_workflow

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Test E2E DISTRIBUTED_LEARNING (2 volontaires, gossip vc-uyr)"

    def add_arguments(self, parser):
        parser.add_argument("--n-volunteers", type=int, default=2)
        parser.add_argument("--max-rounds", type=int, default=1)
        parser.add_argument("--wait-minutes", type=int, default=30)
        parser.add_argument("--gossip-interval", type=int, default=15)
        parser.add_argument(
            "--dataset",
            type=str,
            default="cifar10",
            help="cifar10 (réel) ou fake (smoke sans données)",
        )
        parser.add_argument(
            "--max-train-samples",
            type=int,
            default=512,
            help="Sous-échantillon par volontaire (0 = partition complète)",
        )

    def handle(self, *args, **options):
        n_vol = max(1, int(options["n_volunteers"]))
        max_rounds = max(1, int(options["max_rounds"]))
        wait_s = int(options["wait_minutes"]) * 60
        gossip_interval = int(options["gossip_interval"])
        dataset = str(options["dataset"] or "cifar10").lower().strip()
        max_train_samples = int(options["max_train_samples"] or 0)

        owner = User.objects.filter(remote_id__isnull=False).first()
        if not owner:
            owner = User.objects.create_user(
                email="e2e-dl@vcuy.local",
                password="e2e-dl-pass",
                username=f"e2e_dl_{uuid.uuid4().hex[:8]}",
            )
            owner.remote_id = str(uuid.uuid4())
            owner.save()

        tmp = tempfile.mkdtemp(prefix="vcuy_e2e_dl_")
        inputs_dir = os.path.join(tmp, "inputs")
        os.makedirs(inputs_dir, exist_ok=True)
        use_fake = dataset in {"fake", "synthetic"}
        meta = {
            "n_volunteers": n_vol,
            "max_rounds": max_rounds,
            "gossip_interval": gossip_interval,
            "model": "resnet18",
            "dataset": "fake" if use_fake else "cifar10",
            "partition": "iid",
            "compression": "jointsq",
            "local_epochs": 1,
            "runtime": "vc-uyr",
            # Cache machine (~/.vcuy/datasets) ; download seulement si absent.
            "allow_dataset_download": "0" if use_fake else "1",
        }
        if max_train_samples > 0 and not use_fake:
            meta["max_train_samples"] = max_train_samples
        wf = Workflow.objects.create(
            name=f"E2E DL {timezone.now().strftime('%Y%m%d-%H%M%S')}",
            description="Test E2E gossip AD-PSGD via vc-uyr (2 volontaires)",
            workflow_type=WorkflowType.DISTRIBUTED_LEARNING,
            status=WorkflowStatus.CREATED,
            owner=owner,
            executable_path=tmp,
            # Servi tel quel par /api/workflow-files/<id>/<path>
            input_path=inputs_dir,
            output_path=os.path.join(tmp, "output"),
            metadata=meta,
            input_data_size=1,
            priority=1,
        )

        self.stdout.write(self.style.NOTICE(f"Workflow créé: {wf.id}"))
        tasks = split_workflow(wf.id, WorkflowType.DISTRIBUTED_LEARNING, logger)
        wf.refresh_from_db()
        meta = wf.metadata or {}
        self.stdout.write(
            self.style.SUCCESS(
                f"Split OK: {len(tasks)} tâche(s), manager DL {meta.get('dl_manager_host')}:{meta.get('dl_manager_port')}"
            )
        )
        for t in tasks:
            self.stdout.write(f"  - {t.name} | {t.command} | bundle={t.input_files}")

        server_port = start_file_server(wf)
        self.stdout.write(self.style.NOTICE(f"Serveur fichiers démarré port {server_port}"))

        ok, result = submit_workflow_handler(str(wf.id), timeout=120)
        if not ok:
            self.stdout.write(self.style.ERROR(f"Soumission échouée: {result}"))
            raise SystemExit(1)
        self.stdout.write(self.style.SUCCESS(f"Soumis au coordinateur: {result}"))

        published = publish_tasks_created(wf, tasks, server_port)
        publish_workflow_tasks_ready(
            wf,
            message=f"E2E DL — {published} tâche(s) gossip prêtes",
            file_server_port=server_port,
        )
        self.stdout.write(self.style.SUCCESS(f"Tâches publiées au coordinateur: {published}"))

        wf.status = WorkflowStatus.RUNNING
        wf.save(update_fields=["status", "updated_at"])

        deadline = time.time() + wait_s
        last_status = ""
        while time.time() < deadline:
            wf.refresh_from_db()
            counts = {}
            for t in Task.objects.filter(workflow=wf):
                counts[t.status] = counts.get(t.status, 0) + 1
            line = f"workflow={wf.status} tasks={counts} dl_port={meta.get('dl_manager_port')}"
            if line != last_status:
                self.stdout.write(line)
                last_status = line
            terminal = all(
                t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
                for t in Task.objects.filter(workflow=wf)
            )
            if terminal and Task.objects.filter(workflow=wf, status=TaskStatus.COMPLETED).exists():
                self.stdout.write(self.style.SUCCESS("E2E DL terminé avec succès"))
                return
            if terminal:
                failed = Task.objects.filter(workflow=wf, status=TaskStatus.FAILED)
                self.stdout.write(self.style.ERROR(f"E2E DL échoué: {failed.count()} tâche(s) en FAILED"))
                for t in failed:
                    self.stdout.write(f"  FAIL {t.name}: {t.error_details}")
                raise SystemExit(1)
            time.sleep(15)

        self.stdout.write(self.style.WARNING("Timeout — workflow encore en cours"))
        raise SystemExit(2)
