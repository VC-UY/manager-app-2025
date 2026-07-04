"""
Soumet et assigne les workflows de demo ML + OpenMalaria de facon synchrone.
Usage: python manage.py run_e2e_demo
"""

import logging
import uuid

from django.core.management.base import BaseCommand
from django.utils import timezone

from redis_communication.proxy_rpc import proxy_publish
from redis_communication.utils import (
    build_task_file_transfer_info,
    get_manager_login_token,
)
from tasks.file_server import start_file_server
from tasks.models import Task, TaskStatus
from volunteers.models import Volunteer, VolunteerTask
from workflows.handlers import submit_workflow_handler
from workflows.models import Workflow, WorkflowStatus, WorkflowType
from workflows.split_workflow import split_workflow

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Soumet et assigne les demos ML_TRAINING et OPEN_MALARIA"

    def add_arguments(self, parser):
        parser.add_argument(
            "--workflow-ids",
            nargs="*",
            default=[
                "93cdfd3f-4267-4dc2-819f-e15032127541",
                "87256b80-0ae6-4ba3-965a-ea72c7791c79",
            ],
        )

    def handle(self, *args, **options):
        for workflow_id in options["workflow_ids"]:
            self._run_one(workflow_id)

    def _assign_all_to_first_volunteer(self, workflow, volunteers):
        """Assignation simple FCFS sans dependance torch/A3C."""
        vdata = volunteers[0]
        volunteer_id = vdata["volunteer_id"]
        resources = vdata.get("resources") or {}
        volunteer, _ = Volunteer.objects.update_or_create(
            coordinator_volunteer_id=volunteer_id,
            defaults={
                "name": vdata.get("username", f"Volontaire {volunteer_id}"),
                "cpu_cores": resources.get("cpu_cores", 1),
                "ram_mb": resources.get("memory_mb", 1024),
                "disk_gb": int(resources.get("disk_space_mb", 10240) / 1024),
                "status": "available",
                "gpu": resources.get("gpu", False),
                "ip_address": resources.get("ip_address", "0.0.0.0"),
            },
        )
        assignment = {volunteer_id: []}
        for task in workflow.tasks.all().order_by("created_at"):
            VolunteerTask.objects.create(
                volunteer=volunteer,
                task=task,
                assigned_at=timezone.now(),
                status=TaskStatus.ASSIGNED,
            )
            task.status = TaskStatus.ASSIGNED
            task.save(update_fields=["status"])
            assignment[volunteer_id].append(
                {"task_id": str(task.id), "task_name": task.name}
            )
        return assignment

    def _run_one(self, workflow_id: str):
        workflow = Workflow.objects.get(id=workflow_id)
        self.stdout.write(f"==> {workflow.workflow_type} {workflow.name}")

        workflow.status = WorkflowStatus.CREATED
        workflow.save(update_fields=["status"])

        ok, response = submit_workflow_handler(str(workflow.id), timeout=30)
        self.stdout.write(
            f"  submit: {ok} {response.get('status')} "
            f"vols={len(response.get('volunteers') or [])}"
        )
        if not ok:
            return

        volunteers = response.get("volunteers") or []
        if not volunteers:
            self.stdout.write(self.style.ERROR("  aucun volontaire"))
            return

        workflow.status = WorkflowStatus.SPLITTING
        workflow.submitted_at = timezone.now()
        workflow.save(update_fields=["status", "submitted_at"])

        workflow.tasks.all().delete()
        start_file_server(workflow)
        meta = workflow.metadata or {}
        if workflow.workflow_type == WorkflowType.OPEN_MALARIA:
            tasks = split_workflow(
                id=workflow.id,
                workflow_type=WorkflowType.OPEN_MALARIA,
                logger=logger,
                num_tasks=int(meta.get("num_tasks", 2)),
                population_per_task=int(meta.get("population_per_task", 500)),
            )
        else:
            tasks = split_workflow(
                id=workflow.id,
                workflow_type=workflow.workflow_type,
                logger=logger,
            )
        self.stdout.write(f"  split: {len(tasks)} taches")

        workflow.status = WorkflowStatus.ASSIGNING
        workflow.save(update_fields=["status"])
        assignment = self._assign_all_to_first_volunteer(workflow, volunteers)
        self.stdout.write(f"  assign: { {k: len(v) for k, v in assignment.items()} }")

        token = get_manager_login_token(workflow.owner)
        sender = str(workflow.owner.remote_id or "manager")

        for volunteer_id, task_list in assignment.items():
            enriched = []
            for info in task_list:
                task = Task.objects.get(id=info["task_id"])
                transfer = build_task_file_transfer_info(workflow, task)
                enriched.append(
                    {
                        "task_id": str(task.id),
                        "name": task.name,
                        "description": task.description,
                        "command": task.command,
                        "dependencies": task.dependencies,
                        "is_subtask": task.is_subtask,
                        "status": task.status,
                        "required_resources": task.required_resources,
                        "attempts": task.attempts,
                        "workflow_id": str(workflow.id),
                        "parameters": task.parameters,
                        "estimated_execution_time": task.estimated_max_time,
                        "input_data": transfer,
                        "input_data_size": task.input_size,
                        "docker_information": task.docker_info or {},
                    }
                )
            if not enriched:
                continue
            proxy_publish(
                "task/assignment",
                {
                    "workflow_id": str(workflow.id),
                    "assignments": {volunteer_id: enriched},
                },
                token=token,
                sender_id=sender,
                request_id=str(uuid.uuid4()),
                to_volunteers=False,  # meme Redis direct que le volontaire (port 6381 expose)
            )
            self.stdout.write(f"  published {len(enriched)} tasks -> {volunteer_id}")

        workflow.status = WorkflowStatus.RUNNING
        workflow.save(update_fields=["status"])
        self.stdout.write(self.style.SUCCESS(f"  RUNNING ({workflow.workflow_type})"))
