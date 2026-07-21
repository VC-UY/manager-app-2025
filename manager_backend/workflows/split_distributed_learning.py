"""
Découpage workflow DISTRIBUTED_LEARNING — gossip AD-PSGD via vc-uyr (sans Docker).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

from tasks.models import Task
from workflows.models import Workflow
from workflows.bundle_builder import RUNTIME_META, package_files_as_bundle
from workflows.distributed_learning_service import start_for_workflow
from workflows.split_workflow import get_min_volunteer_resources

logger = logging.getLogger(__name__)

EXAMPLES_DIR = Path(__file__).resolve().parent / "examples" / "distributed_learning"


def split_distributed_learning_workflow(workflow_instance: Workflow, split_logger: logging.Logger):
    """
    Crée une tâche longue durée par slot volontaire pour l'apprentissage gossip.
    Démarre le manager DL TCP et le bridge VC-UY en parallèle.
    """
    metadata = dict(workflow_instance.metadata or {})
    n_volunteers = max(1, int(metadata.get("n_volunteers") or 2))
    max_rounds = int(metadata.get("max_rounds") or 10)
    gossip_interval = int(metadata.get("gossip_interval") or 30)
    model_name = str(metadata.get("model") or "resnet18")
    dataset = str(metadata.get("dataset") or "cifar10")
    partition = str(metadata.get("partition") or "iid")
    compression = str(metadata.get("compression") or "jointsq")
    local_epochs = int(metadata.get("local_epochs") or 1)

    input_dir = os.path.join(workflow_instance.executable_path or "/tmp", "inputs")
    os.makedirs(input_dir, exist_ok=True)

    from django.conf import settings
    from redis_communication.utils import get_local_ip

    public_host = (
        os.environ.get("DL_MANAGER_PUBLIC_HOST")
        or getattr(settings, "DL_MANAGER_PUBLIC_HOST", None)
        or get_local_ip()
        or "127.0.0.1"
    )
    dl_port = start_for_workflow(workflow_instance, public_host=public_host)

    metadata.update(
        {
            "paradigm": "gossip_distributed_learning",
            "n_volunteers": n_volunteers,
            "max_rounds": max_rounds,
            "gossip_interval": gossip_interval,
            "model": model_name,
            "dataset": dataset,
            "partition": partition,
            "compression": compression,
            "local_epochs": local_epochs,
            "dl_manager_host": public_host,
            "dl_manager_port": dl_port,
        }
    )
    workflow_instance.metadata = metadata
    workflow_instance.save(update_fields=["metadata", "updated_at"])

    min_resources = get_min_volunteer_resources()
    # Besoins ML plus élevés (PyTorch)
    # Défauts ML ; surchargeables via metadata.required_resources
    req = metadata.get("required_resources") or {}
    min_resources = {
        **min_resources,
        "memory_mb": int(req.get("memory_mb") or max(int(min_resources.get("memory_mb", 1024)), 2048)),
        "cpu_cores": int(req.get("cpu_cores") or max(int(min_resources.get("cpu_cores", 1)), 1)),
    }

    worker_scripts = [
        EXAMPLES_DIR / "run_volunteer_vcuy.py",
        EXAMPLES_DIR / "volunteer_vcuy.py",
        EXAMPLES_DIR / "volunteer_core.py",
    ]
    src_dir = EXAMPLES_DIR / "src"
    runtime_meta = dict(RUNTIME_META)
    tasks = []

    estimated_seconds = max(600, max_rounds * (gossip_interval + 120))

    for slot in range(n_volunteers):
        slot_dir = os.path.join(input_dir, f"volunteer_{slot}")
        os.makedirs(slot_dir, exist_ok=True)

        dl_config = {
            "volunteer_slot": slot,
            "n_volunteers": n_volunteers,
            "manager_host": public_host,
            "manager_port": dl_port,
            "env": {
                "MODEL_NAME": model_name,
                "DATASET": dataset,
                "DATA_PARTITION": partition,
                "COMPRESSION": compression,
                "MAX_ROUNDS": str(max_rounds),
                "GOSSIP_INTERVAL": str(gossip_interval),
                "LOCAL_EPOCHS": str(local_epochs),
                "VOLUNTEER_ID": str(slot),
                "N_VOLUNTEERS": str(n_volunteers),
                "MANAGER_HOST": public_host,
                "MANAGER_PORT": str(dl_port),
                "VCUY_SKIP_DL_COORDINATOR": "1",
            },
        }
        config_path = os.path.join(slot_dir, "dl_config.json")
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(dl_config, handle, indent=2)

        # Bundle : config + scripts + src/
        bundle_staging = os.path.join(slot_dir, "bundle_staging")
        if os.path.isdir(bundle_staging):
            shutil.rmtree(bundle_staging)
        os.makedirs(bundle_staging, exist_ok=True)

        shutil.copy2(config_path, os.path.join(bundle_staging, "dl_config.json"))
        for script in worker_scripts:
            if script.is_file():
                shutil.copy2(script, os.path.join(bundle_staging, script.name))
        if src_dir.is_dir():
            def _ignore_pycache(dirpath, names):
                return [n for n in names if n == "__pycache__" or n.endswith(".pyc")]

            shutil.copytree(
                src_dir,
                os.path.join(bundle_staging, "src"),
                dirs_exist_ok=True,
                ignore=_ignore_pycache,
            )

        bundle_path = os.path.join(slot_dir, "task_bundle.tar.gz")
        from workflows.bundle_builder import create_task_bundle

        create_task_bundle(
            bundle_staging,
            bundle_path,
            "python3 run_volunteer_vcuy.py",
        )
        shutil.rmtree(bundle_staging, ignore_errors=True)

        input_size = max(1, os.path.getsize(bundle_path) // (1024 * 1024))

        task = Task.objects.create(
            workflow=workflow_instance,
            name=f"Gossip Volontaire {slot + 1}/{n_volunteers}",
            description=(
                f"Apprentissage distribué gossip — slot {slot}, "
                f"{max_rounds} rounds, modèle {model_name}, dataset {dataset}"
            ),
            command="python3 run_volunteer_vcuy.py",
            parameters={
                "volunteer_slot": slot,
                "dl_manager_port": dl_port,
                "paradigm": "gossip_distributed_learning",
            },
            input_files=[f"volunteer_{slot}/task_bundle.tar.gz"],
            output_files=["dl_summary.json", "model_final.pt", "stats/"],
            runtime_info=runtime_meta,
            required_resources={
                "cpu": min_resources.get("cpu_cores", min_resources.get("cpu", 2)),
                "ram": min_resources.get("memory_mb", min_resources.get("ram", 4096)),
                "disk": min_resources.get("disk", 2),
            },
            estimated_max_time=estimated_seconds,
        )
        tasks.append(task)
        split_logger.info(
            "Tâche DL slot %s créée (bundle %s Mo, manager %s:%s)",
            slot,
            input_size,
            public_host,
            dl_port,
        )

    split_logger.info(
        "Workflow DL: %s tâches gossip, manager TCP %s:%s",
        len(tasks),
        public_host,
        dl_port,
    )
    return tasks
