"""
Découpage des workflows MATRIX, ML_INFERENCE et CUSTOM.
"""

import math
import os
import pickle
import logging
from pathlib import Path

from tasks.models import Task, TaskStatus
from workflows.bundle_builder import RUNTIME_META, package_files_as_bundle

logger = logging.getLogger(__name__)

EXAMPLES = Path(__file__).resolve().parent / "examples"


def _base_input_dir(workflow_instance) -> str:
    base = workflow_instance.executable_path or workflow_instance.input_path or ''
    input_dir = os.path.join(base, 'inputs')
    os.makedirs(input_dir, exist_ok=True)
    return input_dir


def _create_task(workflow_instance, index, name, command, input_files, output_files,
                 runtime_info, required_resources, estimated_max_time, dependencies=None):
    task = Task.objects.create(
        workflow=workflow_instance,
        name=name,
        description=name,
        command=command,
        parameters=[],
        dependencies=dependencies or [],
        input_files=input_files,
        output_files=output_files,
        status=TaskStatus.CREATED,
        parent_task=None,
        is_subtask=False,
        progress=0,
        start_time=None,
        runtime_info=runtime_info or dict(RUNTIME_META),
        required_resources=required_resources,
        estimated_max_time=estimated_max_time,
    )
    if input_files:
        total_size = 0
        base = _base_input_dir(workflow_instance)
        for rel_path in input_files:
            full = os.path.join(base, rel_path)
            if os.path.isfile(full):
                total_size += os.path.getsize(full)
        task.input_size = max(1, total_size // (1024 * 1024))
    else:
        task.input_size = 1
    task.save()
    return task


def _bundle_rel_path(shard_subdir: str, name: str = "task_bundle.tar.gz") -> str:
    return f"{shard_subdir}/{name}"


def split_matrix_workflow(workflow_instance, operation: str, split_logger: logging.Logger, num_tasks: int = 4):
    """
    Découpe une opération matricielle (addition ou multiplication) en tâches par blocs de lignes.
    """
    import numpy as np

    metadata = workflow_instance.metadata or {}
    num_tasks = int(metadata.get('num_tasks', num_tasks))
    matrix_size = int(metadata.get('matrix_size', 512))
    min_resources = _get_min_resources()

    input_dir = _base_input_dir(workflow_instance)
    worker_name = "matrix_multiply.py" if operation == "multiply" else "matrix_add.py"
    worker_script = EXAMPLES / "matrix_worker" / worker_name
    command = metadata.get('command', f'python3 {worker_name}')

    rng = np.random.default_rng(int(metadata.get('seed', 42)))
    matrix_a = rng.random((matrix_size, matrix_size), dtype=np.float32)
    matrix_b = rng.random((matrix_size, matrix_size), dtype=np.float32)

    rows_per_task = max(1, math.ceil(matrix_size / num_tasks))
    tasks = []

    for i in range(num_tasks):
        start = i * rows_per_task
        end = min(matrix_size, (i + 1) * rows_per_task)
        if start >= matrix_size:
            break

        shard_dir = os.path.join(input_dir, f'shard_{i}')
        os.makedirs(shard_dir, exist_ok=True)
        data_path = os.path.join(shard_dir, 'data.pkl')

        with open(data_path, 'wb') as f:
            pickle.dump({
                'A': matrix_a[start:end, :],
                'B': matrix_b[start:end, :],
                'operation': operation,
                'row_start': start,
                'row_end': end,
                'full_size': matrix_size,
            }, f)

        bundle_rel = _bundle_rel_path(f'shard_{i}')
        bundle_path = os.path.join(input_dir, bundle_rel)
        package_files_as_bundle(
            files=[data_path],
            command=command,
            bundle_path=bundle_path,
            worker_scripts=[worker_script] if worker_script.is_file() else None,
        )

        block_rows = end - start
        ram_mb = max(512, int(block_rows * matrix_size * 4 * 3 / (1024 * 1024)))

        task = _create_task(
            workflow_instance,
            i,
            name=f'Matrix {operation} block {i}',
            command=command,
            input_files=[bundle_rel],
            output_files=['result.pkl'],
            runtime_info=dict(RUNTIME_META),
            required_resources={
                'cpu': min_resources['min_cpu'],
                'ram': max(min_resources['min_ram'], ram_mb),
                'disk': max(min_resources['disk'], 1),
            },
            estimated_max_time=int(metadata.get('estimated_max_time', 120)),
        )
        tasks.append(task)
        split_logger.info('Tâche matrice %d créée (%d lignes, bundle)', i, block_rows)

    workflow_instance.tasks.add(*tasks)
    workflow_instance.save()
    return tasks


def split_ml_inference_workflow(workflow_instance, split_logger: logging.Logger):
    """Découpe un workflow d'inférence ML en lots (batches)."""
    metadata = workflow_instance.metadata or {}
    num_tasks = int(metadata.get('num_tasks', 4))
    samples_per_task = int(metadata.get('samples_per_task', 256))
    min_resources = _get_min_resources()

    input_dir = _base_input_dir(workflow_instance)
    command = metadata.get('command', 'python3 inference_batch.py')
    if command.startswith('python '):
        command = 'python3 ' + command[len('python '):]
    model_file = metadata.get('model_file', 'model.pth')
    model_abs = os.path.join(input_dir, model_file) if model_file else None

    tasks = []
    for i in range(num_tasks):
        shard_dir = os.path.join(input_dir, f'batch_{i}')
        os.makedirs(shard_dir, exist_ok=True)
        data_path = os.path.join(shard_dir, 'batch.pkl')

        with open(data_path, 'wb') as f:
            pickle.dump({
                'batch_id': i,
                'samples': samples_per_task,
                'model_file': model_file,
                'offset': i * samples_per_task,
            }, f)

        files = [data_path]
        if model_abs and os.path.isfile(model_abs):
            files.append(model_abs)

        bundle_rel = _bundle_rel_path(f'batch_{i}')
        bundle_path = os.path.join(input_dir, bundle_rel)
        package_files_as_bundle(
            files=files,
            command=command,
            bundle_path=bundle_path,
        )

        task = _create_task(
            workflow_instance,
            i,
            name=f'Inference batch {i}',
            command=command,
            input_files=[bundle_rel],
            output_files=['predictions.json'],
            runtime_info=dict(RUNTIME_META),
            required_resources={
                'cpu': max(1, min_resources['min_cpu']),
                'ram': max(min_resources['min_ram'], 1024),
                'disk': max(min_resources['disk'], 2),
                'gpu': metadata.get('gpu_required', False),
            },
            estimated_max_time=int(metadata.get('estimated_max_time', 180)),
        )
        tasks.append(task)
        split_logger.info('Tâche inférence %d créée (bundle)', i)

    workflow_instance.tasks.add(*tasks)
    workflow_instance.save()
    return tasks


def split_custom_workflow(workflow_instance, split_logger: logging.Logger):
    """
    Découpe un workflow CUSTOM à partir de metadata.tasks ou commande réelle (bundle vc-uyr).
    """
    from workflows.custom_validation import validate_custom_metadata

    metadata = workflow_instance.metadata or {}
    ok, err, metadata = validate_custom_metadata(metadata)
    if not ok:
        raise ValueError(err)
    workflow_instance.metadata = metadata
    workflow_instance.save(update_fields=['metadata', 'updated_at'])

    min_resources = _get_min_resources()
    task_specs = metadata.get('tasks', [])

    if not task_specs:
        num_tasks = int(metadata.get('num_tasks', 1))
        base_command = metadata['command']
        task_specs = [
            {'name': f'Custom task {i}', 'command': base_command}
            for i in range(num_tasks)
        ]

    runtime_meta = dict(RUNTIME_META)
    meta_runtime = metadata.get('runtime_info') or {}
    if isinstance(meta_runtime, dict) and meta_runtime.get('runtime') == 'vc-uyr':
        runtime_meta = {**RUNTIME_META, **meta_runtime}
    tasks = []
    input_dir = _base_input_dir(workflow_instance)

    for i, spec in enumerate(task_specs):
        name = spec.get('name', f'Custom task {i}')
        command = spec.get('command', 'true')
        if isinstance(command, str) and command.startswith('python '):
            command = 'python3 ' + command[len('python '):]

        # Bundle self-contained à partir des fichiers déclarés (s'ils existent)
        shard_dir = os.path.join(input_dir, f'custom_{i}')
        os.makedirs(shard_dir, exist_ok=True)
        source_files = []
        for rel in (spec.get('input_files') or []):
            full = os.path.join(input_dir, rel)
            if os.path.isfile(full):
                source_files.append(full)
        bundle_rel = _bundle_rel_path(f'custom_{i}')
        bundle_path = os.path.join(input_dir, bundle_rel)
        package_files_as_bundle(
            files=source_files,
            command=command,
            bundle_path=bundle_path,
        )

        task = _create_task(
            workflow_instance,
            i,
            name=name,
            command=command,
            input_files=[bundle_rel],
            output_files=spec.get('output_files', []),
            runtime_info=spec.get('runtime_info') or runtime_meta,
            required_resources=spec.get('required_resources', {
                'cpu': min_resources['min_cpu'],
                'ram': min_resources['min_ram'],
                'disk': min_resources['disk'],
            }),
            estimated_max_time=int(spec.get('estimated_max_time', metadata.get('estimated_max_time', 300))),
            dependencies=spec.get('dependencies', []),
        )
        if spec.get('parameters'):
            task.parameters = spec['parameters']
            task.save(update_fields=['parameters'])
        tasks.append(task)
        split_logger.info('Tâche custom %s créée', name)

    workflow_instance.tasks.add(*tasks)
    workflow_instance.save()
    return tasks


def _get_min_resources():
    from workflows.split_workflow import get_min_volunteer_resources
    return get_min_volunteer_resources()
