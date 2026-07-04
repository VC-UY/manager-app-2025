"""
Découpage des workflows MATRIX, ML_INFERENCE et CUSTOM.
"""

import math
import os
import pickle
import logging

from tasks.models import Task, TaskStatus

logger = logging.getLogger(__name__)


def _base_input_dir(workflow_instance) -> str:
    base = workflow_instance.executable_path or workflow_instance.input_path or ''
    input_dir = os.path.join(base, 'inputs')
    os.makedirs(input_dir, exist_ok=True)
    return input_dir


def _create_task(workflow_instance, index, name, command, input_files, output_files,
                 docker_info, required_resources, estimated_max_time, dependencies=None):
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
        docker_info=docker_info,
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
    docker_img = metadata.get('docker_info', {'name': 'vcuy-matrix', 'tag': 'latest'})
    command = metadata.get('command', f'python matrix_{operation}.py')

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
        rel_data = f'shard_{i}/data.pkl'
        data_path = os.path.join(input_dir, rel_data)

        with open(data_path, 'wb') as f:
            pickle.dump({
                'A': matrix_a[start:end, :],
                'B': matrix_b[start:end, :],
                'operation': operation,
                'row_start': start,
                'row_end': end,
                'full_size': matrix_size,
            }, f)

        block_rows = end - start
        ram_mb = max(512, int(block_rows * matrix_size * 4 * 3 / (1024 * 1024)))

        task = _create_task(
            workflow_instance,
            i,
            name=f'Matrix {operation} block {i}',
            command=command,
            input_files=[rel_data],
            output_files=[f'shard_{i}/output/result.pkl'],
            docker_info=docker_img,
            required_resources={
                'cpu': min_resources['min_cpu'],
                'ram': max(min_resources['min_ram'], ram_mb),
                'disk': max(min_resources['disk'], 1),
            },
            estimated_max_time=int(metadata.get('estimated_max_time', 120)),
        )
        tasks.append(task)
        split_logger.info('Tâche matrice %d créée (%d lignes)', i, block_rows)

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
    docker_img = metadata.get('docker_info', {'name': 'vcuy-inference', 'tag': 'latest'})
    command = metadata.get('command', 'python inference_batch.py')
    model_file = metadata.get('model_file', 'model.pth')

    tasks = []
    for i in range(num_tasks):
        shard_dir = os.path.join(input_dir, f'batch_{i}')
        os.makedirs(shard_dir, exist_ok=True)
        rel_data = f'batch_{i}/batch.pkl'
        data_path = os.path.join(input_dir, rel_data)

        with open(data_path, 'wb') as f:
            pickle.dump({
                'batch_id': i,
                'samples': samples_per_task,
                'model_file': model_file,
                'offset': i * samples_per_task,
            }, f)

        task = _create_task(
            workflow_instance,
            i,
            name=f'Inference batch {i}',
            command=command,
            input_files=[rel_data, model_file] if model_file else [rel_data],
            output_files=[f'batch_{i}/output/predictions.json'],
            docker_info=docker_img,
            required_resources={
                'cpu': max(1, min_resources['min_cpu']),
                'ram': max(min_resources['min_ram'], 1024),
                'disk': max(min_resources['disk'], 2),
                'gpu': metadata.get('gpu_required', False),
            },
            estimated_max_time=int(metadata.get('estimated_max_time', 180)),
        )
        tasks.append(task)
        split_logger.info('Tâche inférence %d créée', i)

    workflow_instance.tasks.add(*tasks)
    workflow_instance.save()
    return tasks


def split_custom_workflow(workflow_instance, split_logger: logging.Logger):
    """
    Découpe un workflow CUSTOM à partir de metadata.tasks ou commande + image Docker réelles.
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

    docker_img = metadata.get('docker_info')
    tasks = []

    for i, spec in enumerate(task_specs):
        name = spec.get('name', f'Custom task {i}')
        task = _create_task(
            workflow_instance,
            i,
            name=name,
            command=spec.get('command', 'true'),
            input_files=spec.get('input_files', []),
            output_files=spec.get('output_files', []),
            docker_info=spec.get('docker_info', docker_img),
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
