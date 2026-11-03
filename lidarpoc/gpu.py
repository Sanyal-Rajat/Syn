"""Minimal adaptation of PINNs/gpu_optimizer.py and sandbox_optimizer_lib.py.

Reuse: GPU discovery, VRAM budget heuristic and DistributedHandler's model wrapper.
Diffusion training uses torchrun/DDP; small reusable policies can use DataParallel.
"""
import math
import json
import os
from pathlib import Path
import subprocess
import sys

import psutil
import torch
from torch import nn


def resolve_gpus(requested=-1):
    """Resolve all GPUs visible to this process; zero explicitly selects CPU."""
    if requested == 0:
        return 0
    available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if requested == -1:
        return available
    if requested < -1 or requested > available:
        raise ValueError(f'Requested {requested} GPUs, but {available} are visible; use gpus=-1 for all or 0 for CPU')
    return requested


def loader_workers(requested, processes=1):
    if requested >= 0:
        return requested
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    return min(4, max(0, cpus // max(1, processes) - 1))


def launch_training(config, output):
    """One launcher shared by regular training and every experiment arm."""
    from .io import write_json
    from .logs import get_logger

    config = config.model_copy(update={'gpus': resolve_gpus(config.gpus)})
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    request = directory / 'request.json'
    command = [sys.executable, '-u', '-m']
    if config.gpus > 1:
        command += ['torch.distributed.run', '--standalone', f'--nproc_per_node={config.gpus}', '-m']
    command += ['lidarpoc.train', '--request', str(request), '--output', str(directory)]
    retries = []
    while True:
        for marker in directory.glob('oom-rank-*.json'):
            marker.unlink()
        write_json(request, config.model_dump())
        get_logger().info('Launching training: %d GPU(s), %d process(es), batch limit %d, auto optimize=%s',
                          config.gpus, max(1, config.gpus), config.batch_size, config.auto_optimize)
        try:
            subprocess.run(command, check=True, env={**os.environ, 'PYTHONUNBUFFERED': '1'})
            break
        except subprocess.CalledProcessError:
            failures = [json.loads(path.read_text()) for path in directory.glob('oom-rank-*.json')]
            sizes = [row['batch_size'] for row in failures]
            if (not config.auto_optimize or not sizes or min(sizes) <= 1
                    or any(row.get('phase') == 'setup' for row in failures)):
                raise
            reduced = max(1, min(config.batch_size, min(sizes)) // 2)
            retries.append({'previous_limit': config.batch_size, 'new_limit': reduced})
            write_json(directory / 'batch_backoff.json', retries)
            config = config.model_copy(update={'batch_size': reduced})
            get_logger().warning('CUDA OOM: restarting this training arm with batch limit %d', reduced)


def hardware():
    gpus = []
    for i in range(torch.cuda.device_count() if torch.cuda.is_available() else 0):
        props = torch.cuda.get_device_properties(i)
        gpus.append({'id': i, 'name': props.name, 'memory_gb': props.total_memory / 1024**3,
                     'supports_tf32': props.major >= 8})
    return {'gpus': gpus, 'physical_cpus': psutil.cpu_count(logical=False) or 1,
            'available_ram_gb': psutil.virtual_memory().available / 1024**3,
            'torch': str(torch.__version__)}


def auto_configure_training(device='auto', workers=1):
    info = hardware()
    available = [f"cuda:{gpu['id']}" for gpu in info['gpus']]
    if device == 'cpu':
        devices = ['cpu'] * min(workers, info['physical_cpus'])
    elif device == 'auto':
        devices = available[:workers] or ['cpu'] * min(workers, info['physical_cpus'])
    elif device in available:
        devices = [device]
    else:
        raise ValueError(f'Unavailable device {device}; available CUDA devices: {available}')
    minimum_memory = min((gpu['memory_gb'] for gpu in info['gpus']), default=0.)
    available_mb = max(0., minimum_memory * 1024 * .7 - 100 - 500)
    estimate = 2**int(math.log2(max(1., available_mb / 2))) if minimum_memory else 32
    return {'devices': devices, 'estimated_batch_capacity': min(65536, estimate), 'hardware': info}


class DistributedHandler:
    @staticmethod
    def distribute_model(model, enabled=False):
        if enabled and next(model.parameters()).is_cuda and torch.cuda.device_count() > 1:
            return nn.DataParallel(model)
        return model


