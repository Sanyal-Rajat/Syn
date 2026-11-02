"""Minimal adaptation of PINNs/gpu_optimizer.py and sandbox_optimizer_lib.py.

Reuse: GPU discovery, VRAM budget heuristic and DistributedHandler's model wrapper.
Diffusion training uses torchrun/DDP; small reusable policies can use DataParallel.
"""
import math

import psutil
import torch
from torch import nn


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


