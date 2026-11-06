"""Independent seeded diffusion samples, concurrently on one replica per device."""
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from contextlib import nullcontext
from copy import deepcopy
import time
from threading import Lock

import torch

from .logs import get_logger


class SamplingPool:
    def __init__(self, model, devices=()):
        self.model = model.eval().requires_grad_(False)
        self.devices = list(devices) or [next(model.parameters()).device]
        self.models = [self.model]
        self.capacity = [1] * len(self.devices)
        self.rates = [0.] * len(self.devices)
        self.best_batch = [1] * len(self.devices)
        self.ceiling = [None] * len(self.devices)

    @staticmethod
    def combine(requests):
        first = requests[0]
        if len(requests) == 1:
            return first
        for request in requests:
            if request.get('batch_size') != 1 or any(request.get(key) != first.get(key)
                    for key in ('steps', 'guidance', 'force_cfg')):
                return None
            a, b = request.get('conditioning'), first.get('conditioning')
            if (a is None) != (b is None) or (a is not None and a.get('complete_scene', True) != b.get('complete_scene', True)):
                return None
        combined = {**first, 'batch_size': len(requests), 'seed': [request.get('seed', 0) for request in requests]}
        if first.get('conditioning') is not None:
            combined['conditioning'] = {**first['conditioning'], **{
                key: torch.cat([request['conditioning'][key] for request in requests]) for key in ('range', 'valid')}}
        return combined

    def _sample(self, rank, pending, total, lock):
        model = self.models[rank]
        device = next(model.parameters()).device
        context = torch.cuda.device(device) if device.type == 'cuda' else nullcontext()
        results = []
        with context, torch.inference_mode():
            dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
            automatic = (device.type == 'cuda' and getattr(model, 'supports_seed_batch', False)
                         and model.config.get('auto_optimize', True))
            limit = min(model.config.get('batch_size') or total, self.ceiling[rank] or total, total) if automatic else 1
            while True:
                with lock:
                    if not pending:
                        break
                    size = min(self.capacity[rank], limit, len(pending))
                    chunk = [pending.popleft() for _ in range(size)]
                request = self.combine([item[1] for item in chunk])
                if request is None:
                    with lock:
                        pending.extendleft(reversed(chunk[1:]))
                    size, chunk = 1, chunk[:1]
                    request = chunk[0][1]
                precision = torch.autocast('cuda', dtype=dtype) if device.type == 'cuda' else nullcontext()
                started = time.perf_counter()
                try:
                    with precision:
                        sample = model.sample(**request)
                    completed = [(index, {key: (tensor if size == 1 else tensor[offset:offset + 1]).cpu()
                                          for key, tensor in sample.items()}) for offset, (index, _) in enumerate(chunk)]
                    results.extend(completed)
                    for index, original in chunk:
                        get_logger().info('Generated sample %d/%d on %s; seed=%d', index + 1, total, device, original.get('seed', 0))
                    del sample
                except torch.cuda.OutOfMemoryError:
                    if size == 1:
                        raise
                    limit = self.ceiling[rank] = min(limit, max(1, size // 2))
                    self.capacity[rank] = max(1, min(limit, self.best_batch[rank]))
                    with lock:
                        pending.extendleft(reversed(chunk))
                    torch.cuda.empty_cache()
                    get_logger().warning('Sampling OOM on %s: retrying batch size %d', device, self.capacity[rank])
                    continue
                if automatic:
                    rate = size / max(time.perf_counter() - started, 1e-9)
                    free, memory = torch.cuda.mem_get_info(device)
                    if rate > self.rates[rank] * 1.02:
                        self.best_batch[rank] = size
                    if size == self.capacity[rank] and rate >= self.rates[rank] * .98 and free > .15 * memory:
                        self.rates[rank] = max(rate, self.rates[rank])
                        self.capacity[rank] = min(limit, size * 2)
                    elif size > 1 and rate < self.rates[rank] * .98:
                        self.capacity[rank] = min(limit, self.best_batch[rank])
                    get_logger().info('Sampling throughput on %s: batch=%d, samples/s=%.3f, next batch=%d',
                                      device, size, rate, self.capacity[rank])
        return results

    def sample_many(self, requests):
        if not requests:
            return []
        workers = min(len(self.devices), len(requests))
        while len(self.models) < workers:
            self.models.append(deepcopy(self.model).to(self.devices[len(self.models)]))
        pending, lock = deque(enumerate(requests)), Lock()
        if workers == 1:
            results = self._sample(0, pending, len(requests), lock)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(self._sample, rank, pending, len(requests), lock)
                           for rank in range(workers)]
                results = [result for future in futures for result in future.result()]
        return [sample for _, sample in sorted(results, key=lambda item: item[0])]

    def close(self):
        self.models.clear()
        for device in self.devices:
            device = torch.device(device)
            if device.type == 'cuda':
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
