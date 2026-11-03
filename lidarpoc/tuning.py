"""Measure useful throughput within a GPU memory budget, without changing weights."""
import torch

from .logs import get_logger


def search_batch(probe, limit, memory_budget):
    measurements, best, rate = [], 0, 0.

    def measure(size):
        nonlocal best, rate
        try:
            seconds, memory = probe(size)
        except torch.cuda.OutOfMemoryError:
            measurements.append({'batch_size': size, 'out_of_memory': True})
            return False
        current = size / max(seconds, 1e-9)
        measurements.append({'batch_size': size, 'samples_per_second': current, 'peak_bytes': memory})
        if memory > memory_budget:
            return False
        if current > rate * 1.02:
            best, rate = size, current
        return True

    size, lower, upper = 1, 0, None
    while size <= limit:
        if not measure(size):
            upper = size
            break
        lower = size
        if size == limit:
            break
        size = min(limit, size * 2)
    if upper is not None and lower:
        for _ in range(3):
            if upper - lower <= 1:
                break
            middle = (lower + upper) // 2
            if measure(middle):
                lower = middle
            else:
                upper = middle
    if not best:
        raise torch.cuda.OutOfMemoryError('A single training window exceeds the automatic GPU memory budget')
    get_logger().info('Automatic batch tuning: selected %d; measurements=%s', best, measurements)
    return best, measurements
