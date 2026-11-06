"""Data-aware batch bounds and measured throughput, adapted from PINNs' GPU optimizer."""
import psutil
import torch

from .logs import get_logger


def batch_limit(sample, windows, workers, processes=1, requested=0):
    """Bound probes by real window bytes, local data and shared host memory."""
    window_bytes = sum(value.numel() * value.element_size() for value in sample.values()
                       if isinstance(value, torch.Tensor))
    # Decoded arrays, collation, pinned copies and two prefetched batches per worker.
    copies = 4 + 4 * workers
    host_budget = int(psutil.virtual_memory().available * .5 / max(1, processes))
    host_limit = max(1, host_budget // max(1, window_bytes * copies))
    limit = min(windows, host_limit, requested or windows)
    return max(1, limit), {'window_shape': list(sample['range'].shape[1:]),
                           'window_bytes': window_bytes, 'windows_per_rank': windows,
                           'host_memory_budget_bytes': host_budget,
                           'host_batch_capacity': host_limit, 'search_limit': limit,
                           'requested_limit': requested or None}


def search_batch(probe, limit, memory_budget):
    measurements, best, rate = [], 0, 0.

    def measure(size):
        nonlocal best, rate
        try:
            seconds, memory = probe(size)
        except torch.cuda.OutOfMemoryError:
            measurements.append({'batch_size': size, 'out_of_memory': True})
            get_logger().info('Batch probe %d: CUDA memory exhausted', size)
            return False
        current = size / max(seconds, 1e-9)
        measurements.append({'batch_size': size, 'samples_per_second': current, 'peak_bytes': memory,
                             'within_memory_budget': memory <= memory_budget})
        get_logger().info('Batch probe %d: %.3f windows/s, peak %.2f GiB, budget %.2f GiB',
                          size, current, memory / 1024**3, memory_budget / 1024**3)
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
