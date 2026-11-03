"""Spawn independent jobs on disjoint groups of visible GPUs."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os

from .logs import get_logger


def device_groups(gpus, jobs, visible=None):
    tokens = (visible if visible is not None else os.environ.get('CUDA_VISIBLE_DEVICES'))
    devices = tokens.split(',')[:gpus] if tokens else [str(index) for index in range(gpus)]
    width = min(len(devices), jobs)
    groups, offset = [], 0
    for index in range(width):
        size = len(devices) // width + (index < len(devices) % width)
        groups.append(devices[offset:offset + size])
        offset += size
    return groups


def _initialize(queue, workers):
    slot, devices = queue.get()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(devices)
    if hasattr(os, 'sched_getaffinity'):
        cores = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, cores[slot::workers] or [cores[slot % len(cores)]])
    os.environ['OMP_NUM_THREADS'] = '1'
    get_logger().info('Task worker %d: visible GPUs=%s', slot, devices)


def map_tasks(function, tasks, gpus=0, on_done=None, *, groups=None):
    tasks = list(tasks)
    if not tasks:
        return []
    groups = device_groups(gpus, len(tasks)) if groups is None else groups
    if len(groups) <= 1:
        result = []
        for task in tasks:
            result.append(function(task))
            if on_done:
                on_done(len(result), len(tasks))
        return result
    context = mp.get_context('spawn')
    queue = context.SimpleQueue()
    for slot, devices in enumerate(groups):
        queue.put((slot, devices))
    results = [None] * len(tasks)
    try:
        with ProcessPoolExecutor(max_workers=len(groups), mp_context=context,
                                 initializer=_initialize, initargs=(queue, len(groups))) as executor:
            futures = {executor.submit(function, task): index for index, task in enumerate(tasks)}
            for count, future in enumerate(as_completed(futures), 1):
                results[futures[future]] = future.result()
                if on_done:
                    on_done(count, len(tasks))
    finally:
        queue.close()
    return results
