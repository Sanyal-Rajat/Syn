import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pytest
import torch

from lidarpoc.config import RunConfig
from lidarpoc.data.gpu_projection import project as torch_project
from lidarpoc.data.scan_to_range import project
from lidarpoc.gpu import launch_training
from lidarpoc.scheduling import device_groups, map_tasks
from lidarpoc.tuning import search_batch


def scheduled(value):
    time.sleep(.05)
    if value < 0:
        raise ValueError('task failed')
    return value, os.getpid(), os.environ.get('CUDA_VISIBLE_DEVICES')


def frozen_test(task):
    from lidarpoc.experiment_tasks import test as evaluate
    root = Path(task['root'])
    plan = json.loads((root / 'selection.json').read_text())
    assert plan['controllers_frozen'] and not plan['test_arrays_used_for_fitting']
    for seed in (0, 1):
        for method in ('fixed', 'ppo'):
            assert (root / 'experiments' / f'seed_{seed}' / f'E5_{method}' / 'controller.pt').is_file()
    return evaluate(task)


def test_disjoint_groups_and_spawned_task_queue_preserve_order_and_errors(monkeypatch):
    assert device_groups(4, 2, 'GPU-a,GPU-b,GPU-c,GPU-d') == [['GPU-a', 'GPU-b'], ['GPU-c', 'GPU-d']]
    assert device_groups(4, 3, '2,3,5,6') == [['2', '3'], ['5'], ['6']]
    before = os.environ.get('CUDA_VISIBLE_DEVICES')
    result = map_tasks(scheduled, range(6), groups=[['cpu-a'], ['cpu-b']])
    assert [value[0] for value in result] == list(range(6))
    assert len({value[1] for value in result}) == 2
    assert {value[2] for value in result} == {'cpu-a', 'cpu-b'}
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == before
    with pytest.raises(ValueError, match='task failed'):
        map_tasks(scheduled, [-1, 2], groups=[['cpu-a'], ['cpu-b']])


def test_batch_tuner_chooses_throughput_and_respects_memory_and_oom():
    speeds = {1: 10., 2: 15., 4: 30., 8: 20.}
    size, rows = search_batch(lambda n: (n / speeds[n], n * 100), 8, 1000)
    assert size == 4 and len(rows) == 4
    size, rows = search_batch(lambda n: (.1, n * 100), 8, 450)
    assert size == 4 and any(row.get('peak_bytes', 0) > 450 for row in rows)

    def out_of_memory(n):
        if n > 2:
            raise torch.cuda.OutOfMemoryError('simulated')
        return .1, n * 100

    size, rows = search_batch(out_of_memory, 8, 1000)
    assert size == 2 and rows[-1]['out_of_memory']


def test_training_launcher_backs_off_only_after_reported_cuda_oom(tmp_path, monkeypatch):
    import lidarpoc.gpu as gpu
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    attempts = []

    def execute(command, **kwargs):
        attempts.append(json.loads((tmp_path / 'request.json').read_text()))
        if len(attempts) == 1:
            (tmp_path / 'oom-rank-0.json').write_text('{"batch_size": 8}')
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(gpu.subprocess, 'run', execute)
    launch_training(RunConfig(manifest='/manifest.json', gpus=1, batch_size=16), tmp_path)
    assert [row['batch_size'] for row in attempts] == [16, 4]
    assert (tmp_path / 'batch_backoff.json').is_file()
    monkeypatch.setattr(gpu.subprocess, 'run', lambda command, **kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(2, command)))
    with pytest.raises(subprocess.CalledProcessError):
        launch_training(RunConfig(manifest='/manifest.json', gpus=1), tmp_path)


def test_torch_projection_preserves_nearest_indices_intensity_and_transform():
    rng = np.random.default_rng(12)
    beams = np.deg2rad([-12., -4., 4., 12.])
    yaw = rng.uniform(-np.pi, np.pi, 500)
    angle = rng.choice(beams, 500)
    distance = rng.uniform(1, 80, 500)
    xyz = np.column_stack((distance * np.cos(angle) * np.cos(yaw),
                           distance * np.cos(angle) * np.sin(yaw), distance * np.sin(angle)))
    xyz = np.r_[xyz, xyz[:5], 2 * xyz[:5], np.zeros((1, 3))]
    intensity = rng.uniform(0, 255, len(xyz))
    transform = np.eye(4)
    transform[:3, 3] = [1., -2., .1]
    expected = project(xyz + transform[:3, 3], intensity, beams, width=16, return_indices=True)
    actual = torch_project(xyz, intensity, beams, width=16, return_indices=True, device='cpu', transform=transform)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-6)
    empty = torch_project(np.zeros((1, 3)), None, beams, device='cpu', return_indices=True)
    assert not empty['valid'].any() and (empty['point_index'] == -1).all()


def test_parallel_experiment_phases_freeze_all_controllers_before_test(prepared, tmp_path, monkeypatch):
    import lidarpoc.scheduling as scheduling
    import lidarpoc.experiment_tasks as tasks
    from lidarpoc.runner import run

    original = scheduling.map_tasks
    monkeypatch.setattr(scheduling, 'map_tasks', lambda fn, jobs, gpus=0, on_done=None, **kw:
        original(fn, jobs, gpus, on_done, groups=[['cpu-a'], ['cpu-b']]))
    monkeypatch.setattr(tasks, 'test', frozen_test)
    config = RunConfig(action='experiment', manifest=prepared, gpus=0, channels=8, epochs=1,
        batch_size=2, window=2, diffusion_steps=10, sample_steps=2, samples_per_trace=1, budget=6,
        ot={'max_points': 8, 'min_samples': 2, 'iterations': 5},
        experiment={'ablations': ['E0', 'E4', 'E5'], 'seeds': [0, 1], 'controllers': ['fixed', 'ppo'], 'synthetic_scenes': 1})
    report = run(config, tmp_path / 'parallel-experiment')
    assert report['selection']['controllers_frozen']
    assert set(report['compute']) == {0, 1}
    for seed in (0, 1):
        assert set(report['compute'][seed]) == {'E0', 'E4', 'E5_fixed', 'E5_ppo'}
