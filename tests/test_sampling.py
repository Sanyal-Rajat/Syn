from threading import Barrier
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lidarpoc.config import RunConfig
from lidarpoc.data.split_manifest import load_manifest
from lidarpoc.models.lidardm_adapter import LidarDMAdapter
from lidarpoc.ot.controllers import Renderer, controlled_bank
from lidarpoc.sampling import SamplingPool


class ConcurrentModel(torch.nn.Module):
    barrier = None

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def sample(self, batch_size, seed):
        if seed < 0:
            raise ValueError('failed sample')
        if self.barrier:
            self.barrier.wait(timeout=10)
        return {'range': torch.rand(batch_size, generator=torch.Generator().manual_seed(seed))}


def test_samples_overlap_preserve_order_and_propagate_worker_errors(monkeypatch):
    model = ConcurrentModel()
    pool = SamplingPool(model, ['cpu', 'cpu'])
    requests = [{'batch_size': 1, 'seed': seed} for seed in (7, 1, 9, 2)]
    expected = [model.sample(**request)['range'] for request in requests]
    monkeypatch.setattr(ConcurrentModel, 'barrier', Barrier(2))
    actual = pool.sample_many(requests)
    assert all(torch.equal(row['range'], target) for row, target in zip(actual, expected))
    assert pool.sample_many([]) == []
    monkeypatch.setattr(ConcurrentModel, 'barrier', None)
    with pytest.raises(ValueError, match='failed sample'):
        pool.sample_many([{'batch_size': 1, 'seed': -1}, requests[0]])
    pool.close()


def test_adaptive_sampling_retries_oom_without_losing_or_duplicating_seeds(monkeypatch):
    import lidarpoc.sampling as sampling

    calls = []

    class DeviceModel:
        supports_seed_batch = True
        config = {'auto_optimize': True, 'batch_size': 8}

        def eval(self):
            return self

        def requires_grad_(self, enabled):
            return self

        def parameters(self):
            return iter([SimpleNamespace(device=torch.device('cuda:0'))])

        def sample(self, batch_size, seed):
            calls.append(batch_size)
            if batch_size > 2:
                raise torch.cuda.OutOfMemoryError('simulated sampling OOM')
            return {'range': torch.tensor(seed if isinstance(seed, list) else [seed])[:, None]}

    monkeypatch.setattr(torch.cuda, 'device', lambda *args: nullcontext())
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda: True)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda *args: (900, 1000))
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    monkeypatch.setattr(torch, 'autocast', lambda *args, **kwargs: nullcontext())
    ticks = iter(range(10000))
    monkeypatch.setattr(sampling.time, 'perf_counter', lambda: next(ticks))
    pool = SamplingPool(DeviceModel())
    result = pool.sample_many([{'batch_size': 1, 'seed': seed} for seed in range(11)])
    assert [row['range'].item() for row in result] == list(range(11))
    assert 4 in calls and all(size <= 2 for size in calls[calls.index(3) + 1:])
    assert pool.ceiling == [2]
    assert pool.capacity == [2]
    pool.close()


def test_parallel_diffusion_and_renderer_match_serial_seeds_and_geometry(prepared):
    torch.set_num_threads(1)
    sensor = load_manifest(prepared)['sensor_meta']
    config = RunConfig(manifest=prepared, gpus=0, channels=8, diffusion_steps=10, sample_steps=2)
    model = LidarDMAdapter(config.model_dump(), sensor)
    serial, parallel = SamplingPool(model), SamplingPool(model, ['cpu', 'cpu'])
    scene = {'range': np.full((2, 4, 16), 10., dtype='float32'),
             'valid': np.ones((2, 4, 16), dtype=bool)}
    scene['valid'][:, :, 0] = False
    conditioning = {key: torch.from_numpy(value)[None] for key, value in scene.items()}
    requests = [{'batch_size': 1, 'seed': seed, 'conditioning': conditioning} for seed in (3, 8, 6)]
    expected, actual = serial.sample_many(requests), parallel.sample_many(requests)
    for left, right in zip(expected, actual):
        for key in left:
            assert torch.equal(left[key], right[key])
    combined = SamplingPool.combine(requests)
    batched = model.sample(**combined)
    for index, left in enumerate(expected):
        for key in left:
            torch.testing.assert_close(left[key], batched[key][index:index + 1], atol=2e-5, rtol=2e-5)
    assert SamplingPool.combine([requests[0], {**requests[1], 'guidance': 2.}]) is None
    with pytest.raises(ValueError, match='one seed per sample'):
        model.sample(2, seed=[0])
    a, b = Renderer(model, [scene], serial), Renderer(model, [scene], parallel)
    expected, _ = controlled_bank(a, [scene], 3, 18)
    actual, _ = controlled_bank(b, [scene], 3, 18)
    assert a.denoiser_evaluations == b.denoiser_evaluations == 12
    for left, right in zip(expected, actual):
        assert not right['valid'][:, :, 0].any()
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])
    parallel.close()
