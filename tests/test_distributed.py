import json
from pathlib import Path
import subprocess
import sys

import torch

from lidarpoc.config import RunConfig
from lidarpoc.provenance import validate_checkpoint
import pytest

from lidarpoc.gpu import launch_training, loader_workers, resolve_gpus


def test_gpu_auto_selection_and_explicit_limits(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)
    assert RunConfig(manifest='/manifest.json').gpus == -1
    assert resolve_gpus(-1) == 4
    assert resolve_gpus(2) == 2
    assert resolve_gpus(0) == 0
    with pytest.raises(ValueError, match='4 are visible'):
        resolve_gpus(5)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    assert resolve_gpus(-1) == 0
    with pytest.raises(ValueError, match='0 are visible'):
        resolve_gpus(1)


def test_shared_launcher_resolves_all_gpus_before_spawning(tmp_path, monkeypatch):
    import lidarpoc.gpu as gpu
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)
    calls = []
    monkeypatch.setattr(gpu.subprocess, 'run', lambda command, **kwargs: calls.append((command, kwargs)))
    launch_training(RunConfig(manifest='/manifest.json'), tmp_path)
    command, options = calls[-1]
    assert '--nproc_per_node=4' in command and options['check']
    assert json.loads((tmp_path / 'request.json').read_text())['gpus'] == 4
    launch_training(RunConfig(manifest='/manifest.json', gpus=0), tmp_path)
    assert 'torch.distributed.run' not in calls[-1][0]


def test_auto_loader_workers_respect_cpu_affinity_and_explicit_zero(monkeypatch):
    import lidarpoc.gpu as gpu
    monkeypatch.setattr(gpu.os, 'sched_getaffinity', lambda _: set(range(8)))
    assert loader_workers(-1, 4) == 1
    assert loader_workers(-1, 16) == 0
    assert loader_workers(0, 4) == 0


@pytest.mark.parametrize('with_ot', [False, True])
def test_two_rank_training_uses_distributed_synchronization(prepared, tmp_path, with_ot):
    manifest = json.loads(Path(prepared).read_text())
    manifest['splits']['validation'][0]['frames'] = manifest['splits']['validation'][0]['frames'][:2 if with_ot else 3]
    if with_ot:
        manifest['splits']['validation'][1]['frames'] = manifest['splits']['validation'][1]['frames'][:1]
    Path(prepared).write_text(json.dumps(manifest))
    config = RunConfig(manifest=prepared, gpus=0, channels=8, epochs=1, batch_size=1,
                       diffusion_steps=10, sample_steps=2, window=2, gradient_accumulation=3,
                       workers=-1 if with_ot else 1)
    if with_ot:
        from lidarpoc.ot.config import OTConfig
        from lidarpoc.ot.reference import build_reference
        from lidarpoc.data.split_manifest import load_manifest
        from lidarpoc.runner import read_examples
        ot = OTConfig(conditional_weight=.1, mass_weight=.1, max_points=8, min_samples=2, iterations=10)
        ot.reference = build_reference(read_examples(load_manifest(prepared), 'train', 2, 1), prepared, ot,
                                       tmp_path / 'reference.npz')
        config = config.model_copy(update={'ot': ot})
    request = tmp_path / 'request.json'
    request.write_text(config.model_dump_json())
    output = tmp_path / 'ddp'
    result = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                             '-m', 'lidarpoc.train', '--request', str(request), '--output', str(output)],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    checkpoint = torch.load(output / 'checkpoint.pt', weights_only=True)
    validate_checkpoint(checkpoint, prepared)
    assert checkpoint['epoch'] == 1
    resources = checkpoint['compute_config']
    assert resources['world_size'] == 2 and resources['effective_batch_size'] == 6
    assert resources['evaluated_validation_windows'] == (1 if with_ot else 3)
    assert 'Rank 0/2 uses cpu' in result.stdout and 'Rank 1/2 uses cpu' in result.stdout
