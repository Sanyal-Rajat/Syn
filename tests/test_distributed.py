import json
from pathlib import Path
import subprocess
import sys

import torch

from lidarpoc.config import RunConfig
from lidarpoc.provenance import validate_checkpoint
import pytest


@pytest.mark.parametrize('with_ot', [False, True])
def test_two_rank_training_uses_distributed_synchronization(prepared, tmp_path, with_ot):
    config = RunConfig(manifest=prepared, gpus=0, channels=8, epochs=1, batch_size=1,
                       diffusion_steps=10, sample_steps=2, window=2)
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
