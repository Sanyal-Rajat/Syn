import json
from pathlib import Path
import subprocess
import sys

import torch

from lidarpoc.config import RunConfig
from lidarpoc.provenance import validate_checkpoint


def test_two_rank_training_uses_distributed_synchronization(prepared, tmp_path):
    config = RunConfig(manifest=prepared, gpus=0, channels=8, epochs=1, batch_size=1,
                       diffusion_steps=10, sample_steps=2, window=2)
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
