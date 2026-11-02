"""Apply an exported, frozen controller only to its own checkpoint/manifest."""
from pathlib import Path

import numpy as np
import torch

from .config import OTConfig
from .controllers import Renderer, controlled_bank
from ..data.split_manifest import digest
from ..data.synthetic_scene import load_scene


def controlled_generate(model, config):
    bundle = torch.load(Path(config.controller_file).expanduser(), map_location='cpu', weights_only=True)
    if (bundle.get('root_initialization') != 'random'
            or bundle.get('checkpoint_sha256') != digest(Path(config.checkpoint).expanduser())
            or bundle.get('manifest_sha256') != digest(config.manifest)):
        raise ValueError('Generation controller does not match this own checkpoint and frozen manifest')
    scene = load_scene(config.conditioning, model.sensor_meta, model.config['window'])
    renderer = Renderer(model, [scene])
    renderer.reference = {key: np.asarray(value) for key, value in bundle['reference'].items()}
    renderer.ot = OTConfig(**bundle['ot_config'])
    samples, _ = controlled_bank(renderer, [scene], config.samples_per_trace, config.seed, bundle['profile'], bundle['policy'])
    return samples
