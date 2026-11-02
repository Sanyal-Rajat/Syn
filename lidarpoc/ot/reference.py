"""Persist bounded training-only features with immutable split/config provenance."""
import json
from pathlib import Path

import numpy as np
import torch

from ..data.split_manifest import digest, load_manifest
from .features import features, return_mass, select


def specification(config):
    return config.model_dump(exclude={'reference', 'conditional_weight', 'mass_weight', 'physics_weight', 'regularization_weight'})


def tensor_sample(sample, device='cpu'):
    return {key: torch.as_tensor(np.asarray(sample[key]), dtype=torch.float32, device=device)
            for key in ('range', 'valid', 'intensity')}


def stack_samples(samples, device='cpu'):
    tensors = [tensor_sample(sample, device) for sample in samples]
    return {key: torch.stack([sample[key] for sample in tensors])
            for key in ('range', 'valid', 'intensity')}


def build_reference(samples, manifest_path, config, path):
    manifest = load_manifest(manifest_path)
    allowed = {trace['sequence_id'] for trace in manifest['splits']['train']}
    identities = {sample['sequence_id'] for sample in samples}
    if not identities or not identities <= allowed:
        raise ValueError('OT references must contain training traces only')
    if config.feature_weights[1] and not all(bool(np.asarray(sample.get('has_intensity', False)).all()) for sample in samples):
        raise ValueError('OT intensity requires observed calibration-domain intensity; set its feature weight to zero if unavailable')
    if config.range_edges_m[-1] < manifest['sensor_meta']['max_range_m']:
        raise ValueError('Fixed OT range strata must cover the sensor maximum range')
    sample = stack_samples(samples)
    vectors, bins, valid = features(sample, manifest['sensor_meta'], config)
    selected = select(vectors, bins, valid, config.max_points)
    if not any(len(value) >= config.min_samples for value in selected.values()):
        raise ValueError('Training data has no supported OT reference bins')
    centers, mass = return_mass(sample, manifest['sensor_meta'], config)
    metadata = {'split': 'train', 'sequence_ids': sorted(identities), 'manifest_sha256': digest(manifest_path),
                'specification': specification(config), 'mass_units': 'returns / fixed ray exposure',
                'features': ['range_m', 'normalized_intensity', 'elevation_rad', 'local_return_rate', 'sensor_frame_persistence']}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=torch.cat(list(selected.values())).numpy(),
                        bins=np.concatenate([np.full(len(value), key) for key, value in selected.items()]),
                        centers=centers.numpy(), mass=mass.numpy(), metadata=np.asarray(json.dumps(metadata)))
    return str(Path(path).resolve())


def load_reference(path, manifest_path, config):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    metadata = json.loads(str(data.pop('metadata')))
    allowed = {trace['sequence_id'] for trace in load_manifest(manifest_path)['splits']['train']}
    if (metadata['split'] != 'train' or not set(metadata['sequence_ids']) <= allowed
            or metadata['manifest_sha256'] != digest(manifest_path)
            or metadata['specification'] != specification(config)):
        raise ValueError('OT reference split, manifest or fixed feature specification differs')
    return data, metadata
