"""Persist bounded training-only features with immutable split/config provenance."""
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from ..data.split_manifest import digest, load_manifest, load_window, windows
from ..logs import get_logger
from .features import features, return_mass, select


def specification(config):
    return config.model_dump(exclude={'reference', 'conditional_weight', 'mass_weight', 'physics_weight',
        'regularization_weight', 'reference_windows', 'generation_enabled', 'generation_steps', 'generation_lr',
        'range_weight', 'generation_range_weight', 'checkpoint_metric', 'range_validation_windows'})


def prepare_reference(config, directory, manifest=None, required=False):
    """Build/reuse a bounded training-only reference before any training or generation."""
    ot = config.ot
    training = config.action == 'train' and any((ot.conditional_weight, ot.mass_weight,
                                               ot.physics_weight, ot.regularization_weight))
    if not (required or ot.generation_enabled or training):
        return config
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'ot_reference.npz'
    source = Path(ot.reference).expanduser() if ot.reference else target
    if source.is_file():
        try:
            data, metadata = load_reference(source, config.manifest, ot)
            if metadata.get('window', config.window) != config.window:
                raise ValueError('OT reference temporal window differs from the model')
        except ValueError:
            if ot.reference:
                raise
        else:
            if 'range_values' in data or not ot.generation_range_weight:
                if source.resolve() != target:
                    shutil.copy2(source, target)
                get_logger().info('Using training-only OT reference: %s', target)
                return config.model_copy(update={'ot': ot.model_copy(update={'reference': str(target)})})
            get_logger().info('Rebuilding legacy reference with a training-only range distribution')
    elif ot.reference:
        get_logger().warning('Saved OT reference is unavailable; rebuilding from the frozen training split')
    manifest = manifest or load_manifest(config.manifest)
    records = windows(manifest, 'train', config.window)
    if not records:
        raise ValueError('No complete training windows are available for the OT reference')
    indices = np.linspace(0, len(records) - 1, min(len(records), ot.reference_windows), dtype=int)
    samples = [load_window(records[index][1], manifest['sensor_meta'], records[index][0]) for index in indices]
    observed = all(bool(np.asarray(sample.get('has_intensity', False)).all()) for sample in samples)
    if not observed and ot.feature_weights[1]:
        weights = list(ot.feature_weights)
        weights[1] = 0.
        if not any(weights):
            raise ValueError('The only weighted OT feature is unobserved intensity; enable an observed feature such as range')
        ot = ot.model_copy(update={'feature_weights': weights})
        get_logger().info('OT intensity alignment disabled: the reference has no complete observed intensity channel')
    reference = build_reference(samples, config.manifest, ot, target)
    get_logger().info('Built OT reference from %d training windows: %s', len(samples), reference)
    return config.model_copy(update={'ot': ot.model_copy(update={'reference': reference})})


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
    ranges = sample['range'][sample['valid'].bool() & torch.isfinite(sample['range']) & (sample['range'] > 0)].numpy()
    ranges.sort()
    if not len(ranges):
        raise ValueError('No finite observed training ranges are available for the distance reference')
    range_values = ranges if len(ranges) <= 4096 else np.quantile(ranges, (np.arange(4096) + .5) / 4096).astype('float32')
    metadata = {'split': 'train', 'sequence_ids': sorted(identities), 'manifest_sha256': digest(manifest_path),
                'specification': specification(config), 'mass_units': 'returns / fixed ray exposure',
                'window': int(samples[0]['range'].shape[0]), 'window_count': len(samples),
                'range_observations': len(ranges), 'range_reference_points': len(range_values),
                'intensity_observed': all(bool(np.asarray(sample.get('has_intensity', False)).all()) for sample in samples),
                'features': ['range_m', 'normalized_intensity', 'elevation_rad', 'local_return_rate', 'sensor_frame_persistence']}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=torch.cat(list(selected.values())).numpy(),
                        bins=np.concatenate([np.full(len(value), key) for key, value in selected.items()]),
                        centers=centers.numpy(), mass=mass.numpy(), range_values=range_values,
                        metadata=np.asarray(json.dumps(metadata)))
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
