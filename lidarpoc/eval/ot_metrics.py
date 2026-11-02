"""Held-out transport, physical and pose-aware statistics with frozen feature units."""
import numpy as np
import torch

from ..ot.constraints import pose_temporal_loss, scene_metrics
from ..ot.environment import score
from ..ot.features import features, return_mass, select
from ..ot.reference import stack_samples
from .domain_metrics import bootstrap, compare, describe
from .temporal import noise_correlation, real_persistence


def empirical_reference(samples, sensor, config):
    sample = stack_samples(samples)
    vector, bins, valid = features(sample, sensor, config)
    selected = select(vector, bins, valid, config.max_points)
    centers, mass = return_mass(sample, sensor, config)
    return {'features': torch.cat(list(selected.values())).numpy() if selected else np.empty((0, 5), dtype='float32'),
            'bins': np.concatenate([np.full(len(value), key) for key, value in selected.items()]) if selected else np.empty(0, dtype='int64'),
            'centers': centers.numpy(), 'mass': mass.numpy()}


def evaluate(real, generated, canonical, sensor, config, tolerance):
    grouped = {}
    for sample in real:
        grouped.setdefault(sample['sequence_id'], []).append(sample)
    rows = []
    for identity, samples in grouped.items():
        metrics = score(generated, empirical_reference(samples, sensor, config), sensor, config)
        metrics.update(compare(describe(samples), describe(generated), sensor['max_range_m']))
        rows.append({'sequence_id': identity, **metrics})
    geometry = [scene_metrics(sample, original, sensor, tolerance) for sample, original in zip(generated, canonical)]
    pose_values = []
    real_pose = [value for sample in real if (value := real_persistence(sample, sensor, tolerance)) is not None]
    correlations = []
    for sample, original in zip(generated, canonical):
        if 'world_from_sensor' in original:
            batch = stack_samples([original])
            batch['world_from_sensor'] = torch.as_tensor(original['world_from_sensor'])[None]
            value, supported = pose_temporal_loss(torch.as_tensor(sample['range'])[None], batch, sensor, tolerance)
            if supported:
                pose_values.append(float(value))
        correlation = noise_correlation(sample, original, sensor, tolerance)
        if correlation is not None:
            correlations.append(correlation)
    result = {'traces': rows, 'gap': bootstrap([row['gap'] for row in rows]),
              'legacy_loss': bootstrap([row['loss'] for row in rows]), 'geometry': geometry,
              'geometry_pass': all(row['geometry_pass'] for row in geometry),
              'pose_compensated_noise_discrepancy_m': float(np.mean(pose_values)) if pose_values else None,
              'pose_supported_windows': len(pose_values), 'generated_windows': len(generated),
              'real_pose_supported_windows': len(real_pose), 'real_ego_motion_persistence': real_pose or None,
              'synthetic_noise_autocorrelation': float(np.mean(correlations)) if correlations else None}
    return result


def session_intervals(results, metric='gap'):
    grouped = {}
    for result in results:
        for row in result['traces']:
            grouped.setdefault(row['sequence_id'], []).append(row[metric])
    return bootstrap([float(np.mean(values)) for values in grouped.values()])
