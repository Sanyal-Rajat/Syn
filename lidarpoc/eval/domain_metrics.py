"""Range/elevation Wasserstein, per-ring returns/intensity and temporal persistence."""
import numpy as np
from scipy.stats import wasserstein_distance


def describe(samples):
    ranges = np.concatenate([sample['range'].reshape(-1, *sample['range'].shape[-2:]) for sample in samples])
    valid = np.concatenate([sample['valid'].reshape(-1, *sample['valid'].shape[-2:]) for sample in samples])
    intensity = np.concatenate([sample['intensity'].reshape(-1, *sample['intensity'].shape[-2:]) for sample in samples])
    has_intensity = all(bool(np.asarray(sample.get('has_intensity', True)).all()) for sample in samples)
    distributions, intensities = [], []
    for ring in range(ranges.shape[1]):
        intensities.append(intensity[:, ring][valid[:, ring]])
        for lower, upper in [(0., 30.), (30., 60.), (60., np.inf)]:
            selected = valid[:, ring] & (ranges[:, ring] >= lower) & (ranges[:, ring] < upper)
            distributions.append(ranges[:, ring][selected])
    temporal = []
    for sample in samples:
        if sample['range'].shape[-3] < 2:
            continue
        shared = sample['valid'][..., 1:, :, :] & sample['valid'][..., :-1, :, :]
        delta = np.abs(np.diff(sample['range'], axis=-3))
        temporal.append([float(shared.mean()), float(delta[shared].mean()) if shared.any() else 0.])
    return {'range_distributions': distributions, 'intensity_distributions': intensities,
            'has_intensity': has_intensity, 'return_rates': valid.mean(axis=(0, 2)),
            'temporal': np.mean(temporal, axis=0) if temporal else None}


def compare(real, generated, max_range):
    distances, unsupported = [], 0
    for a, b in zip(real['range_distributions'], generated['range_distributions']):
        if len(a) >= 4 and len(b) >= 4:
            distances.append(wasserstein_distance(a, b) / max_range)
        elif len(a) or len(b):
            unsupported += 1
    intensity = None
    if real['has_intensity'] and generated['has_intensity']:
        values = [wasserstein_distance(a, b) for a, b in zip(real['intensity_distributions'], generated['intensity_distributions'])
                  if len(a) >= 4 and len(b) >= 4]
        intensity = float(np.mean(values)) if values else None
    temporal = None
    if real['temporal'] is not None and generated['temporal'] is not None:
        temporal = float(np.mean(np.abs((real['temporal'] - generated['temporal']) / [1., max_range])))
    rate = float(np.mean(np.abs(real['return_rates'] - generated['return_rates'])))
    range_distance = float(np.mean(distances)) if distances else 1.
    support_penalty = unsupported / max(1, len(real['range_distributions']))
    terms = [range_distance, rate, support_penalty]
    terms += [value for value in (intensity, temporal) if value is not None]
    return {'loss': float(np.mean(terms)), 'range_wasserstein_normalized': range_distance,
            'return_rate_mae': rate, 'intensity_wasserstein': intensity, 'temporal_discrepancy': temporal,
            'unsupported_contexts': unsupported}


def bootstrap(values, samples=1000):
    if not values:
        return None
    values = np.asarray(values)
    rng = np.random.default_rng(0)
    means = values[rng.integers(0, len(values), (samples, len(values)))].mean(axis=1)
    return {'mean': float(values.mean()), 'ci95': np.quantile(means, [.025, .975]).tolist(), 'traces': len(values)}
