"""Fixed physical features, geometry-defined strata and non-normalized return mass."""
import torch
from torch.nn import functional as F

from .sinkhorn import physical_cost, sinkhorn_divergence, transport


def features(sample, sensor, config, bin_ranges=None):
    ranges, valid, intensity = (sample[key] for key in ('range', 'valid', 'intensity'))
    ranges = ranges.reshape(-1, *ranges.shape[-2:])
    valid, intensity = valid.reshape_as(ranges).to(ranges.dtype), intensity.reshape_as(ranges)
    elevation = ranges.new_tensor(sensor['elevations'])[None, :, None].expand_as(ranges)
    density = F.avg_pool2d(valid[:, None], 3, stride=1, padding=1)[:, 0]
    sequences = sample['valid'].to(ranges.dtype)
    if sequences.ndim == 3:
        sequences = sequences[None]
    persistence = sequences * sequences.roll(1, dims=1)
    persistence[:, 0] = sequences[:, 0]
    persistence = persistence.reshape_as(ranges)
    vectors = torch.stack((ranges, intensity, elevation, density, persistence), -1).reshape(-1, 5)
    geometry = ranges.detach() if bin_ranges is None else bin_ranges.reshape_as(ranges).detach()
    edges = ranges.new_tensor(config.range_edges_m)
    range_bin = torch.bucketize(geometry, edges[1:-1])
    ring = torch.arange(ranges.shape[1], device=ranges.device) * config.ring_groups // ranges.shape[1]
    bins = (ring[None, :, None] * (len(edges) - 1) + range_bin).expand_as(ranges).reshape(-1)
    accepted = (geometry >= edges[0]) & (geometry <= edges[-1])
    return vectors, bins, (valid * accepted).reshape(-1)


def select(features, bins, valid, max_points):
    result = {}
    for key in bins.unique().tolist():
        indices = torch.where((bins == key) & (valid > .5))[0]
        if len(indices):
            index = torch.linspace(0, len(indices) - 1, min(len(indices), max_points), device=indices.device).long()
            result[int(key)] = features[indices[index]]
    return result


def conditional_ot(source, target, config, return_bins=False):
    losses, weights, unsupported, per_bin = [], [], [], {}
    for key in sorted(set(source) | set(target)):
        a, b = source.get(key), target.get(key)
        if a is None or b is None or min(len(a), len(b)) < config.min_samples:
            unsupported.append(key)
            continue
        losses.append(sinkhorn_divergence(a, b.detach().to(a), config.feature_scales, config.feature_weights,
                                          config.epsilon, config.iterations))
        if return_bins:
            per_bin[str(key)] = float(losses[-1].detach())
        weights.append(min(len(a), len(b)))
    if not losses:
        raise ValueError('No supported conditional OT bins; check the fixed strata and minimum support')
    weight = losses[0].new_tensor(weights)
    result = ((torch.stack(losses) * weight).sum() / weight.sum(), unsupported)
    return (*result, per_bin) if return_bins else result


def return_mass(sample, sensor, config):
    ranges = sample['range'].reshape(-1, *sample['range'].shape[-2:])
    valid = sample['valid'].reshape_as(ranges).to(ranges.dtype)
    edges = ranges.new_tensor(config.range_edges_m)
    range_bin = torch.bucketize(ranges.detach(), edges[1:-1])
    ring = torch.arange(ranges.shape[1], device=ranges.device) * config.ring_groups // ranges.shape[1]
    sector = torch.arange(ranges.shape[2], device=ranges.device) * config.azimuth_sectors // ranges.shape[2]
    indices = ((ring[None, :, None] * config.azimuth_sectors + sector[None, None]) * (len(edges) - 1) + range_bin)
    count = config.ring_groups * config.azimuth_sectors * (len(edges) - 1)
    mass = ranges.new_zeros(count).scatter_add(0, indices.reshape(-1), valid.reshape(-1)) / valid.numel()
    # Divide by fixed ray exposure, never by total observed returns: sums retain return probability.
    centers = []
    for group in range(config.ring_groups):
        for azimuth in range(config.azimuth_sectors):
            for lower, upper in zip(config.range_edges_m, config.range_edges_m[1:]):
                centers.append([(lower + upper) / 2, group / config.ring_groups, azimuth / config.azimuth_sectors])
    return ranges.new_tensor(centers), mass


def mass_ot(source_mass, target_mass, centers, config):
    cost = physical_cost(centers, centers, [config.feature_scales[0], 1., 1.], [1., 1., 1.])
    return transport(cost, source_mass, target_mass.detach().to(source_mass), config.epsilon, config.iterations,
                     config.tau_source, config.tau_target)
