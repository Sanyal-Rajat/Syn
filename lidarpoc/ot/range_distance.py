"""Frame-averaged range W1: differentiable training and exact hard-output scoring."""
import math

import torch

from ..logs import get_logger


def wasserstein_1d(source, target, weights=None):
    """Integrate |CDF_source-CDF_target| on every empirical support interval.

    On finite positive ranges, uniform weights reproduce SciPy's distance.
    Source weights differentiate return probabilities; no histogram or subsampling.
    """
    source, target = source.reshape(-1).double(), target.detach().reshape(-1).double()
    weights = torch.ones_like(source) if weights is None else weights.reshape(-1).double()
    keep = torch.isfinite(source) & (source > 0) & torch.isfinite(weights) & (weights > 0)
    source, weights = source[keep], weights[keep]
    target = target[torch.isfinite(target) & (target > 0)]
    if not len(source) or not len(target):
        return None
    values, order = torch.cat((source, target)).sort()
    mass = torch.cat((weights / weights.sum(), -torch.ones_like(target) / len(target)))
    return ((values[1:] - values[:-1]) * mass[order].cumsum(0)[:-1].abs()).sum()


def frame_distances(sample, target, soft=False, support=None):
    """One W1 per frame; None denotes missing support, never a fabricated zero.

    A one-dimensional target is a shared unconditional training reference. Scene
    targets retain all finite, valid ranges independently for each matching frame.
    """
    width = sample['range'].shape[-2] * sample['range'].shape[-1]
    ranges, valid = (sample[key].reshape(-1, width) for key in ('range', 'valid'))
    if support is not None:
        valid = valid * support.reshape_as(valid)
    if target['range'].ndim == 1:
        reference = [(target['range'], target['valid'])] * len(ranges)
    else:
        reference = list(zip(*(target[key].reshape(-1, width) for key in ('range', 'valid'))))
        if len(reference) != len(ranges):
            raise ValueError('Range-distance scoring requires matching temporal frames')
    distances = []
    for depth, returned, (actual, observed) in zip(ranges, valid, reference):
        actual = actual[observed.bool() & torch.isfinite(actual)]
        distances.append(wasserstein_1d(depth, actual, returned) if soft else
                         wasserstein_1d(depth[returned.bool()], actual))
    return distances


def training_loss(prediction, target, maximum):
    """Soft return masses train the same per-frame range-distribution objective.

    The complete-scene generation mask permits observed real rays only. Match
    that support during training while retaining gradients through return masses.
    """
    sample = {**prediction, 'range': prediction['range'].clamp(.1, maximum)}
    values = [value for value in frame_distances(sample, target, soft=True, support=target['valid'])
              if value is not None]
    return torch.stack(values).mean().float() if values else prediction['range'].sum() * 0


def validate_generated(model, data, indices, device, seed, autocast):
    """Score actual conditioned DDIM outputs; fixed validation indices/seeds only."""
    total, count, missing = 0., 0, 0
    for index in indices:
        real = data[index]
        target = {key: torch.as_tensor(real[key], device=device)[None] for key in ('range', 'valid')}
        with autocast():
            sample = model.sample(1, conditioning={**target, 'complete_scene': True}, seed=seed + index)
        distances = frame_distances(sample, target)
        observed = (target['valid'].bool() & torch.isfinite(target['range']) & (target['range'] > 0)).flatten(2).any(-1).reshape(-1)
        for value, supported in zip(distances, observed):
            if not bool(supported):
                continue
            if value is None or not bool(torch.isfinite(value)):
                missing += 1
            else:
                total += float(value)
                count += 1
        get_logger().info('Validation scene %d on %s: range gap=%s m', index, device,
                          [float(value) if value is not None and bool(torch.isfinite(value)) else None
                           for value in distances])
    return torch.tensor([total, count, missing], dtype=torch.float64, device=device)


def finite_number(value):
    """Translate missing NPZ diagnostics to JSON null, including legacy artifacts."""
    value = float(value)
    return value if math.isfinite(value) else None
