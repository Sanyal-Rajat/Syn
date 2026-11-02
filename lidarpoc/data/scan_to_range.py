"""Nearest-return spherical projection with explicit invalid masks and beam angles."""
import numpy as np
from ..logs import get_logger


def infer_elevations(scans, height=128, tolerance_deg=.15, return_diagnostics=False):
    """Fit empirical angular bins by coverage, without assuming factory ring IDs."""
    if height < 2 or not np.isfinite(tolerance_deg) or tolerance_deg <= 0:
        raise ValueError('Beam count must be at least two and angular tolerance must be positive')
    angles = []
    for xyz in scans:
        xyz = np.asarray(xyz)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise ValueError('Beam estimation requires finite XYZ points in metres')
        distance = np.linalg.norm(xyz, axis=1)
        keep = distance > 2
        angles.append(np.arcsin(np.clip(xyz[keep, 2] / distance[keep], -1, 1)))
    values = np.concatenate(angles).astype('float64') if angles else np.empty(0)
    if len(values) < height * 30:
        raise ValueError('Not enough training returns to infer beam elevations; provide a measured beam table')
    if len(values) > 200_000:
        values = np.random.default_rng(0).choice(values, 200_000, replace=False)
    values.sort()
    if len(np.unique(values)) < height:
        raise ValueError('Not enough distinct training elevations for the requested beam count')

    def cover(radius):
        centers, index = [], 0
        while index < len(values) and len(centers) <= height:
            center = values[index] + radius
            centers.append(center)
            index = np.searchsorted(values, center + radius, side='right')
        return np.asarray(centers)

    # In one dimension greedy interval coverage is exact for a fixed radius.
    # Minimize the worst angular error instead of allocating bins by point density.
    lower, upper = 0., (values[-1] - values[0]) / 2
    for _ in range(50):
        radius = (lower + upper) / 2
        if len(cover(radius)) <= height:
            upper = radius
        else:
            lower = radius
    centers = cover(upper)

    def errors():
        right = np.searchsorted(centers, values).clip(0, len(centers) - 1)
        left = (right - 1).clip(0)
        return np.minimum(abs(values - centers[left]), abs(values - centers[right]))

    # Tied interval boundaries can leave fewer bins; add observed uncovered angles.
    while len(centers) < height:
        centers = np.sort(np.append(centers, values[errors().argmax()]))
    residual = errors()
    p99 = float(np.rad2deg(np.quantile(residual, .99)))
    spacing = float(np.rad2deg(np.diff(centers).min()))
    diagnostics = {'method': 'minimax_angular_coverage', 'sampled_training_returns': len(values),
                   'p99_residual_deg': p99, 'max_residual_deg': float(np.rad2deg(residual.max())),
                   'min_spacing_deg': spacing, 'tolerance_deg': tolerance_deg,
                   'accepted_fraction': float((residual <= np.deg2rad(tolerance_deg)).mean()),
                   'factory_ring_ids': False}
    get_logger().info('Empirical elevation fit: %s', diagnostics)
    if len(centers) != height or p99 > tolerance_deg or np.min(np.diff(centers)) < 1e-5:
        raise ValueError(f'Empirical beam fit failed: p99 angular error {p99:.4f} deg '
                         f'(limit {tolerance_deg:.4f}), minimum spacing {spacing:.6f} deg. '
                         'Check the input coordinate frame or provide measured beam elevations.')
    return (centers, diagnostics) if return_diagnostics else centers


def project(xyz, intensity, elevations, width=512, max_range=120., tolerance_deg=.15, intensity_max=255., return_indices=False):
    elevations = np.asarray(elevations)
    if elevations.ndim != 1 or not np.isfinite(elevations).all() or (np.diff(elevations) <= 0).any():
        raise ValueError('Beam elevations must be finite, strictly increasing radians')
    distance = np.linalg.norm(xyz, axis=1)
    elevation = np.arcsin(np.clip(xyz[:, 2] / np.maximum(distance, 1e-8), -1, 1))
    right = np.searchsorted(elevations, elevation).clip(0, len(elevations) - 1)
    left = (right - 1).clip(0)
    ring = np.where(abs(elevation - elevations[left]) < abs(elevation - elevations[right]), left, right)
    accepted = (distance >= .1) & (distance <= max_range)
    accepted &= np.abs(elevation - elevations[ring]) <= np.deg2rad(tolerance_deg)
    yaw = np.arctan2(xyz[:, 1], xyz[:, 0])
    column = (np.floor((yaw + np.pi) / (2 * np.pi) * width).astype(int) % width)
    index = np.flatnonzero(accepted)
    # Near points win when multiple returns project to the same ray.
    index = index[np.argsort(distance[index], kind='stable')]
    pixels = ring[index] * width + column[index]
    _, unique = np.unique(pixels, return_index=True)
    index = index[unique]
    ranges = np.zeros((len(elevations), width), dtype='float32')
    valid = np.zeros_like(ranges, dtype=bool)
    intensities = np.zeros_like(ranges)
    ranges[ring[index], column[index]] = distance[index]
    valid[ring[index], column[index]] = True
    if intensity is not None:
        if not np.isfinite(intensity).all():
            raise ValueError('Nonfinite intensity values')
        intensities[ring[index], column[index]] = np.clip(intensity[index] / intensity_max, 0, 1)
    result = {'range': ranges, 'valid': valid, 'intensity': intensities,
            'has_intensity': np.bool_(intensity is not None),
            'rejected_fraction': 1 - np.count_nonzero(accepted) / max(1, len(xyz))}
    if return_indices:
        point_index = np.full(ranges.shape, -1, dtype='int64')
        point_index[ring[index], column[index]] = index
        result['point_index'] = point_index
    return result


def unproject(ranges, valid, elevations):
    ring, column = np.nonzero(valid)
    azimuth = (column + .5) / ranges.shape[-1] * (2 * np.pi) - np.pi
    distance, elevation = ranges[ring, column], np.asarray(elevations)[ring]
    return np.column_stack((distance * np.cos(elevation) * np.cos(azimuth),
                            distance * np.cos(elevation) * np.sin(azimuth), distance * np.sin(elevation)))
