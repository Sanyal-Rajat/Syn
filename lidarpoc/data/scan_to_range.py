"""Nearest-return spherical projection with explicit invalid masks and beam angles."""
import numpy as np
from scipy.cluster.vq import kmeans2


def infer_elevations(scans, height=128, tolerance_deg=.15):
    angles = []
    for xyz in scans:
        distance = np.linalg.norm(xyz, axis=1)
        keep = distance > 2
        angles.extend(np.arcsin(np.clip(xyz[keep, 2] / distance[keep], -1, 1)))
    values = np.asarray(angles)
    if len(values) < height * 30:
        raise ValueError('Not enough training returns to infer beam elevations; provide a measured beam table')
    if len(values) > 200_000:
        values = np.random.default_rng(0).choice(values, 200_000, replace=False)
    initial = np.quantile(values, np.linspace(.001, .999, height))
    centers, _ = kmeans2(values[:, None], initial[:, None], iter=100, minit='matrix', missing='raise')
    centers = np.sort(centers[:, 0])
    residual = np.min(np.abs(values[:, None] - centers), axis=1)
    if np.rad2deg(np.quantile(residual, .99)) > tolerance_deg or np.min(np.diff(centers)) < 1e-5:
        raise ValueError('Empirical beam fit failed its residual/spacing check; provide calibrated elevations')
    return centers


def project(xyz, intensity, elevations, width=512, max_range=120., tolerance_deg=.15, intensity_max=255.):
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
    return {'range': ranges, 'valid': valid, 'intensity': intensities,
            'has_intensity': np.bool_(intensity is not None),
            'rejected_fraction': 1 - np.count_nonzero(accepted) / max(1, len(xyz))}


def unproject(ranges, valid, elevations):
    ring, column = np.nonzero(valid)
    azimuth = (column + .5) / ranges.shape[-1] * (2 * np.pi) - np.pi
    distance, elevation = ranges[ring, column], np.asarray(elevations)[ring]
    return np.column_stack((distance * np.cos(elevation) * np.cos(azimuth),
                            distance * np.cos(elevation) * np.sin(azimuth), distance * np.sin(elevation)))
