"""Torch projection with stable nearest-return selection and original point indices."""
import math

import numpy as np
import torch


def project(xyz, intensity, elevations, width=512, max_range=120., tolerance_deg=.15,
            intensity_max=255., return_indices=False, device='cuda:0', transform=None):
    points = torch.as_tensor(xyz, dtype=torch.float64, device=device)
    beams = torch.as_tensor(elevations, dtype=torch.float64, device=device)
    if beams.ndim != 1 or not torch.isfinite(beams).all() or (beams.diff() <= 0).any():
        raise ValueError('Beam elevations must be finite, strictly increasing radians')
    if transform is not None:
        matrix = points.new_tensor(transform)
        points = points @ matrix[:3, :3].T + matrix[:3, 3]
    distance = torch.linalg.vector_norm(points, dim=1)
    angles = (points[:, 2] / distance.clamp_min(1e-8)).clamp(-1, 1).asin()
    right = torch.searchsorted(beams, angles).clamp(0, len(beams) - 1)
    left = (right - 1).clamp_min(0)
    ring = torch.where((angles - beams[left]).abs() < (angles - beams[right]).abs(), left, right)
    accepted = (distance >= .1) & (distance <= max_range) & ((angles - beams[ring]).abs() <= math.radians(tolerance_deg))
    column = (((torch.atan2(points[:, 1], points[:, 0]) + math.pi) / (2 * math.pi) * width).floor().long() % width)
    index = accepted.nonzero().flatten()
    index = index[torch.argsort(distance[index], stable=True)]
    pixels = ring[index] * width + column[index]
    winners = torch.full((len(beams) * width,), len(index), dtype=torch.long, device=device)
    winners.scatter_reduce_(0, pixels, torch.arange(len(index), device=device), reduce='amin', include_self=True)
    valid = winners < len(index)
    chosen = index[winners[valid]]
    ranges = torch.zeros(len(winners), dtype=torch.float32, device=device)
    intensities = torch.zeros_like(ranges)
    ranges[valid] = distance[chosen].float()
    if intensity is not None:
        values = torch.as_tensor(intensity, dtype=torch.float64, device=device)
        if not torch.isfinite(values).all():
            raise ValueError('Nonfinite intensity values')
        intensities[valid] = (values[chosen] / intensity_max).clamp(0, 1).float()
    shape = (len(beams), width)
    result = {key: value.reshape(shape).cpu().numpy() for key, value in
              (('range', ranges), ('valid', valid), ('intensity', intensities))}
    result.update(has_intensity=np.bool_(intensity is not None),
                  rejected_fraction=1 - int(accepted.sum()) / max(1, len(points)))
    if return_indices:
        indices = torch.full_like(winners, -1)
        indices[valid] = chosen
        result['point_index'] = indices.reshape(shape).cpu().numpy()
    return result
