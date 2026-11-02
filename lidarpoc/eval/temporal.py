"""Pose-compensated real persistence and synthetic canonical-noise correlation."""
import numpy as np
import torch

from ..ot.constraints import warp_depth


def real_persistence(sample, sensor, tolerance):
    if 'world_from_sensor' not in sample or len(sample['range']) < 2:
        return None
    ranges = torch.as_tensor(sample['range']).float()
    valid = torch.as_tensor(sample['valid']).bool()
    poses = torch.as_tensor(sample['world_from_sensor']).float()
    rows = []
    for frame in range(len(ranges) - 1):
        warped, mask = warp_depth(ranges[frame], valid[frame], poses[frame], poses[frame + 1], sensor)
        visible = mask & valid[frame + 1]
        if visible.any():
            error = (warped - ranges[frame + 1]).abs()[visible]
            rows.append([float((error <= tolerance).float().mean()), float(error.clamp_max(tolerance).mean())])
    return {'matched_return_fraction': float(np.mean([row[0] for row in rows])),
            'clipped_alignment_residual_m': float(np.mean([row[1] for row in rows]))} if rows else None


def noise_correlation(sample, canonical, sensor, tolerance):
    if 'world_from_sensor' not in canonical or len(sample['range']) < 2:
        return None
    values = []
    ranges, truth = torch.as_tensor(sample['range']).float(), torch.as_tensor(canonical['range']).float()
    valid, known = torch.as_tensor(sample['valid']).bool(), torch.as_tensor(canonical['valid']).bool()
    poses = torch.as_tensor(canonical['world_from_sensor']).float()
    for frame in range(len(ranges) - 1):
        generated, gm = warp_depth(ranges[frame], valid[frame], poses[frame], poses[frame + 1], sensor)
        reference, rm = warp_depth(truth[frame], known[frame], poses[frame], poses[frame + 1], sensor)
        shared = gm & rm & valid[frame + 1] & known[frame + 1]
        shared &= (reference - truth[frame + 1]).abs() <= tolerance
        if int(shared.sum()) >= 2:
            a = (generated - reference)[shared].numpy()
            b = (ranges[frame + 1] - truth[frame + 1])[shared].numpy()
            if a.std() > 1e-8 and b.std() > 1e-8:
                values.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(values)) if values else None
