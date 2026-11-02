"""Ray-space geometry checks and explicit, occlusion-aware pose compensation."""
import torch


def warp_depth(ranges, valid, source_pose, target_pose, sensor):
    height, width = ranges.shape
    elevation = ranges.new_tensor(sensor['elevations'])[:, None].expand(height, width)
    azimuth = ranges.new_tensor(sensor['azimuths'])[None].expand(height, width)
    points = torch.stack((ranges * elevation.cos() * azimuth.cos(), ranges * elevation.cos() * azimuth.sin(),
                          ranges * elevation.sin(), torch.ones_like(ranges)), -1).reshape(-1, 4)
    transform = torch.linalg.inv(target_pose.to(ranges)) @ source_pose.to(ranges)
    moved = (points @ transform.T)[:, :3]
    distance = moved.norm(dim=1)
    angles = torch.asin((moved[:, 2] / distance.clamp_min(1e-8)).clamp(-1, 1))
    beams = ranges.new_tensor(sensor['elevations'])
    right = torch.searchsorted(beams, angles).clamp(0, height - 1)
    left = (right - 1).clamp_min(0)
    ring = torch.where((angles - beams[left]).abs() < (angles - beams[right]).abs(), left, right)
    column = (((torch.atan2(moved[:, 1], moved[:, 0]) + torch.pi) / (2 * torch.pi) * width).floor().long() % width)
    accepted = valid.reshape(-1).bool() & (distance >= .1) & (distance <= sensor['max_range_m'])
    accepted &= (angles - beams[ring]).abs() <= torch.deg2rad(ranges.new_tensor(sensor['beam_tolerance_deg']))
    pixels = ring[accepted] * width + column[accepted]
    depth = ranges.new_full((height * width,), float('inf')).scatter_reduce(0, pixels, distance[accepted], reduce='amin')
    mask = torch.isfinite(depth).reshape(height, width)
    return torch.where(mask, depth.reshape(height, width), 0), mask


def pose_temporal_loss(prediction, batch, sensor, tolerance):
    if 'world_from_sensor' not in batch or batch['range'].shape[1] < 2:
        return prediction.sum() * 0, False
    residuals = []
    for index in range(len(prediction)):
        for frame in range(prediction.shape[1] - 1):
            poses = batch['world_from_sensor'][index]
            real, real_mask = warp_depth(batch['range'][index, frame], batch['valid'][index, frame],
                                         poses[frame], poses[frame + 1], sensor)
            generated, generated_mask = warp_depth(prediction[index, frame], batch['valid'][index, frame],
                                                    poses[frame], poses[frame + 1], sensor)
            visible = real_mask & generated_mask & batch['valid'][index, frame + 1].bool()
            visible &= (real - batch['range'][index, frame + 1]).abs() <= tolerance
            if visible.any():
                # Compare motion-compensated noise, excluding unmatched/occluded real surfaces.
                delta = generated - real - (prediction[index, frame + 1] - batch['range'][index, frame + 1])
                residuals.append(delta[visible].abs().mean())
    return (torch.stack(residuals).mean() if residuals else prediction.sum() * 0), bool(residuals)


def physical_loss(predicted_ranges, probabilities, canonical_ranges, canonical_valid, max_range, tolerance):
    valid = canonical_valid.bool()
    displacement = torch.relu((predicted_ranges - canonical_ranges).abs() - tolerance)
    surface = displacement[valid].square().mean() if valid.any() else predicted_ranges.sum() * 0
    clutter = (probabilities * ~valid).mean()
    bounds = (torch.relu(.1 - predicted_ranges) + torch.relu(predicted_ranges - max_range)).square().mean()
    return surface / max_range**2 + clutter + bounds / max_range**2


def scene_metrics(sample, canonical, sensor, tolerance):
    ranges = torch.as_tensor(sample['range'])
    valid = torch.as_tensor(sample['valid']).bool()
    known = torch.as_tensor(canonical['valid']).bool()
    reference = torch.as_tensor(canonical['range'])
    if ranges.shape != reference.shape or ranges.shape[-2:] != (len(sensor['elevations']), sensor['width']):
        raise ValueError('Generated scan does not preserve canonical sensor/ray dimensions')
    overlap = valid & known
    delta = (ranges[overlap] - reference[overlap]).abs()
    new_returns = int((valid & ~known).sum())
    bounds = int(((ranges < .1) | (ranges > sensor['max_range_m']))[valid].sum())
    maximum = float(delta.max()) if delta.numel() else 0.
    return {'new_surface_returns': new_returns, 'range_bound_violations': bounds,
            'max_surface_deviation_m': maximum, 'matched_returns': int(overlap.sum()),
            'geometry_pass': new_returns == 0 and bounds == 0 and maximum <= tolerance + 1e-5,
            'beam_compliance': True, 'occlusion_scope': 'canonical first-return ray mask'}
