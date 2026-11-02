"""Asset-free first-return ray casting of a road plane, lane markings and boxes."""
import hashlib
from pathlib import Path

import numpy as np


def road_scene(sensor, window=2, seed=0):
    rng = np.random.default_rng(seed)
    elevation = np.asarray(sensor['elevations'])[:, None]
    azimuth = np.asarray(sensor['azimuths'])[None]
    directions = np.stack(np.broadcast_arrays(np.cos(elevation) * np.cos(azimuth),
                          np.cos(elevation) * np.sin(azimuth), np.sin(elevation)), -1)
    height, lane_width, speed = rng.uniform(1.5, 2.), rng.uniform(3., 4.), rng.uniform(0, 10)
    box_center = np.array([rng.uniform(12, 30), rng.uniform(-2, 2), -height + .8])
    lower, upper = box_center - [.9, .8, .8], box_center + [.9, .8, .8]
    frames = []
    for frame in range(window):
        origin = np.array([speed * frame * .05, 0., 0.])
        direction_z = directions[..., 2]
        plane = np.divide(-height, direction_z, out=np.full(direction_z.shape, np.inf), where=direction_z < -1e-8)
        parallel = np.abs(directions) < 1e-8
        safe = np.where(parallel, 1., directions)
        first, last = (lower - origin) / safe, (upper - origin) / safe
        near, far = np.minimum(first, last), np.maximum(first, last)
        inside = (origin >= lower) & (origin <= upper)
        near = np.where(parallel, np.where(inside, -np.inf, np.inf), near)
        far = np.where(parallel, np.where(inside, np.inf, -np.inf), far)
        entry, exit = near.max(-1), far.min(-1)
        box = np.where((exit >= np.maximum(entry, 0)) & (entry > .1), entry, np.inf)
        ranges = np.minimum(plane, box)
        valid = np.isfinite(ranges) & (ranges >= .1) & (ranges <= sensor['max_range_m'])
        ranges = np.where(valid, ranges, 0.)
        points = origin + ranges[..., None] * directions
        on_road = valid & (plane <= box)
        labels = np.zeros(valid.shape, dtype='int64')
        labels[on_road & (abs(points[..., 1] - lane_width / 2) < .15)] = 1
        labels[on_road & (abs(points[..., 1] + lane_width / 2) < .15)] = 2
        labels[~valid] = -1
        intensity = np.where(labels == 1, .9, np.where(labels == 2, .65, .2))
        pose = np.eye(4, dtype='float32')
        pose[:3, 3] = origin
        frames.append({'range': ranges.astype('float32'), 'valid': valid,
                       'intensity': np.where(valid, intensity, 0).astype('float32'), 'labels': labels,
                       'world_from_sensor': pose, 'timestamp_ns': np.int64(frame * 50_000_000), 'has_intensity': True})
    scene = {key: np.stack([value[key] for value in frames]) for key in frames[0]}
    scene['sequence_id'] = f'procedural_{seed}'
    return scene


def load_scene(path, sensor, window):
    with np.load(Path(path).expanduser(), allow_pickle=False) as archive:
        scene = {key: archive[key].copy() for key in archive.files}
    shape = (window, len(sensor['elevations']), sensor['width'])
    if any(key not in scene or scene[key].shape != shape for key in ('range', 'valid')):
        raise ValueError(f'{path}: canonical scene range/valid must match {shape}')
    scene['valid'] = scene['valid'].astype(bool)
    if (not np.isfinite(scene['range']).all() or not scene['valid'].any()
            or (scene['range'][scene['valid']] < .1).any()
            or (scene['range'][scene['valid']] > sensor['max_range_m']).any()):
        raise ValueError(f'{path}: canonical ranges must be finite, nonempty and within sensor bounds')
    scene.setdefault('intensity', np.zeros(shape, dtype='float32'))
    scene.setdefault('has_intensity', np.zeros(window, dtype=bool))
    if scene['intensity'].shape != shape or not np.isfinite(scene['intensity']).all():
        raise ValueError(f'{path}: canonical intensity shape/values are invalid')
    if 'labels' in scene and (scene['labels'].shape != shape or not np.issubdtype(scene['labels'].dtype, np.integer)):
        raise ValueError(f'{path}: semantic labels must be integer arrays matching range/valid')
    if 'world_from_sensor' in scene:
        poses = scene['world_from_sensor']
        if (poses.shape != (window, 4, 4) or not np.isfinite(poses).all()
                or not np.allclose(poses[:, 3], [0, 0, 0, 1])
                or not np.allclose(poses[:, :3, :3].transpose(0, 2, 1) @ poses[:, :3, :3], np.eye(3), atol=1e-3)
                or not np.allclose(np.linalg.det(poses[:, :3, :3]), 1, atol=1e-3)):
            raise ValueError(f'{path}: world_from_sensor needs rigid homogeneous poses')
    scene['sequence_id'] = Path(path).stem
    return scene


def geometry_digest(scene):
    return hashlib.sha256(np.asarray(scene['range'], dtype='float32').tobytes()
                          + np.asarray(scene['valid'], dtype=bool).tobytes()).hexdigest()


def scene_splits(sensor, window, settings, directory):
    directory = Path(directory)
    result, seen = {}, {}
    for split, offset, filenames in [('train', 0, settings.scene_files), ('validation', 10000, settings.development_scene_files),
                                     ('test', 20000, settings.test_scene_files)]:
        scenes = [load_scene(path, sensor, window) for path in filenames] if filenames else [
            road_scene(sensor, window, offset + index) for index in range(settings.synthetic_scenes)]
        for index, scene in enumerate(scenes):
            fingerprint = geometry_digest(scene)
            if fingerprint in seen and seen[fingerprint] != split:
                raise ValueError('Canonical scene geometry overlaps across experiment splits')
            seen[fingerprint] = split
            folder = directory / split
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / f'{index:03d}.npz', **scene)
        result[split] = scenes
    return result
