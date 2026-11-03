"""Prepare whole traces independently; share a loaded trace across its GPU frame workers."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
from queue import SimpleQueue

import numpy as np

from .annotations import load_poses, point_labels
from .pandar_reader import in_sensor_frame, load_arrays, sensor_transform, unpack_scan
from .scan_to_range import project
from .split_manifest import digest
from ..config import RunConfig
from ..logs import get_logger


def prepare_trace(task):
    config = RunConfig(**task['config'])
    path, output = Path(task['path']), Path(task['output'])
    split, transform, elevations = task['split'], np.asarray(task['transform']), np.asarray(task['elevations'])
    trace = path.name.removesuffix('_scan_list.npz')
    get_logger().info('Preparing %s trace %s; loading compressed arrays on CPU before projection', split, trace)
    frames = load_arrays(path)
    poses = load_poses(path, config.odometry_mode,
        np.linalg.inv(sensor_transform(config.calibration_file, 'vehicle'))) if config.odometry_mode != 'disabled' else None
    if poses is not None and len(poses) < min(config.max_frames, len(frames)):
        raise ValueError(f'{trace}: fewer odometry records than scan frames')
    entry = {'sequence_id': trace, 'source_file': str(path), 'source_sha256': digest(path),
             'source_frames': len(frames), 'schema': str(np.asarray(frames[0]).dtype), 'frames': [],
             'odometry_mode': config.odometry_mode if poses is not None else 'unavailable',
             'labels_source': 'native Classification field; semantic validity requires confirmation'}
    if poses is not None:
        entry['odometry_sha256'] = digest(Path(str(path).replace('_scan_list.npz', '_odom_list.npz')))
    folder = output / split / trace
    folder.mkdir(parents=True, exist_ok=True)
    devices = ['cpu']
    if config.gpus:
        try:
            from ..gpu import resolve_gpus
            count = resolve_gpus(-1) if config.gpus < 0 else min(config.gpus, resolve_gpus(-1))
            devices = [f'cuda:{index}' for index in range(count)] or ['cpu']
        except ImportError:
            if config.gpus > 0:
                raise
    available = SimpleQueue()
    for device in devices:
        available.put(device)

    def frame(item):
        frame_index, raw = item
        device = available.get()
        try:
            xyz, intensity = unpack_scan(raw)
            if device != 'cpu':
                from .gpu_projection import project as gpu_project
                projected = gpu_project(xyz, intensity, elevations, config.width, config.max_range_m,
                                        config.beam_tolerance_deg, config.intensity_max, return_indices=True,
                                        device=device, transform=transform)
            else:
                projected = project(in_sensor_frame(xyz, transform), intensity, elevations, config.width,
                                    config.max_range_m, config.beam_tolerance_deg, config.intensity_max, return_indices=True)
            labels = point_labels(raw)
            projected['labels'] = np.full(projected['valid'].shape, -1, dtype='int64')
            if labels is not None:
                projected['labels'][projected['valid']] = labels[projected['point_index'][projected['valid']]]
            if poses is not None:
                projected['world_from_sensor'] = poses[frame_index]
            if not projected['valid'].any():
                raise ValueError(f'{trace}:{frame_index}: zero valid rays; check coordinate frame, units and beams')
            quantized = np.round(projected['range'] / .25).astype('int16')
            fingerprint = hashlib.sha256(quantized.tobytes() + projected['valid'].tobytes()).hexdigest()
            cache = folder / f'{frame_index:06d}.npz'
            np.savez_compressed(cache, **projected, timestamp_ns=np.int64(frame_index * 50_000_000))
            if (frame_index + 1) % max(1, min(config.max_frames, len(frames)) // 20) == 0:
                get_logger().info('Projected %s frame %d/%d on %s', trace, frame_index + 1,
                                  min(config.max_frames, len(frames)), device)
            return {'path': str(cache), 'sha256': digest(cache), 'index': frame_index}, fingerprint

        finally:
            available.put(device)

    inputs = list(enumerate(frames[:config.max_frames]))
    if len(devices) > 1:
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            results = list(executor.map(frame, inputs))
    else:
        results = [frame(item) for item in inputs]
    entry['frames'] = [record for record, fingerprint in results]
    if len(entry['frames']) < config.window:
        raise ValueError(f'{trace}: fewer frames than the temporal window')
    get_logger().info('Prepared %s: %d/%d frames on %s', trace, len(entry['frames']), len(frames), devices)
    return {'split': split, 'entry': entry, 'fingerprints': [fingerprint for record, fingerprint in results]}
