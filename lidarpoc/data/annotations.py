"""Optional publisher labels/odometry; never infer an odometry convention."""
from pathlib import Path

import numpy as np


def point_labels(frame):
    frame = np.asarray(frame)
    if 'Classification' in (frame.dtype.names or ()):
        return frame['Classification'].reshape(-1).astype('int64')
    if frame.dtype == object:
        values = np.asarray(frame.tolist())
        if values.ndim == 2 and values.shape[1] == 5:
            return values[:, 0].astype('int64')
    return None


def load_poses(scan_path, mode, vehicle_from_sensor):
    path = Path(str(scan_path).replace('_scan_list.npz', '_odom_list.npz'))
    if mode == 'disabled' or not path.is_file():
        return None
    with np.load(path, allow_pickle=True) as archive:
        key = next((key for key in ('data', 'odom_list') if key in archive.files), None)
        if key is None:
            raise ValueError(f'{path}: unrecognized odometry array')
        records = archive[key]
    poses = []
    previous = np.eye(4)
    for record in records:
        if isinstance(record, dict):
            matrix = record['transformation']
        elif 'transformation' in (getattr(getattr(record, 'dtype', None), 'names', None) or ()):
            matrix = record['transformation']
        else:
            matrix = record
        matrix = np.asarray(matrix, dtype='float64').reshape(4, 4)
        if (not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5)
                or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-3)
                or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-3)):
            raise ValueError(f'{path}: odometry needs finite, rigid homogeneous transformations')
        if mode == 'previous_from_current':
            previous = previous @ matrix
            matrix = previous.copy()
        poses.append(matrix @ vehicle_from_sensor)
    return np.asarray(poses, dtype='float32')
