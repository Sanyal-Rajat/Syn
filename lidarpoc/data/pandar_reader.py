"""Inspect and read published Pandar128 NPZ variants without hardcoded array keys."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def load_arrays(path):
    # Published variable-length scan lists are object arrays and require pickle.
    with np.load(path, allow_pickle=True) as archive:
        candidates = [key for key in ('data', 'scan_list') if key in archive.files]
        if len(candidates) != 1:
            raise ValueError(f'{path}: select an unambiguous data/scan_list array; found {archive.files}')
        return archive[candidates[0]]


def unpack_scan(frame):
    frame = np.asarray(frame)
    fields = frame.dtype.names or ()
    if {'x', 'y', 'z'} <= set(fields) and frame['x'].ndim == 1:
        xyz = np.column_stack([frame[key] for key in ('x', 'y', 'z')])
    elif 'x' in fields and frame['x'].ndim == 2 and frame['x'].shape[1] == 3:
        xyz = np.asarray(frame['x'])
    elif frame.dtype == object:
        # np.array(list_of_structured_frames, dtype=object) loses field names when
        # frame lengths coincide. The published writer's tuple order is fixed.
        values = np.asarray(frame.tolist(), dtype=float)
        if (values.ndim != 2 or values.shape[1] != 5 or not np.isfinite(values).all()
                or not np.all(values[:, 0] == values[:, 0].astype(int))):
            raise ValueError('Unrecognized object frame; expected published Classification,x,y,z,intensity tuples')
        return values[:, 1:4].astype('float32'), values[:, 4].astype('float32')
    else:
        raise ValueError(f'Unrecognized point schema: dtype={frame.dtype}, shape={frame.shape}. Inspect the NPZ first.')
    intensity_key = next((key for key in ('intensity', 'i') if key in fields), None)
    intensity = np.asarray(frame[intensity_key]).reshape(-1) if intensity_key else None
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('Expected finite XYZ coordinates in metres')
    return xyz.astype('float32'), intensity.astype('float32') if intensity is not None else None


def inspect_dataset(root):
    output = []
    for path in sorted(Path(root).expanduser().glob('**/*_scan_list.npz')):
        frames = load_arrays(path)
        if not len(frames):
            raise ValueError(f'{path}: empty scan list')
        xyz, intensity = unpack_scan(frames[0])
        output.append({'path': str(path), 'frames': len(frames), 'first_frame_points': len(xyz),
                       'dtype': str(frames[0].dtype), 'intensity': intensity is not None,
                       'intensity_percentiles': np.quantile(intensity, [.01, .5, .99]).tolist() if intensity is not None else None,
                       'range_percentiles_m': np.quantile(np.linalg.norm(xyz, axis=1), [.01, .5, .99]).tolist()})
        del frames
    if not output:
        raise ValueError('No *_scan_list.npz files found under the remote dataset directory')
    return output


def sensor_transform(calibration_path, input_frame):
    if input_frame == 'sensor':
        return np.eye(4)
    calibration = json.loads(Path(calibration_path).read_text())['Pandar 128']['Mounting position']
    rotation = Rotation.from_euler('ZYX', [float(calibration[f'{axis}_deg']) for axis in ('yaw', 'pitch', 'roll')],
                                   degrees=True).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = [float(calibration[f'pos_{axis}_m']) for axis in ('x', 'y', 'z')]
    return np.linalg.inv(transform)


def in_sensor_frame(xyz, transform):
    return xyz @ transform[:3, :3].T + transform[:3, 3]
