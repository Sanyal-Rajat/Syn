"""Export a real conditioning window with enough metadata for independent viewing."""
from pathlib import Path

import numpy as np

from .split_manifest import load_window, windows
from .synthetic_scene import load_scene
from ..logs import get_logger


def export_real_scene(manifest, window, output, source=''):
    sensor = manifest['sensor_meta']
    if source:
        path = Path(source).expanduser().resolve()
        scene = load_scene(path, sensor, window)
        if 'elevations' in scene and (scene['elevations'].shape != (len(sensor['elevations']),)
                or not np.allclose(scene['elevations'], sensor['elevations'], atol=1e-7, rtol=0)):
            raise ValueError('Conditioning beam elevations do not match the prepared manifest')
        origin = {'kind': 'file', 'path': str(path)}
    else:
        candidates = windows(manifest, 'train', window)
        if not candidates:
            raise ValueError(f'Automatic real-scene extraction needs a complete {window}-frame training window')
        identity, records = candidates[0]
        scene = load_window(records, sensor, identity)
        origin = {'kind': 'prepared', 'split': 'train', 'sequence_id': identity, 'frames': records}
    arrays = {key: value for key, value in scene.items() if key != 'sensor_meta'}
    arrays.update(elevations=np.asarray(sensor['elevations']), azimuths=np.asarray(sensor['azimuths']),
                  max_range_m=np.asarray(sensor['max_range_m']))
    target = Path(output).resolve() / 'real_scene.npz'
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays)
    get_logger().info('Exported real conditioning scene: %s; source=%s', target, origin)
    return str(target), origin
