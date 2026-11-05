"""Export a real conditioning window with enough metadata for independent viewing."""
from pathlib import Path
import os
import tempfile

import numpy as np

from .split_manifest import load_window, windows
from .synthetic_scene import load_scene
from ..logs import get_logger


def source_metadata(manifest, scene):
    """Keep recorded frame provenance; recover old first-window caches by exact equality."""
    keys = ('source_scan_file', 'source_frame_indices')
    if all(key in scene for key in keys):
        indices = np.asarray(scene['source_frame_indices'])
        if indices.shape != (scene['range'].shape[0],) or not np.issubdtype(indices.dtype, np.integer) or (indices < 0).any():
            raise ValueError('Source frame indices must identify every conditioning frame')
        return {key: scene[key] for key in keys}
    window = scene['range'].shape[0]
    candidates = windows(manifest, 'train', window)
    if candidates:
        identity, records = candidates[0]
        expected = load_window(records, manifest['sensor_meta'], identity)
        if all(np.array_equal(scene[key], expected[key]) for key in ('range', 'valid')):
            trace = next(trace for trace in manifest['splits']['train'] if trace['sequence_id'] == identity)
            return {'source_scan_file': np.asarray(trace['source_file']),
                    'source_frame_indices': np.asarray([record['index'] for record in records], dtype='int64')}
    raise ValueError('Conditioning scene lacks source scan/frame metadata and does not exactly match the automatic training window')


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
    try:
        arrays.update(source_metadata(manifest, scene))
    except ValueError:
        pass  # Independent synthetic scenes remain valid inputs, without camera pairing.
    arrays.update(elevations=np.asarray(sensor['elevations']), azimuths=np.asarray(sensor['azimuths']),
                  max_range_m=np.asarray(sensor['max_range_m']))
    target = Path(output).resolve() / 'real_scene.npz'
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.scene-', suffix='.npz', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    get_logger().info('Exported real conditioning scene: %s; source=%s', target, origin)
    return str(target), origin
