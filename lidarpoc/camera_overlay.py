"""Export camera comparisons for generated windows; usable without loading a model."""
import argparse
import json
from pathlib import Path
import zipfile

import numpy as np

from .camera import draw_overlay, load_camera
from .data.camera import matching_camera_frames
from .data.conditioning import source_metadata
from .data.scan_to_range import unproject
from .io import write_json
from .logs import get_logger


def export_overlays(config, manifest, directory, real_scene, progress=lambda *args: None):
    """Return a visible diagnostic if inputs cannot support a corresponding camera view."""
    if not config.camera_overlay:
        return {'status': 'disabled'}
    directory = Path(directory)
    try:
        if not real_scene:
            raise ValueError('Camera overlays require real-scene conditioning; unconditional samples have no corresponding recorded camera')
        samples = sorted(directory.glob('sample_*.npz'))
        if not samples:
            raise ValueError('No sample_*.npz files found in the Generate job directory')
        with np.load(real_scene, allow_pickle=False) as source:
            scene = {key: source[key].copy() for key in source.files}
        metadata = source_metadata(manifest, scene)
        from .provenance import license_gate
        dataset = license_gate()['dataset']
        camera_path = Path(config.camera_calibration or config.calibration_file).expanduser().resolve()
        camera = load_camera(camera_path, config.camera_model)
        progress('matching recorded camera frames to conditioning LiDAR')
        frames = matching_camera_frames(str(metadata['source_scan_file']), metadata['source_frame_indices'],
            dataset, progress, config.camera_timestamp_unit, config.camera_max_skew_ms)
        folder = directory / 'overlays'
        folder.mkdir(exist_ok=True)
        calibration = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in camera.items()}
        write_json(folder / 'calibration.json', calibration)
        rows = []
        for frame, (image, timing) in enumerate(frames):
            camera_file = f'camera_{frame:03d}.jpg'
            image.save(folder / camera_file, quality=95)
            row = {**timing, 'frame': frame, 'camera': camera_file, 'synthetic': []}
            for path in [Path(real_scene), *samples]:
                with np.load(path, allow_pickle=False) as data:
                    if data['range'].shape != scene['range'].shape or data['valid'].shape != scene['valid'].shape:
                        raise ValueError(f'{path.name}: sample and conditioning window shapes differ')
                    ranges, valid = data['range'][frame], data['valid'][frame].astype(bool)
                    elevations = data['elevations'] if 'elevations' in data else manifest['sensor_meta']['elevations']
                    points = unproject(ranges, valid & np.isfinite(ranges) & (ranges > 0), elevations)
                    status = str(data['training_status'].item()) if 'training_status' in data else ''
                real = path.resolve() == Path(real_scene).resolve()
                label = 'Real LiDAR' if real else f'Synthetic LiDAR {path.stem}'
                if not real and status in ('untrained', 'trained'):
                    label = f'{status.capitalize()} diffusion LiDAR'
                overlay, count = draw_overlay(image, points, camera, label)
                name = f'{"real" if real else path.stem}_frame_{frame:03d}.png'
                overlay.save(folder / name)
                record = {'file': name, 'sample': path.name, 'projected_pixels': count}
                if status and not real:
                    record['training_status'] = status
                if real:
                    row['real'] = record
                else:
                    row['synthetic'].append(record)
                progress(f'camera overlay {name}: {count:,} visible pixels')
            rows.append(row)
        report = {'status': 'completed', 'background': 'recorded RGB; only LiDAR is synthetic',
                  'calibration_source': str(camera_path), 'calibration_convention': camera['convention'],
                  'distortion_model': camera['model'], 'timestamp_unit': config.camera_timestamp_unit,
                  'timestamp_unit_policy': 'Explicit timestamp_s/ms/us/ns fields use their named units; the configured unit applies to t.',
                  'max_skew_ms': config.camera_max_skew_ms, 'frames': rows,
                  'limitations': 'Nearest timestamp, no motion/rolling-shutter compensation; nearest return per pixel is not full RGB occlusion handling.'}
        write_json(folder / 'manifest.json', report)
        get_logger().info('Exported camera overlays: %s', folder)
        return report
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        report = {'status': 'unavailable', 'reason': str(error)}
        get_logger().warning('Camera overlays unavailable: %s', error)
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True, help='Existing completed Generate job directory')
    parser.add_argument('--calibration', help='Optional explicit optical-frame camera JSON')
    parser.add_argument('--model', choices=('rational', 'pinhole'))
    parser.add_argument('--timestamp-unit', choices=('s', 'ms', 'us', 'ns'))
    parser.add_argument('--max-skew-ms', type=float)
    args = parser.parse_args()
    from .config import RunConfig
    from .data.split_manifest import load_manifest
    from .logs import progress
    from .report import write_pdf

    request = json.loads((args.job / 'request.json').read_text())
    overrides = {name: value for name, value in [('camera_calibration', args.calibration), ('camera_model', args.model),
        ('camera_timestamp_unit', args.timestamp_unit), ('camera_max_skew_ms', args.max_skew_ms)] if value is not None}
    config = RunConfig(**{**request, 'camera_overlay': True, **overrides})
    if config.action != 'generate':
        parser.error('--job must point to a Generate job')
    path = args.job / 'result.json'
    report = json.loads(path.read_text())
    report['camera_overlays'] = export_overlays(config, load_manifest(config.manifest), args.job,
        report.get('real_scene'), lambda *values: progress(args.job, *values))
    if report.get('comparison'):
        from .comparison import analyze
        report['comparison'] = analyze(args.job, report, config.comparison_voxel_m)
    write_json(path, report)
    write_pdf(path)
    print(json.dumps(report['camera_overlays'], indent=2))
    if report['camera_overlays']['status'] != 'completed':
        parser.exit(1, 'Camera overlay inputs are unavailable; see the diagnostic above.\n')
