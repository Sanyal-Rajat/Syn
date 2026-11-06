"""Paired untrained/trained generation and spatial analysis, separate from training/RL."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np

from .data.scan_to_range import unproject
from .eval.spatial_gap import covariates, distribution_gap, voxel_gap
from .io import write_json
from .logs import get_logger

ARMS = {'untrained': 'sample_000.npz', 'trained': 'sample_001.npz'}
PAIRS = {'untrained_vs_real': ('real', 'untrained'), 'trained_vs_real': ('real', 'trained'),
         'trained_vs_untrained': ('untrained', 'trained')}


def paired_samples(model, sampler, config, conditioning):
    import torch
    from .models.lidardm_adapter import LidarDMAdapter
    from .runner import generate
    from .sampling import SamplingPool

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config.comparison_init_seed)
        baseline = LidarDMAdapter(dict(model.config), model.sensor_meta)
    second = 'cuda:1' if config.gpus > 1 else str(next(model.parameters()).device)
    baseline.to(second)
    pool = SamplingPool(baseline, [second])
    jobs = [(baseline, pool), (model, sampler)]
    with np.load(config.conditioning, allow_pickle=False) as reference:
        has_intensity = bool(np.asarray(reference['has_intensity']).all()) if 'has_intensity' in reference else False

    def sample(item):
        arm, (current, current_pool) = item
        get_logger().info('Generating %s arm with shared sampling seed %d', arm, config.seed)
        return generate(current, 1, config.seed, conditioning, has_intensity=has_intensity, sampler=current_pool)[0]

    try:
        items = list(zip(ARMS, jobs))
        if config.gpus > 1:
            with ThreadPoolExecutor(max_workers=2) as executor:
                return list(executor.map(sample, items))
        return [sample(item) for item in items]
    finally:
        pool.close()
        baseline.cpu()


def load_arrays(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def frame_cloud(scene, frame):
    ranges = scene['range'][frame]
    valid = scene['valid'][frame].astype(bool) & np.isfinite(ranges) & (ranges > 0)
    points = unproject(ranges, valid, scene['elevations'])
    has_intensity = np.asarray(scene.get('has_intensity', False))
    present = bool(has_intensity) if has_intensity.ndim == 0 else bool(has_intensity[frame])
    intensity = scene['intensity'][frame][valid] if present and 'intensity' in scene else None
    return points, intensity


def analyze(directory, report, voxel_size):
    """Compute all metrics on the compute machine and persist full-resolution maps."""
    from .pixel_graphs import VERSION, analyze_pixels

    directory = Path(directory).resolve()
    scenes = {name: load_arrays(directory / filename) for name, filename in {'real': 'real_scene.npz', **ARMS}.items()}
    reference = scenes['real']
    for arm in ARMS:
        if str(scenes[arm].get('training_status', '')) != arm:
            raise ValueError(f'{ARMS[arm]} is missing its paired-generation training-status metadata')
    for scene in scenes.values():
        if (scene['range'].shape != reference['range'].shape or scene['valid'].shape != reference['valid'].shape
                or not np.allclose(scene['elevations'], reference['elevations'], atol=1e-7, rtol=0)):
            raise ValueError('Comparison scenes must share temporal, range-image and calibrated beam grids')
    overlays = report.get('camera_overlays', {})
    if report.get('pixel_analysis', {}).get('version') != VERSION:
        report['pixel_analysis'] = analyze_pixels(directory, report)
    pixel_frames = report['pixel_analysis']['frames']
    folder = directory / 'comparison'
    folder.mkdir(exist_ok=True)
    frames = []
    for frame in range(reference['range'].shape[0]):
        clouds, features = {}, {}
        for name, scene in scenes.items():
            clouds[name], intensity = frame_cloud(scene, frame)
            features[name] = covariates(clouds[name], intensity, float(reference['max_range_m']))
        pixel_row = next(row for row in pixel_frames if row['frame'] == frame)
        with np.load(directory / pixel_row['maps'], allow_pickle=False) as source:
            maps = {key: source[key].copy() for key in source.files}
        metrics = {}
        for key, (a, b) in PAIRS.items():
            spatial, centers, density = voxel_gap(clouds[a], clouds[b], voxel_size)
            pixels = pixel_row['pairs'][key]
            values = {'voxel': spatial, 'pixel': pixels['camera_pixel'], 'overlay_rgb': pixels['overlay_rgb'],
                      'range_pixel': pixels['range_pixel'], 'covariates': {}}
            maps[key + '_voxel_centers'] = centers.astype('float32')
            maps[key + '_voxel_density_difference'] = density.astype('float32')
            for feature in features[a].keys() & features[b].keys():
                values['covariates'][feature] = distribution_gap(features[a][feature]['counts'], features[b][feature]['counts'])
            metrics[key] = values
        name = f'comparison/maps_{frame:03d}.npz'
        np.savez_compressed(directory / name, **maps)
        frames.append({'frame': frame, 'pairs': metrics, 'covariates': features, 'maps': name, 'images': pixel_row['images'],
                       'camera_skew_ms': pixel_row['camera_skew_ms']})
        get_logger().info('Computed full-resolution pixel/voxel gaps for frame %d', frame)
    result = {**report['comparison'], 'voxel_size_m': voxel_size, 'frames': frames,
        'status': 'completed' if all(frame['images'] for frame in frames) else 'pointcloud_only',
        'camera_diagnostic': next((row['camera_reason'] for row in pixel_frames if row['camera_reason']), None),
        'calibration_convention': overlays.get('calibration_convention'),
        'definitions': {'js_distance': 'Square root of Jensen-Shannon divergence, log base 2; range 0..1.',
            'range_pixel': 'LiDAR beam/azimuth grid; radial range in metres, independent of camera availability.',
            'chamfer_mean_m': 'Mean of the two directed mean Euclidean nearest-neighbour distances; unsquared metres.',
            'voxel': 'Sparse occupancy/count distribution on a shared sensor-frame grid with origin (0,0,0).',
            'pixel': 'Camera-plane optical depth on shared calibrated pixels; missing returns are never filled.',
            'overlay_rgb': 'RGB discrepancy on union foreground dilated by one pixel, excluding label rows 0..24. Shared recorded background is not independently generated RGB.'},
        'limitations': ['Two paired synthetic windows provide descriptive discrepancies, not a statistical covariate-shift test or population estimate.',
            'Both arms use the same real conditioning, sampling seed, ray mask and range-clamping rules; geometry agreement is constrained by construction.',
            ('Both arms use the same OT output refinement; no calibration profile or learned controller is used.'
             if report.get('ot_alignment', {}).get('enabled') else
             'No calibration postprocessor or learned OT controller is used in this weights-only comparison.'),
            'RGB overlays share the same recorded photo; image histograms do not establish synthetic RGB fidelity.',
            'Missing support yields null metrics rather than a fabricated zero. Camera mounting, timestamp and occlusion limitations still apply.']}
    write_json(directory / 'comparison.json', result)
    from .comparison_viewer import write_viewer
    write_viewer(directory, result)
    return result
