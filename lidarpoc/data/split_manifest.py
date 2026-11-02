"""Prepare each whole driving trace once; persist safe, projected per-frame caches."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .pandar_reader import in_sensor_frame, load_arrays, sensor_transform, unpack_scan
from .scan_to_range import infer_elevations, project
from ..logs import get_logger
from .annotations import load_poses, point_labels


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def prepare(config, output, evidence):
    root, output = Path(config.dataset_root).expanduser().resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    transform = sensor_transform(config.calibration_file, config.input_frame)
    files = {split: sorted((root / split).glob('**/*_scan_list.npz')) for split in ('train', 'validation', 'test')}
    if any(not paths for paths in files.values()):
        raise ValueError('Extracted dataset needs train/, validation/, test/ trace directories')
    beam_fit = None
    if config.elevations_file:
        elevations = np.deg2rad(np.loadtxt(config.elevations_file))
        beam_source = 'provided_calibration_degrees'
    elif config.infer_beams:
        first = load_arrays(files['train'][0])
        scans = [in_sensor_frame(unpack_scan(frame)[0], transform) for frame in first[:20]]
        elevations, beam_fit = infer_elevations(scans, config.height, config.beam_tolerance_deg, return_diagnostics=True)
        del first, scans
        beam_source = 'empirical_training_elevation_bins_not_factory_ring_ids'
    else:
        raise ValueError('Configuration.json contains mounting/FOV only. Supply beam elevations or enable training-only inference.')
    if len(elevations) != config.height:
        raise ValueError(f'Expected {config.height} beam elevations, found {len(elevations)}')
    manifest = {'dataset': evidence['dataset']['repo_id'], 'revision': evidence['dataset']['revision'],
                'sensor_meta': {'elevations': elevations.tolist(), 'azimuths': ((np.arange(config.width) + .5) / config.width * 2 * np.pi - np.pi).tolist(),
                                'width': config.width, 'max_range_m': config.max_range_m,
                                'intensity_normalization_max': config.intensity_max,
                                'beam_tolerance_deg': config.beam_tolerance_deg,
                                'calibration_sha256': digest(config.calibration_file),
                                'beam_source': beam_source, 'input_frame': config.input_frame,
                                'input_to_sensor': transform.tolist(), 'timestamp_source': 'nominal_20Hz_sequence_relative'},
                'splits': {split: [] for split in files}, 'leakage_audit': 'whole-trace IDs, source hashes, exact and quantized projected-frame hashes'}
    if beam_fit is not None:
        manifest['sensor_meta']['beam_fit'] = beam_fit
    trace_splits, source_splits, fingerprints = {}, {}, {}
    for split, paths in files.items():
        for path in paths:
            trace = path.name.removesuffix('_scan_list.npz')
            get_logger().info('Preparing %s trace %s', split, trace)
            source_hash = digest(path)
            for mapping, key, kind in [(trace_splits, trace, 'trace'), (source_splits, source_hash, 'source hash')]:
                if key in mapping and mapping[key] != split:
                    raise ValueError(f'Cross-split leakage ({kind}): {trace}')
                mapping[key] = split
            frames = load_arrays(path)
            poses = load_poses(path, config.odometry_mode,
                               np.linalg.inv(sensor_transform(config.calibration_file, 'vehicle'))) if config.odometry_mode != 'disabled' else None
            if poses is not None and len(poses) < min(config.max_frames, len(frames)):
                raise ValueError(f'{trace}: fewer odometry records than scan frames')
            entry = {'sequence_id': trace, 'source_file': str(path), 'source_sha256': source_hash,
                     'source_frames': len(frames), 'schema': str(np.asarray(frames[0]).dtype), 'frames': []}
            folder = output / split / trace
            folder.mkdir(parents=True, exist_ok=True)
            entry['odometry_mode'] = config.odometry_mode if poses is not None else 'unavailable'
            entry['labels_source'] = 'native Classification field; semantic validity requires confirmation'
            if poses is not None:
                entry['odometry_sha256'] = digest(Path(str(path).replace('_scan_list.npz', '_odom_list.npz')))
            for frame_index, raw in enumerate(frames[:config.max_frames]):
                xyz, intensity = unpack_scan(raw)
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
                if fingerprint in fingerprints and fingerprints[fingerprint] != split:
                    raise ValueError(f'Cross-split identical/near-identical projected frame: {trace}:{frame_index}')
                fingerprints[fingerprint] = split
                cache = folder / f'{frame_index:06d}.npz'
                np.savez_compressed(cache, **projected, timestamp_ns=np.int64(frame_index * 50_000_000))
                entry['frames'].append({'path': str(cache), 'sha256': digest(cache), 'index': frame_index})
            if len(entry['frames']) < config.window:
                raise ValueError(f'{trace}: fewer frames than the temporal window')
            manifest['splits'][split].append(entry)
            get_logger().info('Prepared %s: %d/%d frames', trace, len(entry['frames']), len(frames))
            del frames
    target = output / 'manifest.json'
    target.write_text(json.dumps(manifest, indent=2))
    return str(target)


def load_manifest(path, verify_files=False):
    manifest = json.loads(Path(path).read_text())
    seen_ids, seen_hashes = set(), set()
    for split in ('train', 'validation', 'test'):
        if not manifest['splits'].get(split):
            raise ValueError(f'Empty {split} split')
        split_ids, split_hashes = set(), set()
        for trace in manifest['splits'][split]:
            split_ids.add(trace['sequence_id'])
            split_hashes.add(trace['source_sha256'])
            if verify_files:
                for frame in trace['frames']:
                    if digest(frame['path']) != frame['sha256']:
                        raise ValueError(f'Prepared frame hash mismatch: {frame["path"]}')
        if seen_ids & split_ids or seen_hashes & split_hashes:
            raise ValueError('Cross-split trace/source leakage')
        seen_ids |= split_ids
        seen_hashes |= split_hashes
    return manifest


def load_window(records, sensor_meta, sequence_id):
    frames = []
    for record in records:
        with np.load(record['path'], allow_pickle=False) as frame:
            frames.append({key: frame[key].copy() for key in ('range', 'valid', 'intensity', 'has_intensity', 'timestamp_ns')})
            for key in ('labels', 'world_from_sensor'):
                if key in frame.files:
                    frames[-1][key] = frame[key].copy()
    result = {key: np.stack([frame[key] for frame in frames]) for key in set.intersection(*(set(frame) for frame in frames))}
    result['sequence_id'], result['sensor_meta'] = sequence_id, sensor_meta
    return result


def windows(manifest, split, window):
    result = []
    for trace in manifest['splits'][split]:
        for start in range(0, len(trace['frames']) - window + 1, window):
            result.append((trace['sequence_id'], trace['frames'][start:start + window]))
    return result
