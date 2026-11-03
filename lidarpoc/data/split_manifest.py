"""Prepare each whole driving trace once; persist safe, projected per-frame caches."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .pandar_reader import in_sensor_frame, load_arrays, sensor_transform, unpack_scan
from .scan_to_range import infer_elevations
from ..logs import get_logger


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
    from .prepare_tasks import prepare_trace
    from ..scheduling import device_groups, map_tasks

    gpus = 0
    if config.gpus:
        try:
            from ..gpu import resolve_gpus
            gpus = resolve_gpus(config.gpus)
        except ImportError as error:
            if config.gpus > 0:
                raise
            get_logger().warning('GPU projection unavailable; using independent CPU preparation: %s', error)
    resolved = config.model_copy(update={'gpus': gpus})
    tasks = [{'config': resolved.model_dump(), 'path': str(path), 'output': str(output),
              'split': split, 'transform': transform.tolist(), 'elevations': elevations.tolist()}
             for split, paths in files.items() for path in paths]
    groups = None
    if gpus > 1:
        import psutil
        import zipfile
        largest = 0
        for task in tasks:
            with zipfile.ZipFile(task['path']) as archive:
                largest = max(largest, sum(info.file_size for info in archive.infolist()))
        slots = max(1, int(psutil.virtual_memory().available * .6 / max(1, 2 * largest)))
        groups = device_groups(gpus, min(len(tasks), slots))
    from ..logs import progress
    results = map_tasks(prepare_trace, tasks, gpus, groups=groups,
        on_done=lambda done, total: progress(output, 'preparing whole traces', done, total))
    trace_splits, source_splits, fingerprints = {}, {}, {}
    for result in results:
        split, entry = result['split'], result['entry']
        for mapping, key, kind in [(trace_splits, entry['sequence_id'], 'trace'),
                                    (source_splits, entry['source_sha256'], 'source hash')]:
            if key in mapping and mapping[key] != split:
                raise ValueError(f'Cross-split leakage ({kind}): {entry["sequence_id"]}')
            mapping[key] = split
        for fingerprint in result['fingerprints']:
            if fingerprint in fingerprints and fingerprints[fingerprint] != split:
                raise ValueError(f'Cross-split identical/near-identical projected frame: {entry["sequence_id"]}')
            fingerprints[fingerprint] = split
        manifest['splits'][split].append(entry)
    manifest['preparation_compute'] = {'gpus': gpus, 'trace_workers': len(groups) if groups else 1,
                                       'projection': 'torch_cuda' if gpus else 'numpy_cpu'}
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
