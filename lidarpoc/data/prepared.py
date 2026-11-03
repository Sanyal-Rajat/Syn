"""Discover completed preparations from persisted server jobs."""
import json
from pathlib import Path

from .split_manifest import load_manifest


def prepared_manifests(runs):
    records, seen = [], set()
    for state_file in sorted(Path(runs).glob('*/state.json'), key=lambda path: path.stat().st_mtime, reverse=True):
        try:
            state = json.loads(state_file.read_text())
            if state['action'] != 'prepare' or state['status'] != 'completed':
                continue
            result = json.loads(state_file.with_name('result.json').read_text())
            if result['action'] != 'prepare':
                continue
            path = Path(result['manifest']).expanduser().resolve()
            manifest = load_manifest(path)
            if str(path) in seen:
                continue
            sensor = manifest['sensor_meta']
            records.append({'job_id': state['id'], 'manifest': str(path),
                            'height': len(sensor['elevations']), 'width': sensor['width'],
                            'traces': sum(len(traces) for traces in manifest['splits'].values())})
            seen.add(str(path))
        except (OSError, ValueError, KeyError, TypeError):
            continue  # Failed/incomplete/removed preparations are not selectable.
    return records
