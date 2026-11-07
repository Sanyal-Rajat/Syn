"""Discover completed preparations and recover validated manifests from saved jobs."""
import json
from pathlib import Path

from .split_manifest import load_manifest


def prepared_manifests(runs):
    records, seen = [], set()

    def add(path, job_id, source):
        path = Path(path).expanduser().resolve()
        if str(path) in seen:
            return
        manifest = load_manifest(path)
        sensor = manifest['sensor_meta']
        records.append({'job_id': job_id, 'manifest': str(path), 'source': source,
                        'height': len(sensor['elevations']), 'width': sensor['width'],
                        'traces': sum(len(traces) for traces in manifest['splits'].values())})
        seen.add(str(path))

    for state_file in sorted(Path(runs).glob('*/state.json'), key=lambda path: path.stat().st_mtime, reverse=True):
        try:
            state = json.loads(state_file.read_text())
            if state['action'] != 'prepare' or state['status'] != 'completed':
                continue
            result = json.loads(state_file.with_name('result.json').read_text())
            if result['action'] != 'prepare':
                continue
            add(result['manifest'], state['id'], 'completed_preparation')
        except (OSError, ValueError, KeyError, TypeError):
            continue
    # A report/export failure can leave fully prepared data without a completed job.
    # The manifest is written only after every trace and split audit finishes.
    for path in sorted(Path(runs).glob('*/prepared/manifest.json'), key=lambda path: path.stat().st_mtime, reverse=True):
        try:
            directory = path.parent.parent
            state_file = directory / 'state.json'
            if state_file.is_file() and json.loads(state_file.read_text()).get('status') == 'running':
                continue
            add(path, directory.name, 'recovered_preparation')
        except (OSError, ValueError, KeyError, TypeError):
            continue
    # Completed training/generation/experiments may reference data outside this runs folder.
    for request_file in sorted(Path(runs).glob('*/request.json'), key=lambda path: path.stat().st_mtime, reverse=True):
        try:
            state = json.loads(request_file.with_name('state.json').read_text())
            request = json.loads(request_file.read_text())
            if state['status'] == 'completed' and state['action'] in ('train', 'generate', 'experiment'):
                add(request['manifest'], state['id'], 'completed_job_reference')
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return records
