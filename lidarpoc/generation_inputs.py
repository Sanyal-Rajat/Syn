"""Resolve generation files and discover matching completed training jobs."""
import json
from pathlib import Path

from .data.split_manifest import digest, load_manifest


def resolve_checkpoint(value):
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path /= 'checkpoint.pt'
    if not path.is_file():
        raise ValueError(f'Checkpoint file not found: {path}. Select checkpoint.pt from a completed training job.')
    return str(path)


def training_window(directory, manifest_sha256):
    request = json.loads((Path(directory) / 'request.json').read_text())
    if digest(Path(request['manifest']).expanduser()) != manifest_sha256:
        return None
    window = request['window']
    return window if type(window) is int and 1 <= window <= 5 else None


def prepare_generation_inputs(runs, manifest_path, checkpoint='', auto_conditioning=True):
    manifest = load_manifest(manifest_path)
    fingerprint = digest(manifest_path)
    rows = []
    for state_path in sorted(Path(runs).glob('*/state.json'), key=lambda path: path.stat().st_mtime, reverse=True):
        try:
            state = json.loads(state_path.read_text())
            if state['action'] != 'train' or state['status'] != 'completed':
                continue
            window = training_window(state_path.parent, fingerprint)
            if window is None:
                continue
            result = json.loads(state_path.with_name('result.json').read_text())
            if result['action'] == 'train':
                rows.append({'job_id': state['id'], 'checkpoint': resolve_checkpoint(result['checkpoint']), 'window': window})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    selected = resolve_checkpoint(checkpoint) if checkpoint else rows[0]['checkpoint'] if rows else ''
    scene = ''
    if selected and auto_conditioning:
        try:
            window = training_window(Path(selected).parent, fingerprint)
        except (OSError, ValueError, KeyError, TypeError):
            window = None  # A manually moved checkpoint can still generate; the runner reads its window.
        if window is not None:
            from .data.conditioning import export_real_scene
            folder = Path(runs).parent / 'datasets' / 'conditioning' / fingerprint / f'window_{window}'
            path = folder / 'real_scene.npz'
            if not path.is_file():
                export_real_scene(manifest, window, folder)
            scene = str(path.resolve())
    return {'checkpoint': selected, 'conditioning': scene, 'checkpoints': rows}
