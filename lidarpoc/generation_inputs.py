"""Resolve generation files and discover matching saved training checkpoints."""
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


def recover_checkpoints(runs, manifest_sha256):
    rows = []
    for path in sorted(Path(runs).rglob('checkpoint.pt'), key=lambda path: path.stat().st_mtime, reverse=True):
        try:
            root = Path(runs) / path.relative_to(runs).parts[0]
            state_path = root / 'state.json'
            if state_path.is_file() and json.loads(state_path.read_text())['status'] == 'running':
                continue
            history = json.loads(path.with_name('training.json').read_text())
            if (not isinstance(history, list) or not history or not isinstance(history[-1], dict)
                    or history[-1].get('epoch', 0) < 1):
                continue
            window = training_window(path.parent, manifest_sha256)
            if window is not None and path.stat().st_size:
                rows.append({'job_id': root.name, 'checkpoint': str(path.resolve()), 'window': window})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return rows


def successful_training_runs(runs):
    for state_path in sorted(Path(runs).glob('*/state.json'), key=lambda path: path.stat().st_mtime, reverse=True):
        try:
            state = json.loads(state_path.read_text())
            if state['action'] != 'train' or state['status'] != 'completed':
                continue
            request = json.loads(state_path.with_name('request.json').read_text())
            result = json.loads(state_path.with_name('result.json').read_text())
            manifest = Path(request['manifest']).expanduser().resolve()
            load_manifest(manifest)
            window = request['window']
            checkpoint = resolve_checkpoint(result['checkpoint'])
            if (result['action'] == 'train' and type(window) is int and 1 <= window <= 5
                    and Path(checkpoint).stat().st_size):
                yield {'job_id': state['id'], 'manifest': str(manifest), 'checkpoint': checkpoint,
                       'window': window, 'request': request, 'result': result, 'directory': state_path.parent}
        except (OSError, ValueError, KeyError, TypeError):
            continue


def generation_defaults(run):
    from .config import RunConfig

    names = ('gpus', 'sample_steps', 'samples_per_trace', 'auto_optimize', 'auto_conditioning',
             'seed', 'batch_size', 'diffusion_steps', 'window', 'camera_overlay', 'camera_model',
             'camera_timestamp_unit', 'camera_max_skew_ms', 'camera_calibration')
    values = RunConfig(action='generate', manifest=run['manifest'], checkpoint=run['checkpoint'],
                       **{name: run['request'][name] for name in names if name in run['request']}).model_dump()
    defaults = {name: values[name] for name in names}
    defaults['batch_size'] = 0  # Re-evaluate memory instead of inheriting an old training ceiling.
    defaults.update(manifest=run['manifest'], checkpoint=run['checkpoint'], conditioning='',
                    profile_file='', profile_name='uncalibrated', controller_file='')
    controller = run['result'].get('generation_controller')
    if controller:
        path = Path(controller).expanduser().resolve()
        if path.is_file() and path.is_relative_to(run['directory'].resolve()):
            defaults['controller_file'] = str(path)
    path = run['directory'] / 'profiles.json'
    try:
        profiles = json.loads(path.read_text())['profiles']
        selected = run['result'].get('selected_profile_on_validation', 'uncalibrated')
        if selected in profiles:
            defaults.update(profile_file=str(path.resolve()), profile_name=selected)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return defaults


def prepare_generation_inputs(runs, manifest_path='', checkpoint='', auto_conditioning=True):
    completed = list(successful_training_runs(runs))
    if not manifest_path:
        if not completed:
            return {'manifest': '', 'checkpoint': '', 'conditioning': '', 'checkpoints': [], 'defaults': {},
                    'training_job_id': None, 'diagnostic': f'No successful training run was found under {Path(runs).resolve()}.'}
        manifest_path = completed[0]['manifest']
    manifest = load_manifest(manifest_path)
    fingerprint = digest(manifest_path)
    rows = [{'job_id': run['job_id'], 'checkpoint': run['checkpoint'], 'window': run['window']}
            for run in completed if digest(run['manifest']) == fingerprint]
    recovered = not rows
    if recovered:
        rows = recover_checkpoints(runs, fingerprint)
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
    diagnostic = ''
    if not selected:
        diagnostic = (f'No matching checkpoint.pt was found under the server runs folder: {Path(runs).resolve()}. '
                      'Select the manifest used for training, or enter the checkpoint file path manually.')
    elif recovered and rows and not checkpoint:
        diagnostic = 'Using a saved checkpoint whose job has no completed training report; model provenance is checked before generation.'
    source = next((run for run in completed if run['checkpoint'] == selected), None)
    defaults = generation_defaults(source) if source is not None else {}
    if defaults:
        defaults['conditioning'] = scene
    return {'manifest': str(Path(manifest_path).resolve()), 'checkpoint': selected, 'conditioning': scene,
            'checkpoints': rows, 'diagnostic': diagnostic, 'defaults': defaults,
            'training_job_id': source['job_id'] if source else None}
