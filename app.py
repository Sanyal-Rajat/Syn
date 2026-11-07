"""Thin Streamlit client: all data processing, generation and RL run remotely."""
import json
from pathlib import Path
import subprocess

import httpx
import numpy as np
import pandas as pd
import streamlit as st

from lidarpoc.logs import get_logger, sync_remote_log
from lidarpoc.methods.registry import CALIBRATION_METHODS, CONTROLLER_METHODS

LOGGER = get_logger()

st.set_page_config(page_title='Pandar128 generation', layout='wide')
st.title('Pandar128 LiDAR generation')
st.caption('LidarDM range-image adaptation · random-initialized diffusion · RL calibration · remote compute over SSH')

with st.sidebar:
    st.subheader('Remote connection')
    local_port = st.number_input('Local tunnel port', 1024, 65535, 9000)
    endpoint = f'http://127.0.0.2:{local_port}'
    with st.expander('SSH tunnel'):
        host = st.text_input('SSH host or configured alias')
        user = st.text_input('SSH user (optional for an alias)')
        key = st.text_input('Private key path on this client')
        control_socket = st.text_input('Existing SSH control socket (optional)',
                                       help='Use an existing ControlMaster connection. Supply its absolute socket path.')
        ssh_port = st.number_input('SSH port', 1, 65535, 22)
        remote_port = st.number_input('Remote API port', 1024, 65535, 9000)
        a, b, c = st.columns(3)
        command = 'start' if a.button('Connect') else 'status' if b.button('Status') else 'stop' if c.button('Close') else None
        if command:
            args = ['bash', str(Path(__file__).with_name('ssh_tunnel.sh')), command, '-l', str(local_port)]
            if control_socket:
                args += ['-S', control_socket]
            if command == 'start':
                args += ['-H', host, '-P', str(ssh_port), '-p', str(remote_port)]
                if user:
                    args += ['-u', user]
                if key:
                    args += ['-k', key]
            try:
                connected = False
                if command == 'start':
                    try:
                        probe = httpx.request('GET', endpoint + '/health', timeout=2, trust_env=False)
                        connected = probe.is_success and probe.json().get('status') == 'ok'
                    except (httpx.HTTPError, ValueError):
                        pass
                if connected:
                    message = f'Using existing API tunnel at {endpoint}'
                    st.success(message)
                    LOGGER.info(message)
                else:
                    response = subprocess.run(args, capture_output=True, text=True, timeout=15)
                    (st.success if response.returncode == 0 else st.error)(response.stdout + response.stderr)
                    LOGGER.info('SSH %s: %s%s', command, response.stdout, response.stderr)
            except (OSError, subprocess.TimeoutExpired) as error:
                st.error(str(error))
                LOGGER.error('SSH %s: %s', command, error)


def api(method, path, **kwargs):
    response = httpx.request(method, endpoint + path, timeout=kwargs.pop('timeout', 10), trust_env=False, **kwargs)
    response.raise_for_status()
    return response


def json_setting(label, default):
    value = st.text_input(label, json.dumps(default))
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        st.error(f'{label}: invalid JSON ({error.msg})')
        st.stop()


def training_charts(rows):
    history = pd.DataFrame(rows).set_index('epoch')
    losses = [name for name in history if not name.endswith(('_m', '_frames'))]
    if losses:
        st.line_chart(history[losses])
    distances = {name: label for name, label in (
        ('train_range_w1_m', 'During training'), ('validation_range_w1_m', 'Generated validation scenes'))
        if name in history}
    if distances:
        st.caption('Distance difference: lower means a closer match. ' +
                   ('The best validation result selects the checkpoint.' if 'validation_range_w1_m' in distances else
                    'Training uses differentiable return probabilities.'))
        st.line_chart(history[list(distances)].rename(columns=distances),
                      x_label='Epoch', y_label='Distance difference (metres)')


def manifest_input(default='', generation=False):
    try:
        prepared = api('GET', '/dataset/prepared').json()
    except httpx.HTTPError:
        prepared = []
        st.caption('Prepared dataset list unavailable. Enter a manifest path below, or update the remote API.')
    if not prepared and not default:
        try:
            detected = api('POST', '/generation/inputs', json={'auto_conditioning': False}).json()
            default = detected.get('manifest', '')
        except httpx.HTTPError:
            pass  # Older servers can still accept an explicitly entered manifest.
    paths = [row['manifest'] for row in prepared]
    manual = 'Enter a server path manually'
    prefix = 'generate_' if generation else ''
    saved_key, field_key, choice_key = (f'{prefix}{name}:{endpoint}' for name in ('manifest_saved', 'manifest_field', 'manifest_choice'))
    saved = st.session_state.setdefault(saved_key, {'path': '', 'manual': False})
    if not generation and st.button('Use latest prepared dataset', key=f'{prefix}manifest_detect:{endpoint}'):
        saved.update(path='', manual=False)
        st.session_state[field_key] = ''
        st.session_state[choice_key] = manual
    if default and not saved['manual']:
        saved['path'] = default
        st.session_state[field_key] = default
    if paths and not saved['path'] and not saved['manual']:
        saved['path'] = paths[0]
        st.session_state[field_key] = saved['path']
        st.session_state[choice_key] = saved['path']
    if field_key not in st.session_state:
        st.session_state[field_key] = saved['path']
    if st.session_state.get(choice_key) not in paths + [manual]:
        st.session_state[choice_key] = saved['path'] if saved['path'] in paths else manual

    def select():
        chosen = st.session_state[choice_key]
        path = '' if chosen == manual else chosen
        st.session_state[saved_key].update(path=path, manual=True)
        st.session_state[field_key] = path

    def edit():
        path = st.session_state[field_key].strip()
        st.session_state[saved_key].update(path=path, manual=bool(path))
        st.session_state[choice_key] = path if path in paths else manual

    if prepared:
        rows = {row['manifest']: row for row in prepared}
        st.selectbox('Completed preparation', paths + [manual], key=choice_key, on_change=select,
            format_func=lambda path: manual if path == manual else
                f"{rows[path]['job_id'][:12]} · {rows[path]['height']}×{rows[path]['width']} · {rows[path]['traces']} traces",
            help='The latest completed preparation is selected initially. You can choose another or enter a custom path.')
        selected = rows.get(saved['path'], {})
        if selected.get('source') in ('recovered_preparation', 'completed_job_reference'):
            st.caption('Using validated prepared data recovered from saved server files.')
    elif not saved['path']:
        st.info('Complete Prepare Pandar128 on this server, or enter an existing prepared manifest path below.')
    return st.text_input('Prepared manifest path on the server', key=field_key,
                         placeholder='/data/cache/manifest.json', on_change=edit).strip()


def generation_saved_path(name):
    key = f'{name}_saved:{endpoint}'
    saved = st.session_state.setdefault(key, {'path': '', 'manual': False})
    if isinstance(saved, str):
        saved = st.session_state[key] = {'path': saved, 'manual': bool(saved)}
    return saved


def generation_path_input(label, name, default='', help=None, disabled=False):
    saved = generation_saved_path(name)
    field_key = f'{name}_field:{endpoint}'
    if not saved['manual']:
        saved['path'] = default
        st.session_state[field_key] = default
    elif field_key not in st.session_state:
        st.session_state[field_key] = saved['path']

    def edit():
        path = st.session_state[field_key].strip()
        saved.update(path=path, manual=bool(path) if name == 'checkpoint' else True)

    return st.text_input(label, key=field_key, on_change=edit, help=help, disabled=disabled).strip()


def generation_setting(name, default):
    saved = st.session_state.setdefault(f'generation_settings:{endpoint}', {})
    key = f'generation_{name}:{endpoint}'
    if key not in st.session_state:
        st.session_state[key] = saved.setdefault(name, default)

    def edit():
        saved[name] = st.session_state[key]

    return {'key': key, 'on_change': edit}


def restore_generation_defaults(detected, force=False):
    source = detected.get('training_job_id')
    defaults = detected.get('defaults', {})
    source_key = f'generation_source:{endpoint}'
    if not source or not defaults or (not force and st.session_state.get(source_key) == source):
        return
    for name in ('checkpoint', 'conditioning', 'profile_file', 'profile_name', 'controller_file', 'camera_calibration'):
        generation_saved_path(name).update(path=defaults.get(name, ''), manual=False)
        st.session_state[f'{name}_field:{endpoint}'] = defaults.get(name, '')
    st.session_state[f'generate_manifest_saved:{endpoint}'] = {'path': defaults['manifest'], 'manual': False}
    st.session_state[f'generate_manifest_field:{endpoint}'] = defaults['manifest']
    st.session_state[f'generate_manifest_choice:{endpoint}'] = defaults['manifest']
    saved = st.session_state.setdefault(f'generation_settings:{endpoint}', {})
    for name in ('gpus', 'sample_steps', 'samples_per_trace', 'auto_optimize', 'auto_conditioning', 'seed',
                 'camera_overlay', 'camera_model', 'camera_timestamp_unit', 'camera_max_skew_ms'):
        if name in defaults:
            saved[name] = defaults[name]
            st.session_state[f'generation_{name}:{endpoint}'] = defaults[name]
    st.session_state[source_key] = source


try:
    health = api('GET', '/health').json()
except httpx.HTTPError:
    st.info('Start the remote API and connect the SSH tunnel to this client.')
    st.code('Remote: python server.py\n'
            'Client: bash ssh_tunnel.sh start -H YOUR_SERVER -u YOUR_USER\n'
            'Client: streamlit run app.py --server.address 127.0.0.1 --server.port 9000')
    st.stop()

st.sidebar.success('Remote API connected')
with st.sidebar.expander('Remote hardware'):
    st.json(health['hardware'])

configure, results = st.tabs(['Configure', 'Monitor and results'])
with configure:
    task = st.radio('Operation', ['Prepare Pandar128', 'Train diffusion + calibration', 'Generate', 'OT + diffusion + RL experiment'], horizontal=True)
    action = 'prepare' if task.startswith('Prepare') else 'train' if task.startswith('Train') else 'experiment' if task.startswith('OT') else 'generate'
    config = {'action': action}
    if action == 'prepare':
        st.caption('Preparation automatically downloads and extracts Pandar128 on the server. Completed files are reused.')
        st.caption(f"Default server dataset folder: {health.get('dataset_root', '<project>/datasets/pandar128')}")
        config['dataset_root'] = st.text_input('Dataset directory on the server (optional override)',
                                              placeholder='Leave blank for the project datasets/pandar128 folder')
        left, right = st.columns(2)
        config['input_frame'] = left.selectbox('Input coordinate frame', ['vehicle', 'sensor'],
                                               help='Confirm against the publisher scan schema. Vehicle points are transformed with Configuration.json.')
        beam_source = right.selectbox('Beam elevations', ['Estimate from training scans', 'Measured calibration file'])
        config['infer_beams'] = beam_source.startswith('Estimate')
        config['elevations_file'] = st.text_input('Server path to 128 elevations in degrees, sorted ascending',
                                                 disabled=config['infer_beams'])
        if config['infer_beams']:
            st.caption('Estimation uses training scans only and must pass a spacing/residual check. The resulting rows are elevation bins.')
        config['width'] = st.selectbox('Azimuth columns', [256, 512, 1024, 2048], index=1)
        config['intensity_max'] = st.number_input('Intensity normalization maximum', min_value=.001, value=255.)
        config['max_frames'] = st.number_input('Frames to prepare per trace', 2, 100000, 128)
        config['odometry_mode'] = st.selectbox('Odometry convention (confirm before enabling)',
            ['disabled', 'world_from_vehicle', 'previous_from_current'], help='Publisher conventions must be confirmed; no identity poses are invented.')
        st.caption('Inspect NPZ schemas checks existing files. Start remote job downloads missing data before preparation.')
        if st.button('Inspect NPZ schemas'):
            try:
                st.dataframe(api('POST', '/dataset/inspect', json={'root': config['dataset_root']}, timeout=120).json()['recordings'])
            except httpx.HTTPError as error:
                st.error(str(error))
    else:
        latest = {}
        if action == 'generate':
            try:
                latest = api('POST', '/generation/inputs', json={'auto_conditioning': False}).json()
            except httpx.HTTPError:
                pass  # Keep manual entry available with older/unavailable APIs.
            reset = st.button('Use detected paths', help='Restore every Generate field from the last successful training run.')
            restore_generation_defaults(latest, force=reset)
            if reset and not latest.get('defaults'):
                for name in ('checkpoint', 'conditioning', 'profile_file', 'profile_name', 'controller_file', 'camera_calibration'):
                    generation_saved_path(name).update(path='', manual=False)
                    st.session_state[f'{name}_field:{endpoint}'] = ''
            if latest.get('training_job_id'):
                st.caption(f"Defaults from successful training job {latest['training_job_id'][:12]}. Manual edits remain available.")
        defaults = latest.get('defaults', {})
        config['manifest'] = manifest_input(defaults.get('manifest', ''), generation=action == 'generate')
        setting = lambda name, default: generation_setting(name, default) if action == 'generate' else {'value': default}
        paired = False
        if action == 'generate':
            paired = config['compare_training'] = st.checkbox('Compare without training / with training (one output each)',
                **setting('compare_training', False),
                help='A fresh random-initialized copy of the same diffusion architecture versus your trained checkpoint, using the same scene and sampling seed.')
            if paired:
                st.caption('Exports two synthetic NPZ windows, their recorded-camera overlays and an interactive pixel/voxel gap viewer. Calibration profiles and OT controllers are bypassed to isolate diffusion weights.')
                a, b = st.columns(2)
                config['comparison_init_seed'] = a.number_input('Untrained model initialization seed', 0, 2147483647,
                    **setting('comparison_init_seed', 0))
                config['comparison_voxel_m'] = b.number_input('Comparison voxel size (metres)', .01, 5.,
                    step=.05, **setting('comparison_voxel_m', .25))
        left, middle, right = st.columns(3)
        available_gpus = len(health['hardware']['gpus'])
        gpu_options = setting('gpus', -1 if available_gpus else 0)
        if action == 'generate':
            key = gpu_options['key']
            st.session_state[key] = min(st.session_state[key], available_gpus) if available_gpus else 0
        config['gpus'] = left.number_input('Remote GPUs (-1 = all, 0 = CPU)', -1, max(1, available_gpus),
            **gpu_options,
            help='All visible GPUs are used by default. Set a positive count to limit the job.')
        selected_gpus = available_gpus if config['gpus'] == -1 else config['gpus']
        st.caption(f'{selected_gpus} remote GPU(s) selected. Training uses one process per GPU; independent samples run in parallel.')
        config['sample_steps'] = middle.number_input('Sampling steps', 2, 200, **setting('sample_steps', 50))
        if paired:
            config['samples_per_trace'] = 2
            right.metric('Synthetic outputs', '2', help='One untrained and one trained NPZ, each containing the checkpoint temporal window.')
        else:
            config['samples_per_trace'] = right.number_input('Evaluation windows per trace / generated samples', 1, 100, **setting('samples_per_trace', 4))
        config['batch_size'] = 0
        if action == 'generate':
            config['auto_optimize'] = st.checkbox('Automatically optimize GPU throughput', **setting('auto_optimize', True),
                help='Adapt inference batches to measured throughput and available GPU memory.')
        else:
            config['auto_optimize'] = True
            st.caption('Training batch size is selected automatically from the data, available GPU memory and measured speed.')
        if action == 'generate':
            config['auto_conditioning'] = st.checkbox('Automatically extract a real scene from the prepared dataset', **setting('auto_conditioning', True),
                help='Create or reuse the first complete training window matching the checkpoint. Fill the scene path automatically and export real_scene.npz beside the synthetic samples.')
            detected, lookup_error = {}, ''
            if config['manifest']:
                saved_checkpoint = generation_saved_path('checkpoint')
                saved_scene = generation_saved_path('conditioning')
                try:
                    detected = api('POST', '/generation/inputs', json={'manifest': config['manifest'],
                        'checkpoint': saved_checkpoint['path'] if saved_checkpoint['manual'] else '',
                        'auto_conditioning': config['auto_conditioning'] and not (saved_scene['manual'] and saved_scene['path'])}).json()
                except httpx.HTTPError as error:
                    detail = error.response.text if isinstance(error, httpx.HTTPStatusError) else str(error)
                    if isinstance(error, httpx.HTTPStatusError) and error.response.status_code in (404, 405):
                        lookup_error = ('Automatic checkpoint lookup is unavailable on the connected server. '
                            'Update the server code and restart its API, then click Use detected paths.')
                    else:
                        lookup_error = f'Automatic checkpoint lookup failed: {detail}'
            config['checkpoint'] = generation_path_input('Own checkpoint path on the server', 'checkpoint', detected.get('checkpoint', ''),
                help='Automatically selects the latest completed training checkpoint matching this manifest. You can enter a checkpoint file or its training job directory. Manual entries are remembered across operations.')
            config['conditioning'] = generation_path_input('Optional synthetic scene NPZ on the server', 'conditioning', detected.get('conditioning', ''),
                help='Automatically populated with a real training window when enabled. You can override it with matching range/valid arrays [T,H,W].')
            if config['checkpoint'] and (lookup_error or detected.get('diagnostic')):
                st.caption(lookup_error or detected['diagnostic'])
            st.caption('The real scene and synthetic samples are included in the artifact download. '
                       'To generate without conditioning, turn off automatic extraction and leave the scene path blank.')
            selected_defaults = detected.get('defaults', {})
            for name in ('diffusion_steps', 'window'):
                if name in selected_defaults:
                    config[name] = selected_defaults[name]
            config['profile_file'] = generation_path_input('Optional calibration profiles.json on the server', 'profile_file', selected_defaults.get('profile_file', ''), disabled=paired)
            config['profile_name'] = generation_path_input('Calibration profile name', 'profile_name', selected_defaults.get('profile_name', 'uncalibrated'), disabled=paired)
            config['controller_file'] = generation_path_input('Optional frozen OT controller.pt on the server', 'controller_file', selected_defaults.get('controller_file', ''),
                help='Requires a canonical scene NPZ and the exact own checkpoint used to train this controller.', disabled=paired)
            if paired:
                config.update(profile_file='', profile_name='uncalibrated', controller_file='')
            if latest.get('training_job_id') and not config['controller_file'] and not paired:
                st.caption('This training run produced no frozen OT controller; the optional controller field is empty.')
            camera_options = setting('camera_overlay', True)
            if paired:
                st.session_state[camera_options['key']] = True
            config['camera_overlay'] = st.checkbox('Overlay synthetic LiDAR on recorded camera images', **camera_options, disabled=paired,
                help='Match the conditioning trace to timestamped camera frames, export PNG comparisons, and display them in results. Unavailable inputs are reported without stopping LiDAR generation.')
            with st.expander('Camera overlay settings'):
                config['camera_calibration'] = generation_path_input('Optional camera calibration JSON on the server', 'camera_calibration',
                    selected_defaults.get('camera_calibration', ''),
                    help='Leave blank for the bundled Pandar128 mounting parameters. An explicit optical camera_from_sensor calibration can override mounting/distortion assumptions.')
                config['camera_model'] = st.selectbox('Camera projection model', ['rational', 'pinhole'], **setting('camera_model', 'rational'),
                    help='Rational applies radial/tangential distortion. Pinhole is for rectified images with matching rectified intrinsics.')
                config['camera_timestamp_unit'] = st.selectbox('Source camera/odometry timestamp unit', ['s', 'ms', 'us', 'ns'],
                    **setting('camera_timestamp_unit', 's'),
                    help='Applies to generic t fields. Explicit timestamp_s/ms/us/ns fields use their own units automatically. For Pandar128 odometry t and video timestamp_us, leave this at s.')
                config['camera_max_skew_ms'] = st.number_input('Maximum camera time difference (ms)', 1., 1000.,
                    **setting('camera_max_skew_ms', 50.))
        else:
            left, right = st.columns(2)
            config['epochs'] = left.number_input('Epochs', 1, 10000, 5)
            config['window'] = right.number_input('Frames per temporal window', 1, 5, 2)
            config['rl_enabled'] = st.checkbox('Run RL calibration and optimizer comparisons' if action == 'train' else 'Enable experimental RL controllers', value=True)
            if action == 'train':
                config['methods'] = st.multiselect('Calibration methods', CALIBRATION_METHODS,
                                                   default=['ppo', 'random_search', 'bayesian'])
            config['budget'] = st.number_input('Calibration calls per method' if action == 'train' else 'Generation-call budget per controller', 6, 5000, 30)
            with st.expander('Training settings'):
                config['channels'] = st.selectbox('Base model channels', [8, 16, 32, 64, 128], index=2)
                config['diffusion_steps'] = st.number_input('Training diffusion steps', 10, 2000, 1000)
                config['gradient_accumulation'] = st.number_input('Gradient accumulation', 1, 64, 1)
                config['workers'] = st.number_input('DataLoader workers per GPU (-1 = auto)', -1, 16, -1)
                config['lr'] = st.number_input('Learning rate', value=.0001, min_value=.000001, max_value=.01, format='%.6f')
            with st.expander('Additional RL method settings'):
                st.caption('Replay methods learn from two transitions and use batches capped by available replay. These settings apply to the newly added methods; SAC keeps its existing experiment settings.')
                config['rl'] = {
                    'warmup': st.number_input('RL random-action warmup', 0, 1000, 4),
                    'batch_size': st.number_input('RL replay batch limit', 2, 256, 16),
                    'rollout_steps': st.number_input('Actor-critic rollout steps', 1, 256, 5),
                    'n_steps': st.number_input('DQN return steps', 1, 64, 3),
                    'value_min': st.number_input('C51 minimum return support', -10000., 10000., -20.),
                    'value_max': st.number_input('C51 maximum return support', -10000., 10000., 20.),
                }
            st.caption('The selected batch per GPU and total batch per update appear in the processing logs and results.')
        config['seed'] = st.number_input('Seed', 0, 2147483647, **setting('seed', 0))
        if action == 'experiment':
            st.caption('Adds E0–E5 experiments. All choices freeze on development data before final real test arrays are decoded.')
            config['checkpoint'] = st.text_input('Optional own checkpoint for adaptation (random-root provenance required)')
            config['experiment'] = {
                'ablations': st.multiselect('Experiment arms', ['E0', 'E1', 'E2', 'E3', 'E4', 'E5'],
                                            default=['E0', 'E1', 'E2', 'E3', 'E4', 'E5']),
                'seeds': json_setting('Experiment seeds (JSON)', [0, 1, 2]),
                'controllers': st.multiselect('Frozen-generator controllers',
                    CONTROLLER_METHODS,
                    default=['fixed', 'random_search', 'bayesian', 'cma_es', 'sac']),
                'synthetic_scenes': st.number_input('Procedural canonical scenes per split', 1, 100, 4),
                'controller_seconds': st.number_input('Controller time budget per method (seconds)', 1., 86400., 300.),
                'controller_max_samples': st.number_input('Maximum generated samples per controller action', 1, 16, 2),
                'downstream': st.checkbox('Run labeled point-segmentation transfer probe', value=False),
                'label_source_confirmed': st.checkbox('I confirmed the native semantic label source and class meanings', value=False),
            }
            config['ot'] = {
                'epsilon': st.number_input('Sinkhorn entropy regularization', min_value=.001, value=.1),
                'iterations': st.number_input('Sinkhorn iterations', 5, 500, 60),
                'min_samples': st.number_input('Minimum returns per conditional bin', 2, 64, 4),
                'max_points': st.number_input('Maximum transport points per conditional bin', 4, 512, 64),
            }
            with st.expander('Transport units and external simulator scenes'):
                st.caption('Default features: range, intensity, elevation, local return rate and sensor-frame persistence. Scales freeze before fitting.')
                config['ot']['range_edges_m'] = json_setting('Fixed range-bin edges (JSON, metres)', [0, 30, 60, 120])
                config['ot']['feature_scales'] = json_setting('Physical feature scales (JSON)', [120, 1, 1.5707963268, 1, 1])
                config['ot']['feature_weights'] = json_setting('Physical feature weights (JSON)', [1, 1, .25, .5, .5])
                config['experiment']['label_budgets'] = json_setting('Real label budgets (JSON)', [512, 2048])
                for split, key in [('Training', 'scene_files'), ('Development', 'development_scene_files'), ('Test', 'test_scene_files')]:
                    config['experiment'][key] = [line.strip() for line in st.text_area(f'{split} canonical NPZ paths on server (one per line)').splitlines() if line.strip()]
    missing_inputs = []
    if action != 'prepare' and not config['manifest']:
        missing_inputs.append('Prepared manifest path on the server')
    if action == 'generate' and not config['checkpoint']:
        missing_inputs.append('Own checkpoint path on the server')
    if action == 'generate' and config.get('compare_training') and not (config['conditioning'] or config['auto_conditioning']):
        missing_inputs.append('a conditioning NPZ or automatic real-scene extraction')
    if missing_inputs:
        message = 'Start remote job is disabled: enter ' + ' and '.join(missing_inputs) + '.'
        if action in ('train', 'experiment') and not config['manifest']:
            message += (' Click Use latest prepared dataset to detect existing server data. '
                        'If none is available, complete Prepare Pandar128 first.')
        if action == 'generate' and not config['checkpoint']:
            message += (' Generation requires a trained checkpoint matching the selected manifest. '
                        'Automatic real-scene extraction supplies geometry; it does not train the model.')
            if config['manifest']:
                message += ' ' + (lookup_error or detected.get('diagnostic') or
                    'No checkpoint was returned by the server. Click Use detected paths, or enter the full checkpoint.pt path.')
        st.info(message)
    if st.button('Start remote job', type='primary', disabled=bool(missing_inputs)):
        try:
            state = api('POST', '/jobs', json=config).json()
            st.session_state['job_id'] = state['id']
            st.success(f"Started {state['id'][:12]}. Open Monitor and results.")
            LOGGER.info('Submitted remote %s job %s', action, state['id'])
        except httpx.HTTPError as error:
            st.error(error.response.text if isinstance(error, httpx.HTTPStatusError) else str(error))
            LOGGER.error('Job submission failed: %s', error)


@st.fragment(run_every=2)
def gpu_monitor():
    st.subheader('Remote GPU usage')
    try:
        snapshot = api('GET', '/metrics/gpu').json()
        if snapshot['error']:
            st.info(f"GPU monitoring unavailable: {snapshot['error']}")
        if not snapshot['gpus']:
            if not snapshot['error']:
                st.info('No NVIDIA GPUs were detected on the server.')
            return
        cache = f'gpu_history:{endpoint}'
        rows = [{'time': snapshot['timestamp'], 'gpu': f"GPU {gpu['id']}",
                 'utilization_percent': gpu['utilization_percent'], 'memory_used_gb': gpu['memory_used_gb']}
                for gpu in snapshot['gpus']]
        history = [row for row in st.session_state.get(cache, []) if row['time'] > snapshot['timestamp'] - 600]
        history = (history + rows)[-300 * len(rows):]
        st.session_state[cache] = history
        frame = pd.DataFrame(history)
        frame['time'] = pd.to_datetime(frame['time'], unit='s', utc=True)
        utilization, memory = st.columns(2)
        utilization.caption('GPU utilization (%)')
        utilization.line_chart(frame, x='time', y='utilization_percent', color='gpu')
        memory.caption('GPU memory used (GiB)')
        memory.line_chart(frame, x='time', y='memory_used_gb', color='gpu')
        st.dataframe(pd.DataFrame(snapshot['gpus']), hide_index=True)
    except httpx.HTTPError as error:
        st.info(f'GPU monitoring connection interrupted: {error}')


@st.fragment(run_every=2)
def monitor():
    try:
        jobs = api('GET', '/jobs').json()
        if not jobs:
            st.info('No remote jobs yet.')
            return
        identifiers = [job['id'] for job in jobs]
        preferred = st.session_state.get('job_id')
        job_id = st.selectbox('Remote job', identifiers,
                              index=identifiers.index(preferred) if preferred in identifiers else 0,
                              format_func=lambda key: next(f"{j['id'][:12]} · {j['action']} · {j['status']}" for j in jobs if j['id'] == key))
        state = api('GET', f'/jobs/{job_id}').json()
        st.write(f"Status: **{state['status']}**")
        if state.get('resources'):
            resource = state['resources']
            st.caption(f"Training: {resource['world_size']} processes · batch {resource['batch_size_per_process']} per process · effective batch {resource['effective_batch_size']}")
        progress = state.get('progress', {})
        st.progress(progress.get('completed', 0) / max(1, progress.get('total', 1)), text=progress.get('stage', 'starting'))
        if state['status'] == 'running':
            if st.button('Cancel job'):
                api('POST', f'/jobs/{job_id}/cancel')
                st.rerun(scope='fragment')
            if state.get('training'):
                training_charts(state['training'])
        cursor = st.session_state.setdefault(f'log:{endpoint}:{job_id}', {})
        log = sync_remote_log(api, job_id, cursor)
        st.subheader('Live processing log')
        with st.container(height=320):
            st.code(log or 'Waiting for worker output', language='log')
        if state['status'] != 'completed':
            return
        cache = f'{endpoint}:{job_id}'
        if cache not in st.session_state:
            st.session_state[cache] = api('GET', f'/jobs/{job_id}/result', timeout=60).json()
        report = st.session_state[cache]
        if report['action'] == 'generate':
            if st.button('Build / refresh pixel and OT graphs', key=f'pixel_rebuild:{job_id}',
                         help='Analyze the saved NPZs and camera overlays, without retraining or generation.'):
                with st.spinner('Building pixel graphs from saved data on the server...'):
                    report = api('POST', f'/jobs/{job_id}/pixel-graphs/rebuild', timeout=180).json()
                st.session_state[cache] = report
                for key in list(st.session_state):
                    if key.startswith(f'pixel:{cache}:') or key in (f'pixel_zip:{cache}', f'export:{cache}', f'pdf:{cache}'):
                        del st.session_state[key]
        if report['action'] == 'prepare':
            st.write('Prepared manifest on the remote server:')
            st.code(report['manifest'])
            st.dataframe(report['inventory'])
        elif report['action'] == 'generate':
            st.success(f"Generated {report['samples']} LiDAR windows.")
            comparison = report.get('comparison')
            pixels = report.get('pixel_analysis', {})
            st.subheader('Pixel and optimal-transport domain-gap graphs')
            if pixels.get('frames'):
                st.caption('LiDAR range-grid pixels and camera pixels are separate measurements. Blank map pixels have no valid support; metrics use full resolution.')
                pixel_row = st.selectbox('Pixel graph frame', pixels['frames'],
                    format_func=lambda row: f"Frame {row['frame']}", key=f'pixel_frame:{job_id}')
                if pixel_row.get('transport'):
                    st.caption('OT graphs compare distributions: W1/W2, sliced W2, debiased Sinkhorn, GW, fused GW and unbalanced OT. Multivariate supports are sampled; scalar Wasserstein metrics use all values. Units and solver diagnostics are recorded in the export.')
                    with st.expander('Optimal-transport scores and solver details'):
                        st.dataframe([{'space': space, 'pair': pair, 'measure': metric, 'value': value,
                            'unit': record['units'][metric]}
                            for space, data in pixel_row['transport']['spaces'].items()
                            for pair, record in data['pairs'].items() for metric, value in record['values'].items()], hide_index=True)
                        st.json({'settings': pixels['transport_settings'], 'spaces': pixel_row['transport']['spaces']})
                errors = pd.DataFrame([{'pair': name, 'LiDAR radial range MAE (m)': values['range_pixel']['depth_mae_m'],
                    'Camera optical-depth MAE (m)': (values['camera_pixel'] or {}).get('depth_mae_m')}
                    for name, values in pixel_row['pairs'].items()]).set_index('pair')
                occupancy = pd.DataFrame([{'pair': name, 'LiDAR range-grid IoU': values['range_pixel']['occupancy_iou'],
                    'Camera pixel IoU': (values['camera_pixel'] or {}).get('occupancy_iou')}
                    for name, values in pixel_row['pairs'].items()]).set_index('pair')
                left, right = st.columns(2)
                left.caption('Pixel depth errors (metres; lower is better)')
                if errors.dropna(axis=1, how='all').empty:
                    left.info('No shared valid pixels for depth errors.')
                else:
                    left.bar_chart(errors.dropna(axis=1, how='all'))
                right.caption('Pixel occupancy IoU (higher is better)')
                if occupancy.dropna(axis=1, how='all').empty:
                    right.info('No occupied pixels for an occupancy comparison.')
                else:
                    right.bar_chart(occupancy.dropna(axis=1, how='all'))
                if pixel_row.get('camera_reason'):
                    st.info('Camera pixel graphs unavailable: ' + pixel_row['camera_reason'] + '. LiDAR range-grid graphs are shown below.')
                for graph in pixel_row['graphs']:
                    key = f'pixel:{cache}:{graph["file"]}'
                    if key not in st.session_state:
                        st.session_state[key] = api('GET', f'/jobs/{job_id}/pixel-graphs/{Path(graph["file"]).name}', timeout=60).content
                    st.image(st.session_state[key], caption=graph['label'])
                st.link_button('Open interactive pixel graphs', f'{endpoint}/jobs/{job_id}/pixel-graphs')
                if st.button('Prepare pixel graphs download', key=f'pixel_download:{job_id}'):
                    st.session_state[f'pixel_zip:{cache}'] = api('GET', f'/jobs/{job_id}/pixel-graphs.zip', timeout=180).content
                if f'pixel_zip:{cache}' in st.session_state:
                    st.download_button('Download pixel PNGs, HTML, CSV and maps', st.session_state[f'pixel_zip:{cache}'],
                        'pixel_graphs.zip', 'application/zip', key=f'pixel_zip_button:{job_id}')
            else:
                st.info(pixels.get('reason', 'This older run has no pixel graphs yet. Click Build / refresh pixel graphs above; no retraining is needed.'))
            if comparison:
                st.subheader('Without training / with training')
                st.caption('Same conditioning scene and sampling seed. Overlays share a recorded RGB photo. Geometry is constrained by the conditioning rules.')
                st.link_button('Open side-by-side images, point clouds and gap graphs', f'{endpoint}/jobs/{job_id}/comparison')
                st.dataframe([{'frame': row['frame'], 'comparison': name,
                    'pixel_depth_MAE_m': (values['pixel'] or {}).get('depth_mae_m'),
                    'voxel_IoU': values['voxel']['occupancy_iou'],
                    'Chamfer_m': values['voxel']['chamfer_mean_m'],
                    'voxel_JS_distance': values['voxel']['js_distance']}
                    for row in comparison['frames'] for name, values in row['pairs'].items()], hide_index=True)
                if comparison.get('camera_diagnostic'):
                    st.info('Camera comparison unavailable: ' + comparison['camera_diagnostic'])
            if report.get('real_scene'):
                st.write('Real conditioning scene on the remote server:')
                st.code(report['real_scene'])
                st.caption('Download artifacts to get real_scene.npz and sample_*.npz for comparison in the same viewer.')
            overlays = report.get('camera_overlays', {})
            if overlays.get('status') == 'completed':
                st.subheader('Synthetic LiDAR on recorded camera images')
                st.caption(overlays['background'])
                frames = overlays['frames']
                selected = st.selectbox('Overlay frame', range(len(frames)), key=f'overlay_frame:{job_id}')
                frame = frames[selected]
                samples = frame['synthetic']
                if comparison:
                    shown = samples
                    labels = ['Without training', 'With training']
                else:
                    chosen = st.selectbox('Synthetic overlay sample', range(len(samples)), key=f'overlay_sample:{job_id}',
                        format_func=lambda index: samples[index]['sample'])
                    shown, labels = (frame['real'], samples[chosen]), ('Real LiDAR', 'Synthetic LiDAR')
                for panel, record, label in zip(st.columns(2), shown, labels):
                    key = f'overlay:{cache}:{record["file"]}'
                    if key not in st.session_state:
                        st.session_state[key] = api('GET', f'/jobs/{job_id}/overlays/{record["file"]}', timeout=60).content
                    panel.image(st.session_state[key], caption=f'{label} on the same recorded RGB frame')
                st.caption(f"LiDAR source frame {frame['lidar_frame']} · camera frame {frame['camera_frame']} · time difference {frame['skew_ms']:.3f} ms. Depth colours: blue 0 m, green 40 m, red 80 m and beyond.")
                with st.expander('Camera alignment details'):
                    st.write(overlays['calibration_convention'])
                    st.write(overlays['limitations'])
            elif overlays.get('status') == 'unavailable':
                st.info('Camera overlay unavailable: ' + overlays['reason'])
        elif report['action'] == 'experiment':
            st.write(f"Selected on development data: {report['selected_arm']}")
            st.dataframe([{'arm': name, 'test_OT_gap': value['mean'], 'session_CI95': str(value['ci95'])}
                          for name, value in report['session_intervals'].items()], hide_index=True)
            st.json(report['gates'])
            with st.expander('OT components, physical checks, task budgets and compute'):
                st.json(report['development'])
                st.json(report['evaluations'])
                st.json(report['data_efficiency'])
                st.json(report['compute'])
                st.write(report['limitations'])
        else:
            training_charts(report['training'])
            st.dataframe([{'method': name, 'held_out_loss': value['loss']['mean'], 'CI95': str(value['loss']['ci95'])}
                          for name, value in report['evaluations'].items()], hide_index=True)
            st.caption(f"Export profile selected on validation: {report['selected_profile_on_validation']}")
            curves = {name: pd.Series([row['loss'] for row in value['history']], index=range(1, len(value['history']) + 1))
                      for name, value in report['profiles'].items() if 'history' in value}
            if curves:
                st.line_chart(pd.DataFrame(curves), x_label='Calibration calls', y_label='Discrepancy')
            left, right = st.columns(2)
            left.image(np.asarray(report['preview']['range']) / 200., clamp=True, caption='Generated range image')
            points = pd.DataFrame(report['preview']['points'], columns=['x', 'y', 'z'])
            right.scatter_chart(points, x='x', y='y')
            with st.expander('Per-trace metrics, profiles and scope'):
                st.json(report['evaluations'])
                st.json(report['profiles'])
                st.write(report['limitations'])
        pdf_key = f'pdf:{cache}'
        if st.button('Prepare PDF report'):
            st.session_state[pdf_key] = api('GET', f'/jobs/{job_id}/report', timeout=60).content
        if pdf_key in st.session_state:
            st.download_button('Download PDF report', st.session_state[pdf_key], 'result.pdf', 'application/pdf')
        st.download_button('Download JSON report', json.dumps(report, indent=2), 'result.json', 'application/json')
        if report['action'] != 'prepare' and st.button('Prepare artifact download'):
            st.session_state[f'export:{cache}'] = api('GET', f'/jobs/{job_id}/artifacts', timeout=120).content
        if f'export:{cache}' in st.session_state:
            st.download_button('Download checkpoint, profiles and samples', st.session_state[f'export:{cache}'], 'lidarpoc.zip', 'application/zip')
    except httpx.HTTPError as error:
        st.error(f'Connection interrupted: {error}. The remote job continues independently.')


with results:
    gpu_monitor()
    monitor()
