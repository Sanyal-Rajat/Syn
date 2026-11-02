"""Thin Streamlit client: all data processing, generation and RL run remotely."""
import json
from pathlib import Path
import subprocess

import httpx
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Pandar128 generation', layout='wide')
st.title('Pandar128 LiDAR generation')
st.caption('LidarDM range-image adaptation · random-initialized diffusion · PPO calibration · remote compute over SSH')

with st.sidebar:
    st.subheader('Remote connection')
    local_port = st.number_input('Local tunnel port', 1024, 65535, 8000)
    endpoint = f'http://127.0.0.1:{local_port}'
    with st.expander('SSH tunnel'):
        host = st.text_input('SSH host or configured alias')
        user = st.text_input('SSH user (optional for an alias)')
        key = st.text_input('Private key path on this client')
        ssh_port = st.number_input('SSH port', 1, 65535, 22)
        remote_port = st.number_input('Remote API port', 1024, 65535, 8000)
        a, b, c = st.columns(3)
        command = 'start' if a.button('Connect') else 'status' if b.button('Status') else 'stop' if c.button('Close') else None
        if command:
            args = ['bash', str(Path(__file__).with_name('ssh_tunnel.sh')), command, '-l', str(local_port)]
            if command == 'start':
                args += ['-H', host, '-P', str(ssh_port), '-p', str(remote_port)]
                if user:
                    args += ['-u', user]
                if key:
                    args += ['-k', key]
            try:
                response = subprocess.run(args, capture_output=True, text=True, timeout=15)
                (st.success if response.returncode == 0 else st.error)(response.stdout + response.stderr)
            except (OSError, subprocess.TimeoutExpired) as error:
                st.error(str(error))


def api(method, path, **kwargs):
    response = httpx.request(method, endpoint + path, timeout=kwargs.pop('timeout', 10), **kwargs)
    response.raise_for_status()
    return response


try:
    health = api('GET', '/health').json()
except httpx.HTTPError:
    st.info('Start the remote API and connect the SSH tunnel to this client.')
    st.code('Remote: uvicorn server:app --host 127.0.0.1 --port 8000 --workers 1\n'
            'Client: bash ssh_tunnel.sh start -H YOUR_SERVER -u YOUR_USER\n'
            'Client: streamlit run app.py --server.address 127.0.0.1')
    st.stop()

st.sidebar.success('Remote API connected')
with st.sidebar.expander('Remote hardware'):
    st.json(health['hardware'])

configure, results = st.tabs(['Configure', 'Monitor and results'])
with configure:
    task = st.radio('Operation', ['Prepare Pandar128', 'Train diffusion + calibration', 'Generate'], horizontal=True)
    action = 'prepare' if task.startswith('Prepare') else 'train' if task.startswith('Train') else 'generate'
    config = {'action': action}
    if action == 'prepare':
        config['dataset_root'] = st.text_input('Extracted dataset directory on the server', placeholder='/data/pandar128')
        left, right = st.columns(2)
        config['input_frame'] = left.selectbox('Input coordinate frame', ['vehicle', 'sensor'],
                                               help='Confirm against the publisher scan schema. Vehicle points are transformed with Configuration.json.')
        beam_source = right.selectbox('Beam elevations', ['Measured calibration file', 'Estimate from training scans'])
        config['infer_beams'] = beam_source.startswith('Estimate')
        config['elevations_file'] = st.text_input('Server path to 128 elevations in degrees, sorted ascending',
                                                 disabled=config['infer_beams'])
        if config['infer_beams']:
            st.caption('Estimation uses training scans only and must pass a spacing/residual check. The resulting rows are elevation bins.')
        config['width'] = st.selectbox('Azimuth columns', [256, 512, 1024, 2048], index=1)
        config['intensity_max'] = st.number_input('Intensity normalization maximum', min_value=.001, value=255.)
        config['max_frames'] = st.number_input('Frames to prepare per trace', 2, 100000, 128)
        if st.button('Inspect NPZ schemas'):
            try:
                st.dataframe(api('POST', '/dataset/inspect', json={'root': config['dataset_root']}, timeout=120).json()['recordings'])
            except httpx.HTTPError as error:
                st.error(str(error))
    else:
        config['manifest'] = st.text_input('Prepared manifest path on the server', key='manifest', placeholder='/data/cache/manifest.json')
        left, middle, right = st.columns(3)
        config['gpus'] = left.number_input('Remote GPUs (0 = CPU)', 0, 16, min(1, len(health['hardware']['gpus'])))
        config['sample_steps'] = middle.number_input('Sampling steps', 2, 200, 50)
        config['samples_per_trace'] = right.number_input('Evaluation windows per trace / generated samples', 1, 100, 4)
        if action == 'generate':
            config['checkpoint'] = st.text_input('Own checkpoint path on the server')
            config['conditioning'] = st.text_input('Optional synthetic scene NPZ on the server',
                                                  help='range and valid arrays [T,H,W], matching the checkpoint sensor. Full-scene geometry is preserved.')
            config['profile_file'] = st.text_input('Optional calibration profiles.json on the server')
            config['profile_name'] = st.text_input('Calibration profile name', 'ppo')
        else:
            left, middle, right = st.columns(3)
            config['epochs'] = left.number_input('Epochs', 1, 10000, 5)
            config['batch_size'] = middle.number_input('Batch size per GPU', 1, 64, 2)
            config['window'] = right.number_input('Frames per temporal window', 1, 5, 2)
            config['rl_enabled'] = st.checkbox('Run PPO calibration and optimizer comparisons', value=True)
            config['methods'] = st.multiselect('Calibration methods', ['ppo', 'random_search', 'bayesian', 'cma_es'],
                                               default=['ppo', 'random_search', 'bayesian'])
            config['budget'] = st.number_input('Calibration calls per method', 6, 5000, 30)
            with st.expander('Training settings'):
                config['channels'] = st.selectbox('Base model channels', [8, 16, 32, 64, 128], index=2)
                config['diffusion_steps'] = st.number_input('Training diffusion steps', 10, 2000, 1000)
                config['gradient_accumulation'] = st.number_input('Gradient accumulation', 1, 64, 1)
                config['workers'] = st.number_input('DataLoader workers per GPU', 0, 16, 0)
                config['lr'] = st.number_input('Learning rate', value=.0001, min_value=.000001, max_value=.01, format='%.6f')
        config['seed'] = st.number_input('Seed', 0, 2147483647, 0)
    if st.button('Start remote job', type='primary'):
        try:
            state = api('POST', '/jobs', json=config).json()
            st.session_state['job_id'] = state['id']
            st.success(f"Started {state['id'][:12]}. Open Monitor and results.")
        except httpx.HTTPError as error:
            st.error(error.response.text if isinstance(error, httpx.HTTPStatusError) else str(error))


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
        progress = state.get('progress', {})
        st.progress(progress.get('completed', 0) / max(1, progress.get('total', 1)), text=progress.get('stage', 'starting'))
        if state['status'] == 'running':
            if st.button('Cancel job'):
                api('POST', f'/jobs/{job_id}/cancel')
                st.rerun(scope='fragment')
            if state.get('training'):
                st.line_chart(pd.DataFrame(state['training']).set_index('epoch'))
        with st.expander('Server log', expanded=state['status'] == 'failed'):
            st.code(state.get('log', 'Waiting for worker output'))
        if state['status'] != 'completed':
            return
        cache = f'{endpoint}:{job_id}'
        if cache not in st.session_state:
            st.session_state[cache] = api('GET', f'/jobs/{job_id}/result', timeout=60).json()
        report = st.session_state[cache]
        if report['action'] == 'prepare':
            st.write('Prepared manifest on the remote server:')
            st.code(report['manifest'])
            st.dataframe(report['inventory'])
        elif report['action'] == 'generate':
            st.success(f"Generated {report['samples']} LiDAR windows.")
        else:
            st.line_chart(pd.DataFrame(report['training']).set_index('epoch'))
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
        st.download_button('Download report', json.dumps(report, indent=2), 'result.json', 'application/json')
        if report['action'] != 'prepare' and st.button('Prepare artifact download'):
            st.session_state[f'export:{cache}'] = api('GET', f'/jobs/{job_id}/artifacts', timeout=120).content
        if f'export:{cache}' in st.session_state:
            st.download_button('Download checkpoint, profiles and samples', st.session_state[f'export:{cache}'], 'lidarpoc.zip', 'application/zip')
    except httpx.HTTPError as error:
        st.error(f'Connection interrupted: {error}. The remote job continues independently.')


with results:
    monitor()
