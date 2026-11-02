from pathlib import Path
import time
import httpx
import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def remote(monkeypatch):
    submitted = []
    jobs = []
    def request(method, url, **kwargs):
        assert url.startswith('http://127.0.0.2:9000')
        assert kwargs['trust_env'] is False
        path = url.removeprefix('http://127.0.0.2:9000')
        if path == '/health':
            result = {'status': 'ok', 'hardware': {'gpus': [], 'physical_cpus': 4, 'available_ram_gb': 8}}
        elif path == '/metrics/gpu':
            result = {'timestamp': time.time(), 'error': None, 'gpus': [
                {'id': 0, 'name': 'Test GPU', 'uuid': 'GPU-test', 'utilization_percent': 73,
                 'memory_used_gb': 4., 'memory_total_gb': 24., 'temperature_c': 50, 'power_w': 90}]}
        elif path == '/jobs' and method == 'POST':
            submitted.append(kwargs['json'])
            jobs.append({'id': 'a' * 32, 'action': kwargs['json']['action'], 'status': 'running'})
            result = jobs[-1]
        elif path == '/jobs':
            result = jobs
        elif path.endswith('/logs'):
            text = 'Preparing remote trace\n' if kwargs['params']['offset'] == 0 else ''
            result = {'text': text, 'offset': len('Preparing remote trace\n'), 'reset': False}
        else:
            result = {**jobs[-1], 'progress': {'completed': 0, 'total': 1, 'stage': 'training'}}
        return httpx.Response(200, json=result, request=httpx.Request(method, url))
    monkeypatch.setattr(httpx, 'request', request)
    return submitted


def test_thin_client_keeps_rl_and_submits_remote_config(remote):
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=20)
    assert not app.exception
    app.radio[0].set_value('Train diffusion + calibration').run()
    next(field for field in app.text_input if field.label == 'Prepared manifest path on the server').set_value('/remote/manifest.json').run()
    assert app.checkbox[0].value is True
    assert 'ppo' in app.multiselect[0].value
    next(button for button in app.button if button.label == 'Start remote job').click().run()
    assert not app.exception
    assert remote[-1]['rl_enabled'] is True
    assert remote[-1]['gpus'] == 0
    assert 'ppo' in remote[-1]['methods']


def test_prepare_automatically_downloads_without_a_dataset_path(remote):
    from lidarpoc.config import RunConfig
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=20)
    next(button for button in app.button if button.label == 'Start remote job').click().run()
    assert not app.exception
    request = RunConfig(**remote[-1])
    assert request.action == 'prepare'
    assert request.dataset_root == ''
    assert request.infer_beams is True


def test_gpu_graphs_and_incremental_client_logs(remote, capsys):
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=20)
    history = app.session_state['gpu_history:http://127.0.0.2:9000']
    assert history[-1]['utilization_percent'] == 73
    assert history[-1]['memory_used_gb'] == 4.
    assert len(app.get('vega_lite_chart')) + len(app.get('arrow_vega_lite_chart')) == 2
    next(button for button in app.button if button.label == 'Start remote job').click().run()
    assert not app.exception
    assert any('Preparing remote trace' in block.value for block in app.code)
    assert 'Preparing remote trace' in capsys.readouterr().out
    app.run()
    assert 'Preparing remote trace' not in capsys.readouterr().out


def test_connect_uses_an_existing_healthy_api_tunnel(remote, monkeypatch):
    import subprocess

    def unexpected_connection(*args, **kwargs):
        raise AssertionError('An active API tunnel must be reused without invoking SSH')

    monkeypatch.setattr(subprocess, 'run', unexpected_connection)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=20)
    next(button for button in app.button if button.label == 'Connect').click().run()
    assert not app.exception
    assert any('Using existing API tunnel' in message.value for message in app.success)
