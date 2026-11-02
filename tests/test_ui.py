from pathlib import Path
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
            result = {'hardware': {'gpus': [], 'physical_cpus': 4, 'available_ram_gb': 8}}
        elif path == '/jobs' and method == 'POST':
            submitted.append(kwargs['json'])
            jobs.append({'id': 'a' * 32, 'action': kwargs['json']['action'], 'status': 'running'})
            result = jobs[-1]
        elif path == '/jobs':
            result = jobs
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
