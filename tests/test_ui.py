from pathlib import Path
import httpx
from streamlit.testing.v1 import AppTest


def test_thin_client_keeps_rl_and_submits_remote_config(monkeypatch):
    submitted = []
    jobs = []
    def request(method, url, **kwargs):
        path = url.removeprefix('http://127.0.0.1:8000')
        if path == '/health':
            result = {'hardware': {'gpus': [], 'physical_cpus': 4, 'available_ram_gb': 8}}
        elif path == '/jobs' and method == 'POST':
            submitted.append(kwargs['json'])
            jobs.append({'id': 'a' * 32, 'action': 'train', 'status': 'running'})
            result = jobs[-1]
        elif path == '/jobs':
            result = jobs
        else:
            result = {**jobs[-1], 'progress': {'completed': 0, 'total': 1, 'stage': 'training'}}
        return httpx.Response(200, json=result, request=httpx.Request(method, url))
    monkeypatch.setattr(httpx, 'request', request)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=20)
    assert not app.exception
    app.radio[0].set_value('Train diffusion + calibration').run()
    next(field for field in app.text_input if field.label == 'Prepared manifest path on the server').set_value('/remote/manifest.json').run()
    assert app.checkbox[0].value is True
    assert 'ppo' in app.multiselect[0].value
    next(button for button in app.button if button.label == 'Start remote job').click().run()
    assert not app.exception
    assert submitted[-1]['rl_enabled'] is True
    assert submitted[-1]['gpus'] == 0
    assert 'ppo' in submitted[-1]['methods']
