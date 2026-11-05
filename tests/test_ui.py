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
        elif path == '/dataset/prepared':
            result = []
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


@pytest.mark.parametrize('operation', ['Train diffusion + calibration', 'OT + diffusion + RL experiment'])
def test_gpu_server_defaults_to_all_gpus_and_reports_effective_batch(remote, monkeypatch, operation):
    original = httpx.request

    def request(method, url, **kwargs):
        if url.endswith('/health'):
            result = {'status': 'ok', 'hardware': {'gpus': [{'id': i} for i in range(4)]}}
            return httpx.Response(200, json=result, request=httpx.Request(method, url))
        return original(method, url, **kwargs)

    monkeypatch.setattr(httpx, 'request', request)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=20)
    app.radio[0].set_value(operation).run()
    next(field for field in app.text_input if field.label == 'Prepared manifest path on the server').set_value('/remote/manifest.json').run()
    assert next(field for field in app.number_input if field.label == 'Remote GPUs (-1 = all, 0 = CPU)').value == -1
    assert any('Maximum training batch: 2048 ' in caption.value for caption in app.caption)
    next(button for button in app.button if button.label == 'Start remote job').click().run()
    assert not app.exception
    assert remote[-1]['gpus'] == -1 and remote[-1]['workers'] == -1 and remote[-1]['rl_enabled']
    assert remote[-1]['auto_optimize']


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


def test_optional_ot_experiment_submits_typed_config_with_rl_preserved(remote):
    from lidarpoc.config import RunConfig
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=20)
    app.radio[0].set_value('OT + diffusion + RL experiment').run()
    next(field for field in app.text_input if field.label == 'Prepared manifest path on the server').set_value('/remote/manifest.json').run()
    next(button for button in app.button if button.label == 'Start remote job').click().run()
    assert not app.exception
    request = RunConfig(**remote[-1])
    assert request.action == 'experiment' and request.rl_enabled
    assert 'sac' in request.experiment.controllers
    assert request.experiment.seeds == [0, 1, 2]
    assert request.ot.feature_scales[0] == 120


@pytest.mark.parametrize('operation,label,field', [
    ('Train diffusion + calibration', 'Calibration methods', 'methods'),
    ('OT + diffusion + RL experiment', 'Frozen-generator controllers', 'controllers'),
])
def test_all_rl_methods_are_selectable_and_settings_reach_server(remote, operation, label, field):
    from lidarpoc.config import RunConfig
    from lidarpoc.rl import RL_METHODS

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=20)
    app.radio[0].set_value(operation).run()
    next(value for value in app.text_input if value.label == 'Prepared manifest path on the server').set_value('/remote/manifest.json').run()
    choices = next(value for value in app.multiselect if value.label == label)
    assert set(RL_METHODS) <= set(choices.options)
    choices.set_value(list(RL_METHODS)).run()
    next(value for value in app.number_input if value.label == 'RL replay batch limit').set_value(8).run()
    next(button for button in app.button if button.label == 'Start remote job').click().run()
    assert not app.exception
    config = RunConfig(**remote[-1])
    actual = config.methods if field == 'methods' else config.experiment.controllers
    assert actual == list(RL_METHODS) and config.rl.batch_size == 8


@pytest.fixture
def prepared_remote(remote, monkeypatch):
    original = httpx.request
    choices = [{'job_id': 'b' * 32, 'manifest': '/remote/new/prepared/manifest.json', 'height': 128, 'width': 512, 'traces': 12},
               {'job_id': 'c' * 32, 'manifest': '/remote/old/prepared/manifest.json', 'height': 128, 'width': 256, 'traces': 6}]

    def request(method, url, **kwargs):
        if url.endswith('/dataset/prepared'):
            return httpx.Response(200, json=choices, request=httpx.Request(method, url))
        return original(method, url, **kwargs)

    monkeypatch.setattr(httpx, 'request', request)
    return remote, choices


def test_empty_manifest_disables_submission_without_hiding_monitoring(remote):
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=20)
    app.radio[0].set_value('OT + diffusion + RL experiment').run()
    assert not app.exception
    assert next(button for button in app.button if button.label == 'Start remote job').disabled
    assert not remote
    assert any('Complete Prepare Pandar128' in message.value for message in app.info)
    assert any('Start remote job is disabled: enter Prepared manifest path on the server.' in message.value for message in app.info)
    assert len(app.get('vega_lite_chart')) + len(app.get('arrow_vega_lite_chart')) == 2


@pytest.mark.parametrize('checkpoint', ['', '   '])
def test_generate_requires_checkpoint_even_with_manifest_and_scene(remote, checkpoint):
    from lidarpoc.config import RunConfig

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=20)
    app.radio[0].set_value('Generate').run()
    next(field for field in app.text_input if field.label == 'Prepared manifest path on the server').set_value('/remote/prepared/manifest.json').run()
    next(field for field in app.text_input if field.label == 'Optional synthetic scene NPZ on the server').set_value('/remote/real_scene.npz').run()
    next(field for field in app.text_input if field.label == 'Own checkpoint path on the server').set_value(checkpoint).run()
    assert not app.exception
    assert next(button for button in app.button if button.label == 'Start remote job').disabled
    assert not remote
    assert any('Generation requires a trained checkpoint' in message.value for message in app.info)
    assert any(message.value.startswith('Start remote job is disabled: enter Own checkpoint path on the server.') for message in app.info)
    assert len(app.get('vega_lite_chart')) + len(app.get('arrow_vega_lite_chart')) == 2
    next(field for field in app.text_input if field.label == 'Own checkpoint path on the server').set_value(' /remote/train/checkpoint.pt ').run()
    assert not next(button for button in app.button if button.label == 'Start remote job').disabled
    next(button for button in app.button if button.label == 'Start remote job').click().run()
    assert not app.exception
    request = RunConfig(**remote[-1])
    assert request.action == 'generate' and request.checkpoint == '/remote/train/checkpoint.pt'
    assert request.conditioning == '/remote/real_scene.npz'
    assert request.auto_conditioning


@pytest.mark.parametrize('automatic', [True, False])
def test_generate_supports_automatic_real_scene_and_unconditional_opt_out(remote, automatic):
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=20)
    app.radio[0].set_value('Generate').run()
    next(field for field in app.text_input if field.label == 'Prepared manifest path on the server').set_value('/remote/manifest.json').run()
    next(field for field in app.text_input if field.label == 'Own checkpoint path on the server').set_value('/remote/checkpoint.pt').run()
    next(field for field in app.checkbox if field.label == 'Automatically extract a real scene from the prepared dataset').set_value(automatic).run()
    next(button for button in app.button if button.label == 'Start remote job').click().run()
    assert not app.exception
    assert remote[-1]['auto_conditioning'] is automatic
    assert remote[-1]['conditioning'] == ''


def test_generate_keeps_checkpoint_when_switching_operations(prepared_remote):
    submitted, _ = prepared_remote
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=20)
    app.radio[0].set_value('Generate').run()
    next(field for field in app.text_input if field.label == 'Own checkpoint path on the server').set_value('/remote/train/checkpoint.pt').run()
    assert not next(button for button in app.button if button.label == 'Start remote job').disabled
    app.radio[0].set_value('Train diffusion + calibration').run()
    app.radio[0].set_value('Prepare Pandar128').run()
    app.radio[0].set_value('Generate').run()
    assert next(field for field in app.text_input if field.label == 'Own checkpoint path on the server').value == '/remote/train/checkpoint.pt'
    assert not next(button for button in app.button if button.label == 'Start remote job').disabled
    next(button for button in app.button if button.label == 'Start remote job').click().run()
    assert not app.exception
    assert submitted[-1]['checkpoint'] == '/remote/train/checkpoint.pt'


def test_checkpoint_can_be_entered_before_manifest_and_explicitly_cleared(remote):
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=20)
    app.radio[0].set_value('Generate').run()
    assert any(message.value.startswith('Start remote job is disabled: enter Prepared manifest path on the server and Own checkpoint path on the server.') for message in app.info)
    next(field for field in app.text_input if field.label == 'Own checkpoint path on the server').set_value(' /remote/checkpoint.pt ').run()
    assert any(message.value == 'Start remote job is disabled: enter Prepared manifest path on the server.' for message in app.info)
    next(field for field in app.text_input if field.label == 'Prepared manifest path on the server').set_value('/remote/manifest.json').run()
    assert not next(button for button in app.button if button.label == 'Start remote job').disabled
    next(field for field in app.text_input if field.label == 'Own checkpoint path on the server').set_value('   ').run()
    app.radio[0].set_value('Prepare Pandar128').run()
    app.radio[0].set_value('Generate').run()
    assert not app.exception
    assert next(field for field in app.text_input if field.label == 'Own checkpoint path on the server').value == ''
    assert next(button for button in app.button if button.label == 'Start remote job').disabled
    assert not remote


def test_experiment_automatically_uses_latest_completed_preparation(prepared_remote):
    from lidarpoc.config import RunConfig
    submitted, choices = prepared_remote
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=20)
    app.radio[0].set_value('OT + diffusion + RL experiment').run()
    assert next(field for field in app.text_input if field.label == 'Prepared manifest path on the server').value == choices[0]['manifest']
    next(button for button in app.button if button.label == 'Start remote job').click().run()
    assert not app.exception
    request = RunConfig(**submitted[-1])
    assert request.manifest == choices[0]['manifest'] and request.action == 'experiment'
    assert request.rl_enabled and 'sac' in request.experiment.controllers


def test_chosen_and_custom_manifests_persist_without_silent_replacement(prepared_remote):
    submitted, choices = prepared_remote
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=20)
    app.radio[0].set_value('Train diffusion + calibration').run()
    next(field for field in app.selectbox if field.label == 'Completed preparation').set_value(choices[1]['manifest']).run()
    assert next(field for field in app.text_input if field.label == 'Prepared manifest path on the server').value == choices[1]['manifest']
    next(field for field in app.text_input if field.label == 'Prepared manifest path on the server').set_value('/remote/custom/manifest.json').run()
    app.radio[0].set_value('Prepare Pandar128').run()
    app.radio[0].set_value('OT + diffusion + RL experiment').run()
    assert next(field for field in app.text_input if field.label == 'Prepared manifest path on the server').value == '/remote/custom/manifest.json'
    next(button for button in app.button if button.label == 'Start remote job').click().run()
    assert submitted[-1]['manifest'] == '/remote/custom/manifest.json'
    next(field for field in app.text_input if field.label == 'Prepared manifest path on the server').set_value('   ').run()
    assert next(button for button in app.button if button.label == 'Start remote job').disabled
    next(field for field in app.selectbox if field.label == 'Completed preparation').set_value(choices[0]['manifest']).run()
    assert not app.exception
    assert next(field for field in app.text_input if field.label == 'Prepared manifest path on the server').value == choices[0]['manifest']


def test_completed_job_exposes_pdf_and_retains_json_download(remote, monkeypatch):
    original = httpx.request
    identifier, downloads = 'f' * 32, []
    completed = {'id': identifier, 'action': 'prepare', 'status': 'completed'}

    def request(method, url, **kwargs):
        path = url.removeprefix('http://127.0.0.2:9000')
        if path == '/jobs':
            result = [completed]
        elif path == f'/jobs/{identifier}':
            result = completed
        elif path == f'/jobs/{identifier}/result':
            result = {'action': 'prepare', 'manifest': '/remote/prepared/manifest.json', 'inventory': []}
        elif path == f'/jobs/{identifier}/report':
            downloads.append(path)
            return httpx.Response(200, content=b'%PDF-test', headers={'content-type': 'application/pdf'},
                                  request=httpx.Request(method, url))
        else:
            return original(method, url, **kwargs)
        return httpx.Response(200, json=result, request=httpx.Request(method, url))

    monkeypatch.setattr(httpx, 'request', request)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=20)
    assert not app.exception and not downloads
    next(button for button in app.button if button.label == 'Prepare PDF report').click().run()
    assert not app.exception and len(downloads) == 1
    buttons = {button.proto.label for button in app.get('download_button')}
    assert {'Download PDF report', 'Download JSON report'} <= buttons
    app.run()
    assert len(downloads) == 1
