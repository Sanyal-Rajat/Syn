import io
import json
import os
from pathlib import Path
import time
import zipfile

from fastapi.testclient import TestClient
import server


def test_api_resolves_all_gpus_and_rejects_unavailable_counts(prepared, tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    import torch

    monkeypatch.setattr(server, 'ROOT', tmp_path / 'runs')
    monkeypatch.setattr(server, 'PROCESSES', {})
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)
    process = MagicMock()
    process.poll.return_value = 0
    monkeypatch.setattr(server.subprocess, 'Popen', lambda *args, **kwargs: process)
    monkeypatch.setattr(server, 'Thread', MagicMock())
    with TestClient(server.app) as client:
        response = client.post('/jobs', json={'manifest': prepared})
        assert response.status_code == 202, response.text
        path = server.ROOT / response.json()['id'] / 'request.json'
        assert json.loads(path.read_text())['gpus'] == 4
        response = client.post('/jobs', json={'manifest': prepared, 'gpus': 5})
        assert response.status_code == 422 and '4 are visible' in response.text


def test_remote_training_and_artifacts(prepared, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(server, 'ROOT', tmp_path / 'runs')
    with TestClient(server.app) as client:
        assert client.get('/health').status_code == 200
        assert client.get('/jobs/no-such-job').status_code == 404
        config = {'manifest': prepared, 'gpus': 0, 'channels': 8, 'epochs': 1, 'height': 4, 'width': 16,
                  'window': 2, 'batch_size': 2, 'diffusion_steps': 10, 'sample_steps': 2,
                  'samples_per_trace': 1, 'budget': 6, 'methods': ['ppo', 'random_search']}
        response = client.post('/jobs', json=config)
        assert response.status_code == 202, response.text
        identifier = response.json()['id']
        assert client.post('/jobs', json=config).status_code == 409
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            state = client.get(f'/jobs/{identifier}').json()
            if state['status'] != 'running':
                break
            time.sleep(.1)
        assert state['status'] == 'completed', state
        report = client.get(f'/jobs/{identifier}/result').json()
        assert report['pretrained_weights_used'] is False
        assert 'ppo' in report['profiles']
        output = client.get(f'/jobs/{identifier}/logs').json()
        assert 'Epoch metrics:' in output['text']
        assert 'Starting ppo calibration' in output['text']
        assert 'Epoch metrics:' in capsys.readouterr().out
        archive = zipfile.ZipFile(io.BytesIO(client.get(f'/jobs/{identifier}/artifacts').content))
        assert {'checkpoint.pt', 'ppo_policy.pt', 'profiles.json', 'sample_000.npz'} <= set(archive.namelist())
        assert 'LICENSES/MODEL-MIT.txt' in archive.namelist()
        generation = client.post('/jobs', json={'action': 'generate', 'manifest': prepared,
            'checkpoint': report['checkpoint'], 'gpus': 0, 'sample_steps': 2, 'samples_per_trace': 1,
            'profile_file': str(server.ROOT / identifier / 'profiles.json'), 'profile_name': 'ppo'})
        assert generation.status_code == 202, generation.text
        generation_id = generation.json()['id']
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            state = client.get(f'/jobs/{generation_id}').json()
            if state['status'] != 'running':
                break
            time.sleep(.1)
        assert state['status'] == 'completed', state
        assert client.get(f'/jobs/{generation_id}/result').json()['calibration_profile'] == 'ppo'


def test_cancellation_and_rejected_inputs(prepared, tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'ROOT', tmp_path / 'runs')
    with TestClient(server.app) as client:
        assert client.post('/jobs', json={'manifest': ''}).status_code == 422
        assert client.post('/jobs', json={'manifest': prepared, 'checkpoint': '/foreign.pt'}).status_code == 422
        response = client.post('/jobs', json={'manifest': prepared, 'gpus': 0})
        identifier = response.json()['id']
        process = server.PROCESSES[identifier]
        assert client.post(f'/jobs/{identifier}/cancel').status_code == 200
        process.wait(timeout=10)
        assert client.get(f'/jobs/{identifier}').json()['status'] == 'cancelled'


def test_failed_worker_traceback_is_available_in_logs_and_server_terminal(prepared, tmp_path, monkeypatch, capsys):
    from pathlib import Path
    monkeypatch.setattr(server, 'ROOT', tmp_path / 'runs')
    with TestClient(server.app) as client:
        response = client.post('/jobs', json={'action': 'prepare',
            'dataset_root': str(Path(prepared).parents[1] / 'raw'), 'input_frame': 'sensor',
            'elevations_file': str(tmp_path / 'missing-elevations.txt'), 'height': 4, 'width': 16})
        assert response.status_code == 202, response.text
        identifier = response.json()['id']
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            state = client.get(f'/jobs/{identifier}').json()
            if state['status'] != 'running':
                break
            time.sleep(.1)
        assert state['status'] == 'failed', state
        log = client.get(f'/jobs/{identifier}/logs').json()['text']
        assert 'Traceback' in log and 'FileNotFoundError' in log
        assert 'FileNotFoundError' in capsys.readouterr().out


def test_optional_ot_experiment_api_and_artifacts(prepared, tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'ROOT', tmp_path / 'runs')
    with TestClient(server.app) as client:
        response = client.post('/jobs', json={'action': 'experiment', 'manifest': prepared, 'gpus': 0,
            'channels': 8, 'height': 4, 'width': 16, 'window': 2, 'diffusion_steps': 10, 'sample_steps': 2,
            'samples_per_trace': 1, 'ot': {'max_points': 8, 'min_samples': 2, 'iterations': 10},
            'experiment': {'ablations': ['E0'], 'seeds': [0], 'synthetic_scenes': 1}})
        assert response.status_code == 202, response.text
        identifier = response.json()['id']
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            state = client.get(f'/jobs/{identifier}').json()
            if state['status'] != 'running':
                break
            time.sleep(.1)
        assert state['status'] == 'completed', state
        result = client.get(f'/jobs/{identifier}/result').json()
        assert result['selected_arm'] == 'E0'
        assert result['gates']['task_transfer'] == 'not_evaluated'
        archive = zipfile.ZipFile(io.BytesIO(client.get(f'/jobs/{identifier}/artifacts').content))
        assert {'experiment_plan.json', 'selection.json', 'ot_reference.npz',
                'experiments/seed_0/E0/checkpoint.pt', 'scenes/train/000.npz'} <= set(archive.namelist())


def test_discovery_lists_only_completed_existing_preparations_after_restart(prepared, tmp_path, monkeypatch):
    runs = tmp_path / 'runs'
    monkeypatch.setattr(server, 'ROOT', runs)

    def record(index, action='prepare', status='completed', manifest=prepared):
        identifier = f'{index:032x}'
        folder = runs / identifier
        folder.mkdir(parents=True)
        (folder / 'state.json').write_text(json.dumps({'id': identifier, 'action': action, 'status': status}))
        (folder / 'result.json').write_text(json.dumps({'action': action, 'manifest': str(manifest)}))
        os.utime(folder / 'state.json', (index + 1, index + 1))
        return folder

    record(0)
    second = tmp_path / 'second-manifest.json'
    second.write_text(Path(prepared).read_text())
    latest = record(1, manifest=second)
    record(2, manifest=tmp_path / 'missing.json')
    record(3, status='failed')
    record(4, action='train')
    record(5, status='running')
    broken = record(6)
    (broken / 'state.json').write_text('{')
    record(7, manifest=tmp_path / 'corrupt.json')
    (tmp_path / 'corrupt.json').write_text('{}')
    for index in range(8, 60):
        record(index, status='failed')
    # Discovery must not be limited to the latest 50 jobs or process memory.
    # Remove the deliberately corrupt state before lifespan's existing recovery pass.
    (broken / 'state.json').write_text(json.dumps({'id': broken.name, 'action': 'prepare', 'status': 'failed'}))
    with TestClient(server.app) as client:
        (broken / 'state.json').write_text('{')
        rows = client.get('/dataset/prepared').json()
        assert [row['manifest'] for row in rows] == [str(second), str(Path(prepared).resolve())]
        assert rows[0]['job_id'] == latest.name and rows[0]['traces'] == 6
        assert rows[0]['height'] == 4 and rows[0]['width'] == 16
        # A missing manifest remains a server-side validation error, with no worker started.
        assert client.post('/jobs', json={'action': 'experiment', 'manifest': ''}).status_code == 422
    (broken / 'state.json').write_text(json.dumps({'id': broken.name, 'action': 'prepare', 'status': 'failed'}))
    with TestClient(server.app) as client:
        assert client.get('/dataset/prepared').json() == rows
