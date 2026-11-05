import io
import json
import os
from pathlib import Path
import time
import zipfile

import numpy as np

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


def test_remote_training_and_artifacts(prepared, camera_inputs, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(server, 'ROOT', tmp_path / 'runs')
    with TestClient(server.app) as client:
        assert client.get('/health').status_code == 200
        assert client.get('/jobs/no-such-job').status_code == 404
        missing = client.post('/jobs', json={'action': 'generate', 'manifest': prepared,
            'checkpoint': str(tmp_path), 'gpus': 0})
        assert missing.status_code == 422 and 'checkpoint.pt' in missing.text
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
        detected = client.post('/generation/inputs', json={'manifest': prepared})
        assert detected.status_code == 200, detected.text
        detected = detected.json()
        assert detected['checkpoint'] == report['checkpoint']
        assert Path(detected['conditioning']).is_file()
        assert detected['checkpoints'][0]['job_id'] == identifier
        latest = client.post('/generation/inputs', json={})
        assert latest.status_code == 200, latest.text
        latest = latest.json()
        assert latest['training_job_id'] == identifier
        assert latest['defaults']['manifest'] == prepared
        assert latest['defaults']['profile_name'] == report['selected_profile_on_validation']
        assert latest['defaults']['profile_file'] == str(server.ROOT / identifier / 'profiles.json')
        assert 'ppo' in report['profiles']
        output = client.get(f'/jobs/{identifier}/logs').json()
        assert 'Epoch metrics:' in output['text']
        assert 'Starting ppo calibration' in output['text']
        assert 'Epoch metrics:' in capsys.readouterr().out
        archive = zipfile.ZipFile(io.BytesIO(client.get(f'/jobs/{identifier}/artifacts').content))
        assert {'checkpoint.pt', 'ppo_policy.pt', 'profiles.json', 'sample_000.npz', 'result.pdf'} <= set(archive.namelist())
        assert (server.ROOT / identifier / 'result.pdf').read_bytes().startswith(b'%PDF-')
        pdf = client.get(f'/jobs/{identifier}/report')
        assert pdf.status_code == 200 and pdf.headers['content-type'] == 'application/pdf'
        assert pdf.content.startswith(b'%PDF-')
        assert 'LICENSES/MODEL-MIT.txt' in archive.namelist()
        generation = client.post('/jobs', json={'action': 'generate', **latest['defaults'],
            'camera_calibration': str(camera_inputs[1]),
            'checkpoint': str(Path(report['checkpoint']).parent)})
        assert generation.status_code == 202, generation.text
        generation_id = generation.json()['id']
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            state = client.get(f'/jobs/{generation_id}').json()
            if state['status'] != 'running':
                break
            time.sleep(.1)
        assert state['status'] == 'completed', state
        generated = client.get(f'/jobs/{generation_id}/result').json()
        assert generated['checkpoint'] == report['checkpoint']
        assert generated['calibration_profile'] == report['selected_profile_on_validation']
        assert generated['conditioning_source']['path'] == latest['defaults']['conditioning']
        assert generated['geometry']['new_surface_returns'] == 0
        overlays = generated['camera_overlays']
        assert overlays['status'] == 'completed', overlays
        assert [row['camera_frame'] for row in overlays['frames']] == [2, 3]
        image_name = overlays['frames'][0]['synthetic'][0]['file']
        preview = client.get(f'/jobs/{generation_id}/overlays/{image_name}')
        assert preview.status_code == 200 and preview.headers['content-type'] == 'image/png'
        assert preview.content.startswith(b'\x89PNG')
        assert client.get(f'/jobs/{generation_id}/overlays/request.json').status_code == 404
        secret = tmp_path / 'outside.png'
        secret.write_bytes(b'outside job')
        (server.ROOT / generation_id / 'overlays/outside.png').symlink_to(secret)
        assert client.get(f'/jobs/{generation_id}/overlays/outside.png').status_code == 404
        reference_path = generated['real_scene']
        assert Path(reference_path).name == 'real_scene.npz'
        with np.load(reference_path, allow_pickle=False) as real, np.load(server.ROOT / generation_id / 'sample_000.npz') as synthetic:
            assert real['range'].shape == synthetic['range'].shape == (2, 4, 16)
            valid = synthetic['valid']
            assert not valid[~real['valid']].any()
            assert (abs(synthetic['range'][valid] - real['range'][valid]) <= .10001).all()
        archive = zipfile.ZipFile(io.BytesIO(client.get(f'/jobs/{generation_id}/artifacts').content))
        assert {'real_scene.npz', 'sample_000.npz', 'result.json', 'result.pdf'} <= set(archive.namelist())
        assert f'overlays/{image_name}' in archive.namelist()
        assert 'overlays/calibration.json' in archive.namelist()
        assert 'overlays/outside.png' not in archive.namelist()
        assert client.get(f'/jobs/{generation_id}/report').content.startswith(b'%PDF-')
        for scene in ('', reference_path):
            response = client.post('/jobs', json={'action': 'generate', 'manifest': prepared,
                'checkpoint': report['checkpoint'], 'gpus': 0, 'sample_steps': 2, 'samples_per_trace': 1,
                'auto_conditioning': False, 'conditioning': scene})
            assert response.status_code == 202, response.text
            job = response.json()['id']
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                state = client.get(f'/jobs/{job}').json()
                if state['status'] != 'running':
                    break
                time.sleep(.1)
            assert state['status'] == 'completed', state
            result = client.get(f'/jobs/{job}/result').json()
            assert bool(result['real_scene']) == bool(scene)
            assert (server.ROOT / job / 'real_scene.npz').exists() == bool(scene)
            if scene:
                assert result['conditioning_source'] == {'kind': 'file', 'path': reference_path}
            else:
                assert result['conditioning'] is None and result['geometry'] is None


def test_cancellation_and_rejected_inputs(prepared, tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'ROOT', tmp_path / 'runs')
    with TestClient(server.app) as client:
        assert client.post('/jobs', json={'manifest': ''}).status_code == 422
        assert client.post('/jobs', json={'manifest': prepared, 'checkpoint': '/foreign.pt'}).status_code == 422
        response = client.post('/jobs', json={'manifest': prepared, 'gpus': 0})
        identifier = response.json()['id']
        process = server.PROCESSES[identifier]
        assert client.get(f'/jobs/{identifier}/report').status_code == 409
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
        assert client.get(f'/jobs/{identifier}/report').content.startswith(b'%PDF-')
        archive = zipfile.ZipFile(io.BytesIO(client.get(f'/jobs/{identifier}/artifacts').content))
        assert {'experiment_plan.json', 'selection.json', 'ot_reference.npz',
                'experiments/seed_0/E0/checkpoint.pt', 'scenes/train/000.npz', 'result.pdf'} <= set(archive.namelist())


def test_pdf_export_backfills_old_results_and_refreshes_changed_json(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'ROOT', tmp_path / 'runs')
    folder = server.ROOT / ('d' * 32)
    folder.mkdir(parents=True)
    (folder / 'state.json').write_text(json.dumps({'id': folder.name, 'action': 'generate', 'status': 'completed'}))
    source = folder / 'result.json'
    source.write_text('{"action":"generate","samples":1}')
    with TestClient(server.app) as client:
        assert client.get('/jobs/' + 'e' * 32 + '/report').status_code == 404
        url = f'/jobs/{folder.name}/report'
        first = client.get(url)
        assert first.status_code == 200 and first.content.startswith(b'%PDF-')
        target = folder / 'result.pdf'
        timestamp = target.stat().st_mtime_ns
        assert client.get(url).content == first.content
        assert target.stat().st_mtime_ns == timestamp
        source.write_text('{"action":"generate","samples":2}')
        os.utime(source, ns=(timestamp + 1_000_000, timestamp + 1_000_000))
        assert client.get(url).content != first.content
        archive = zipfile.ZipFile(io.BytesIO(client.get(f'/jobs/{folder.name}/artifacts').content))
        assert archive.read('result.pdf').startswith(b'%PDF-')


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
