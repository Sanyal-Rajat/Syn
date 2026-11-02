import io
import time
import zipfile

from fastapi.testclient import TestClient
import server


def test_remote_training_and_artifacts(prepared, tmp_path, monkeypatch):
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
