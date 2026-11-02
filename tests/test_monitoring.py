import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from lidarpoc.logs import read_log, sync_remote_log
from lidarpoc.telemetry import gpu_usage
import server


def test_log_cursors_preserve_unicode_and_handle_rewritten_files(tmp_path, capsys):
    path = tmp_path / 'worker.log'
    path.write_text('abc€ first\n')
    first = read_log(path, limit=4)
    assert first == {'text': 'abc', 'offset': 3, 'reset': False}
    second = read_log(path, first['offset'])
    assert second['text'] == '€ first\n'
    assert read_log(path, second['offset'])['text'] == ''
    path.write_text('new\n')
    assert read_log(path, second['offset']) == {'text': 'new\n', 'offset': 4, 'reset': True}

    def api(method, route, **kwargs):
        return httpx.Response(200, json=read_log(path, kwargs['params']['offset']))

    cursor = {}
    assert sync_remote_log(api, 'a' * 32, cursor) == 'new\n'
    assert 'new' in capsys.readouterr().out
    assert sync_remote_log(api, 'a' * 32, cursor) == 'new\n'
    assert capsys.readouterr().out == ''


def test_api_log_endpoint_reads_only_new_output(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'ROOT', tmp_path)
    job = tmp_path / ('a' * 32)
    job.mkdir()
    (job / 'state.json').write_text(json.dumps({'id': job.name, 'status': 'failed'}))
    (job / 'worker.log').write_text('worker started\nTraceback: test error\n')
    with TestClient(server.app) as client:
        response = client.get(f'/jobs/{job.name}/logs')
        assert response.status_code == 200
        result = response.json()
        assert 'Traceback: test error' in result['text']
        assert client.get(f'/jobs/{job.name}/logs', params={'offset': result['offset']}).json()['text'] == ''
        assert client.get(f'/jobs/{job.name}/logs', params={'offset': -1}).status_code == 422
        assert client.get('/jobs/no-such-job/logs').status_code == 404


def test_nvml_gpu_metrics_and_unsupported_fields(monkeypatch):
    class NVMLError(Exception):
        pass

    shutdown = []

    def power(handle):
        if handle == 1:
            raise NVMLError('Not supported')
        return 125000

    nvml = SimpleNamespace(NVMLError=NVMLError, NVML_TEMPERATURE_GPU=0, nvmlInit=lambda: None,
        nvmlShutdown=lambda: shutdown.append(True), nvmlDeviceGetCount=lambda: 2,
        nvmlDeviceGetHandleByIndex=lambda index: index, nvmlDeviceGetUUID=lambda handle: f'GPU-{handle}',
        nvmlDeviceGetName=lambda handle: b'Test NVIDIA GPU',
        nvmlDeviceGetUtilizationRates=lambda handle: SimpleNamespace(gpu=70 + handle),
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(used=(4 + handle) * 1024**3, total=24 * 1024**3),
        nvmlDeviceGetPowerUsage=power, nvmlDeviceGetTemperature=lambda *_: 52)
    monkeypatch.setitem(sys.modules, 'pynvml', nvml)
    result = gpu_usage()
    assert result['error'] is None and len(result['gpus']) == 2
    assert result['gpus'][0]['utilization_percent'] == 70
    assert result['gpus'][0]['memory_used_gb'] == 4
    assert result['gpus'][0]['power_w'] == 125
    assert result['gpus'][1]['power_w'] is None
    assert len(shutdown) == 1
    with TestClient(server.app) as client:
        assert client.get('/metrics/gpu').json()['gpus'][1]['utilization_percent'] == 71


@pytest.mark.parametrize('missing', [True, False])
def test_gpu_monitoring_is_optional_on_cpu_or_driverless_servers(monkeypatch, missing):
    if missing:
        monkeypatch.setitem(sys.modules, 'pynvml', None)
    else:
        class NVMLError(Exception):
            pass

        def unavailable():
            raise NVMLError('No NVIDIA driver')

        monkeypatch.setitem(sys.modules, 'pynvml', SimpleNamespace(NVMLError=NVMLError, nvmlInit=unavailable))
    result = gpu_usage()
    assert not result['gpus'] and result['error']


def test_preparation_does_not_import_torch_or_optional_model_packages(prepared, tmp_path):
    raw = Path(prepared).parents[1] / 'raw'
    request = {'action': 'prepare', 'dataset_root': str(raw), 'input_frame': 'sensor',
               'elevations_file': str(tmp_path / 'beams.txt'), 'height': 4, 'width': 16, 'max_frames': 4}
    code = '''
import sys
from importlib.abc import MetaPathFinder
class NoModels(MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'diffusers', 'peft', 'transformers', 'bitsandbytes'}:
            raise ImportError('Model import prohibited during preparation: ' + fullname)
sys.meta_path.insert(0, NoModels())
from lidarpoc.config import RunConfig
from lidarpoc.runner import run
run(RunConfig.model_validate_json(sys.argv[1]), sys.argv[2])
'''
    result = subprocess.run([sys.executable, '-c', code, json.dumps(request), str(tmp_path / 'independent-prepare')],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'prepared [1/1]' in result.stdout
