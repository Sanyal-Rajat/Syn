"""Loopback-only remote compute API. Start with one Uvicorn worker behind SSH."""
from contextlib import asynccontextmanager
from functools import lru_cache
import io
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from threading import RLock, Thread
from uuid import uuid4
import zipfile

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from lidarpoc.config import RunConfig
from lidarpoc.data.download import DATASET_ROOT
from lidarpoc.io import write_json as save
from lidarpoc.logs import get_logger, read_log, relay_output
from lidarpoc.telemetry import gpu_usage

ROOT = Path(os.environ.get('LIDARPOC_RUNS', Path(__file__).parent / 'runs')).expanduser().resolve()
PROCESSES = {}
LOCK = RLock()
LOGGER = get_logger()


def job_dir(job_id):
    if not re.fullmatch(r'[0-9a-f]{32}', job_id):
        raise HTTPException(404, 'Unknown job')
    path = ROOT / job_id
    if not (path / 'state.json').is_file():
        raise HTTPException(404, 'Unknown job')
    return path


def stop_process(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


@asynccontextmanager
async def lifespan(_):
    ROOT.mkdir(parents=True, exist_ok=True)
    for state_file in ROOT.glob('*/state.json'):
        state = json.loads(state_file.read_text())
        if state['status'] == 'running':
            state['status'] = 'interrupted'
            save(state_file, state)
    yield
    with LOCK:
        processes = list(PROCESSES.values())
        for process in processes:
            stop_process(process)
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


app = FastAPI(title='LiDAR generation compute server', lifespan=lifespan)


@lru_cache(maxsize=1)
def system_info():
    from lidarpoc.gpu import hardware
    return hardware()


@app.get('/health')
def health():
    return {'status': 'ok', 'hardware': system_info(), 'dataset_root': str(DATASET_ROOT)}


@app.get('/metrics/gpu')
def metrics():
    return gpu_usage()


@app.get('/dataset/prepared')
def prepared():
    from lidarpoc.data.prepared import prepared_manifests
    return prepared_manifests(ROOT)


class GenerationInputsRequest(BaseModel):
    manifest: str = ''
    checkpoint: str = ''
    auto_conditioning: bool = True


@app.post('/generation/inputs')
def generation_inputs(request: GenerationInputsRequest):
    from lidarpoc.generation_inputs import prepare_generation_inputs
    try:
        return prepare_generation_inputs(ROOT, request.manifest, request.checkpoint, request.auto_conditioning)
    except (OSError, ValueError, KeyError) as error:
        raise HTTPException(422, str(error)) from error


class DatasetRequest(BaseModel):
    root: str = ''


@app.post('/dataset/inspect')
def inspect(request: DatasetRequest):
    from lidarpoc.data.pandar_reader import inspect_dataset
    try:
        return {'recordings': inspect_dataset(request.root or DATASET_ROOT)}
    except (ValueError, OSError) as error:
        raise HTTPException(422, str(error)) from error


def monitor(job_id, process, directory):
    code = relay_output(process, directory / 'worker.log')
    with LOCK:
        path = directory / 'state.json'
        state = json.loads(path.read_text())
        if state['status'] == 'running':
            state['status'] = 'completed' if code == 0 and (path.parent / 'result.json').is_file() else 'failed'
            state['returncode'] = code
            save(path, state)
        PROCESSES.pop(job_id, None)
        LOGGER.info('Job %s: %s (exit %s)', job_id, state['status'], code)


@app.post('/jobs', status_code=202)
def start(config: RunConfig):
    if config.action != 'prepare':
        from lidarpoc.data.split_manifest import load_manifest
        from lidarpoc.gpu import resolve_gpus
        try:
            load_manifest(config.manifest)
            if config.action == 'generate':
                from lidarpoc.generation_inputs import resolve_checkpoint
                config = config.model_copy(update={'checkpoint': resolve_checkpoint(config.checkpoint)})
            config = config.model_copy(update={'gpus': resolve_gpus(config.gpus)})
        except (ValueError, OSError, KeyError) as error:
            raise HTTPException(422, str(error)) from error
    with LOCK:
        if any(process.poll() is None for process in PROCESSES.values()):
            raise HTTPException(409, 'A compute job is already running. Wait or cancel it first.')
        job_id = uuid4().hex
        path = ROOT / job_id
        path.mkdir(parents=True)
        # Preserve which nested options the client supplied when handing off to the worker.
        (path / 'request.json').write_text(config.model_dump_json(indent=2, exclude_unset=True))
        state = {'id': job_id, 'status': 'running', 'action': config.action}
        save(path / 'state.json', state)
        try:
            process = subprocess.Popen([sys.executable, '-u', '-m', 'lidarpoc.runner',
                                         '--request', str(path / 'request.json'), '--output', str(path)],
                                        cwd=Path(__file__).parent, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        env={**os.environ, 'PYTHONUNBUFFERED': '1'}, start_new_session=True)
        except OSError as error:
            state['status'] = 'failed'
            save(path / 'state.json', state)
            raise HTTPException(500, str(error)) from error
        PROCESSES[job_id] = process
        LOGGER.info('Starting %s job %s; log: %s', config.action, job_id, path / 'worker.log')
        Thread(target=monitor, args=(job_id, process, path), daemon=True).start()
    return state


@app.get('/jobs/{job_id}/logs')
def logs(job_id: str, offset: int = Query(0, ge=0), limit: int = Query(65536, ge=4, le=262144)):
    return read_log(job_dir(job_id) / 'worker.log', offset, limit)


@app.get('/jobs')
def jobs():
    return [json.loads(path.read_text()) for path in sorted(ROOT.glob('*/state.json'),
                                                           key=lambda path: path.stat().st_mtime, reverse=True)[:50]]


@app.get('/jobs/{job_id}')
def status(job_id: str):
    path = job_dir(job_id)
    state = json.loads((path / 'state.json').read_text())
    if (path / 'progress.json').is_file():
        state['progress'] = json.loads((path / 'progress.json').read_text())
    if (path / 'training.json').is_file():
        state['training'] = json.loads((path / 'training.json').read_text())
    if (path / 'resources.json').is_file():
        state['resources'] = json.loads((path / 'resources.json').read_text())
    if (path / 'worker.log').is_file():
        with (path / 'worker.log').open('rb') as stream:
            stream.seek(max(0, (path / 'worker.log').stat().st_size - 12000))
            state['log'] = stream.read().decode(errors='replace')
    return state


@app.post('/jobs/{job_id}/cancel')
def cancel(job_id: str):
    with LOCK:
        path = job_dir(job_id)
        process = PROCESSES.get(job_id)
        if process is None or process.poll() is not None:
            raise HTTPException(409, 'Job is not running in this server')
        state = json.loads((path / 'state.json').read_text())
        state['status'] = 'cancelled'
        save(path / 'state.json', state)
        stop_process(process)
        LOGGER.info('Cancelling job %s', job_id)
    return state


@app.get('/jobs/{job_id}/result')
def result(job_id: str):
    path = job_dir(job_id) / 'result.json'
    if not path.is_file():
        raise HTTPException(409, 'Result is not available yet')
    return FileResponse(path, media_type='application/json', filename='result.json')


def pdf_report(directory):
    source, target = directory / 'result.json', directory / 'result.pdf'
    if not source.is_file():
        raise HTTPException(409, 'Result is not available yet')
    if not target.is_file() or target.stat().st_mtime_ns < source.stat().st_mtime_ns:
        from lidarpoc.report import write_pdf
        write_pdf(source, target)
    return target


@app.get('/jobs/{job_id}/report')
def report(job_id: str):
    directory = job_dir(job_id)
    state = json.loads((directory / 'state.json').read_text())
    if state['status'] != 'completed':
        raise HTTPException(409, 'Complete the job before exporting its PDF report')
    if json.loads((directory / 'result.json').read_text()).get('action') == 'generate':
        ensure_pixel_graphs(directory)
    return FileResponse(pdf_report(directory), media_type='application/pdf', filename='result.pdf')


@app.get('/jobs/{job_id}/comparison')
def comparison_viewer(job_id: str):
    directory = job_dir(job_id)
    ensure_pixel_graphs(directory)
    state = json.loads((directory / 'state.json').read_text())
    path = directory / 'comparison.html'
    if state['status'] != 'completed' or not path.is_file():
        raise HTTPException(409, 'Complete a paired Generate job before opening its comparison viewer')
    return FileResponse(path, media_type='text/html')


def ensure_pixel_graphs(directory, force=False):
    """Backfill completed Generate jobs when requested; no model execution."""
    from lidarpoc.pixel_graphs import VERSION, rebuild
    with LOCK:
        state = json.loads((directory / 'state.json').read_text())
        if state['status'] != 'completed':
            raise HTTPException(409, 'Complete the job before exporting pixel graphs')
        report = json.loads((directory / 'result.json').read_text())
        if report.get('action') != 'generate':
            raise HTTPException(409, 'Pixel graphs require a Generate job')
        pixels = report.get('pixel_analysis', {})
        files = [directory / 'pixel_graphs.html', directory / 'pixel_graphs/metrics.csv',
                 directory / 'pixel_graphs.json',
                 *(directory / row['maps'] for row in pixels.get('frames', [])),
                 *(directory / graph['file'] for row in pixels.get('frames', []) for graph in row.get('graphs', []))]
        if force or pixels.get('version') != VERSION or not all(path.is_file() for path in files):
            try:
                report = rebuild(directory)
            except (OSError, ValueError, KeyError) as error:
                raise HTTPException(422, f'Cannot build pixel graphs: {error}') from error
        return report


@app.post('/jobs/{job_id}/pixel-graphs/rebuild')
def rebuild_pixel_graphs(job_id: str):
    return ensure_pixel_graphs(job_dir(job_id), force=True)


@app.get('/jobs/{job_id}/pixel-graphs')
def pixel_viewer(job_id: str):
    directory = job_dir(job_id)
    report = ensure_pixel_graphs(directory)
    path = directory / 'pixel_graphs.html'
    if not report['pixel_analysis'].get('frames') or not path.is_file():
        raise HTTPException(409, report['pixel_analysis'].get('reason', 'Pixel graphs are unavailable'))
    return FileResponse(path, media_type='text/html')


@app.get('/jobs/{job_id}/pixel-graphs.zip')
def pixel_download(job_id: str):
    directory = job_dir(job_id)
    report = ensure_pixel_graphs(directory)
    if not report['pixel_analysis'].get('frames'):
        raise HTTPException(409, report['pixel_analysis'].get('reason', 'Pixel graphs are unavailable'))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for file in [directory / 'result.json', directory / 'pixel_graphs.json', directory / 'pixel_graphs.html',
                     *(directory / 'pixel_graphs').iterdir()]:
            if file.is_file() and file.resolve().is_relative_to(directory.resolve()):
                archive.write(file, str(file.relative_to(directory)))
    return Response(buffer.getvalue(), media_type='application/zip',
                    headers={'Content-Disposition': f'attachment; filename="pixel-graphs-{job_id}.zip"'})


@app.get('/jobs/{job_id}/pixel-graphs/{filename}')
def pixel_image(job_id: str, filename: str):
    folder = job_dir(job_id) / 'pixel_graphs'
    path = (folder / filename).resolve()
    if path.parent != folder.resolve() or path.suffix != '.png' or not path.is_file():
        raise HTTPException(404, 'Unknown pixel graph')
    return FileResponse(path, media_type='image/png')


@app.get('/jobs/{job_id}/artifacts')
def artifacts(job_id: str):
    path = job_dir(job_id)
    if not (path / 'result.json').is_file():
        raise HTTPException(409, 'Complete the job before exporting artifacts')
    if json.loads((path / 'result.json').read_text()).get('action') == 'generate':
        ensure_pixel_graphs(path)
    pdf_report(path)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for filename in ('request.json', 'result.json', 'result.pdf', 'dependency-lock.json', 'profiles.json', 'training.json', 'real_scene.npz',
                         'experiment_plan.json', 'selection.json', 'development.json', 'ot_reference.npz',
                         'resources.json', 'batch_backoff.json', 'comparison.json', 'comparison.html',
                         'pixel_graphs.json', 'pixel_graphs.html'):
            if (path / filename).is_file():
                archive.write(path / filename, filename)
        for pattern in ('*.pt', 'sample_*.npz'):
            for file in path.glob(pattern):
                archive.write(file, file.name)
        for file in (Path(__file__).parent / 'LICENSES').iterdir():
            if file.is_file():
                archive.write(file, f'LICENSES/{file.name}')
        for folder in ('experiments', 'scenes', 'overlays', 'comparison', 'pixel_graphs'):
            for file in (path / folder).rglob('*'):
                if file.is_file() and file.resolve().is_relative_to(path.resolve()):
                    archive.write(file, str(file.relative_to(path)))
    return Response(buffer.getvalue(), media_type='application/zip',
                    headers={'Content-Disposition': f'attachment; filename="lidarpoc-{job_id}.zip"'})


@app.get('/jobs/{job_id}/overlays/{filename}')
def overlay_image(job_id: str, filename: str):
    folder = job_dir(job_id) / 'overlays'
    path = (folder / filename).resolve()
    if (not path.is_relative_to(folder.resolve()) or path.parent != folder.resolve()
            or path.suffix not in ('.png', '.jpg') or not path.is_file()):
        raise HTTPException(404, 'Unknown overlay image')
    return FileResponse(path, media_type='image/png' if path.suffix == '.png' else 'image/jpeg')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=9000, workers=1, access_log=False)
