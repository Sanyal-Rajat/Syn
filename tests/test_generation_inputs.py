import json
import os
from pathlib import Path

import numpy as np
import pytest

from lidarpoc.data.split_manifest import load_manifest, load_window, windows
from lidarpoc.generation_inputs import prepare_generation_inputs, resolve_checkpoint


def training_job(runs, index, manifest, status='completed', window=2):
    folder = runs / f'{index:032x}'
    folder.mkdir(parents=True)
    (folder / 'checkpoint.pt').write_bytes(b'fixture checkpoint; discovery must not load model weights')
    (folder / 'state.json').write_text(json.dumps({'id': folder.name, 'action': 'train', 'status': status}))
    (folder / 'request.json').write_text(json.dumps({'manifest': str(manifest), 'window': window}))
    (folder / 'training.json').write_text('[{"epoch":1}]')
    (folder / 'result.json').write_text(json.dumps({'action': 'train', 'checkpoint': str(folder / 'checkpoint.pt')}))
    os.utime(folder / 'state.json', (index + 1, index + 1))
    return folder


def test_checkpoint_file_or_training_directory_and_missing_file(tmp_path):
    folder = tmp_path / 'training'
    folder.mkdir()
    with pytest.raises(ValueError, match='checkpoint.pt'):
        resolve_checkpoint(folder)
    checkpoint = folder / 'checkpoint.pt'
    checkpoint.write_bytes(b'weights')
    assert resolve_checkpoint(folder) == resolve_checkpoint(checkpoint) == str(checkpoint)
    with pytest.raises(ValueError, match='Checkpoint file not found'):
        resolve_checkpoint(folder / 'missing.pt')


def test_discovery_selects_matching_completed_training_and_creates_cached_scene(prepared, tmp_path):
    runs = tmp_path / 'runs'
    oldest = training_job(runs, 0, prepared)
    latest = training_job(runs, 1, prepared, window=3)
    training_job(runs, 2, prepared, status='failed')
    other = tmp_path / 'other-manifest.json'
    other.write_text(Path(prepared).read_text() + '\n')
    training_job(runs, 3, other)
    missing = training_job(runs, 4, prepared)
    (missing / 'checkpoint.pt').unlink()
    corrupt = training_job(runs, 5, prepared)
    (corrupt / 'request.json').write_text('{')
    result = prepare_generation_inputs(runs, prepared)
    assert [row['job_id'] for row in result['checkpoints']] == [latest.name, oldest.name]
    assert result['checkpoint'] == str(latest / 'checkpoint.pt')
    path = Path(result['conditioning'])
    manifest = load_manifest(prepared)
    identity, records = windows(manifest, 'train', 3)[0]
    expected = load_window(records, manifest['sensor_meta'], identity)
    with np.load(path, allow_pickle=False) as scene:
        for key in ('range', 'valid', 'intensity', 'timestamp_ns'):
            np.testing.assert_array_equal(scene[key], expected[key])
    timestamp = path.stat().st_mtime_ns
    assert prepare_generation_inputs(runs, prepared)['conditioning'] == str(path)
    assert path.stat().st_mtime_ns == timestamp
    assert not list(path.parent.glob('.scene-*'))


def test_manual_directory_resolves_its_window_and_unconditional_does_not_create_scene(prepared, tmp_path):
    runs = tmp_path / 'runs'
    first = training_job(runs, 0, prepared, window=1)
    training_job(runs, 1, prepared, window=3)
    result = prepare_generation_inputs(runs, prepared, str(first), False)
    assert result['checkpoint'] == str(first / 'checkpoint.pt') and not result['conditioning']
    assert not (tmp_path / 'datasets').exists()
    result = prepare_generation_inputs(runs, prepared, str(first))
    with np.load(result['conditioning']) as scene:
        assert scene['range'].shape == (1, 4, 16)


def test_no_matching_job_keeps_fields_empty_and_moved_checkpoints_remain_usable(prepared, tmp_path):
    result = prepare_generation_inputs(tmp_path / 'runs', prepared)
    assert not result['checkpoint'] and not result['conditioning'] and not result['checkpoints']
    assert str(tmp_path / 'runs') in result['diagnostic']
    moved = tmp_path / 'own-checkpoint.pt'
    moved.write_bytes(b'moved weights; no request metadata')
    result = prepare_generation_inputs(tmp_path / 'runs', prepared, str(moved))
    assert result['checkpoint'] == str(moved) and not result['conditioning']


@pytest.mark.parametrize('status', ['failed', 'interrupted', 'legacy', 'nested'])
def test_saved_checkpoints_are_recovered_without_completed_result(prepared, tmp_path, status):
    runs = tmp_path / 'runs'
    parent = runs / 'experiment' / 'experiments' / 'seed_0' if status == 'nested' else runs
    folder = training_job(parent, 0, prepared, status=status)
    (folder / 'result.json').unlink()
    if status in ('legacy', 'nested'):
        (folder / 'state.json').unlink()
    result = prepare_generation_inputs(runs, prepared)
    assert result['checkpoint'] == str(folder / 'checkpoint.pt')
    assert Path(result['conditioning']).is_file()
    assert 'saved checkpoint' in result['diagnostic']


def test_recovery_rejects_running_empty_and_mismatched_checkpoints(prepared, tmp_path):
    runs = tmp_path / 'runs'
    training_job(runs, 0, prepared, status='running')
    other = tmp_path / 'other.json'
    other.write_text(Path(prepared).read_text() + '\n')
    training_job(runs, 1, other, status='failed')
    empty = training_job(runs, 2, prepared, status='failed')
    (empty / 'checkpoint.pt').write_bytes(b'')
    assert not prepare_generation_inputs(runs, prepared)['checkpoint']


@pytest.mark.parametrize('history', [None, [], [{'epoch': 0}], ['invalid record']])
def test_recovery_does_not_select_an_untrained_checkpoint(prepared, tmp_path, history):
    folder = training_job(tmp_path / 'runs', 0, prepared, status='failed')
    path = folder / 'training.json'
    if history is None:
        path.unlink()
    else:
        path.write_text(json.dumps(history))
    assert not prepare_generation_inputs(tmp_path / 'runs', prepared)['checkpoint']


def test_latest_successful_run_provides_all_generation_defaults(prepared, tmp_path):
    runs = tmp_path / 'runs'
    training_job(runs, 0, prepared)
    older_manifest = tmp_path / 'training-manifest.json'
    older_manifest.write_text(Path(prepared).read_text() + '\n')
    latest = training_job(runs, 1, older_manifest, window=3)
    request = {'manifest': str(older_manifest), 'window': 3, 'gpus': 2, 'sample_steps': 12,
               'samples_per_trace': 7, 'auto_optimize': False, 'seed': 37, 'batch_size': 6, 'diffusion_steps': 40}
    (latest / 'request.json').write_text(json.dumps(request))
    (latest / 'profiles.json').write_text(json.dumps({'profiles': {'bayesian': {'theta': [0, 0, 0, 1]}}}))
    result = json.loads((latest / 'result.json').read_text())
    result['selected_profile_on_validation'] = 'bayesian'
    (latest / 'result.json').write_text(json.dumps(result))
    training_job(runs, 2, prepared, status='failed')
    training_job(runs, 3, prepared, status='running')
    found = prepare_generation_inputs(runs)
    assert found['training_job_id'] == latest.name
    defaults = found['defaults']
    assert defaults['manifest'] == str(older_manifest)
    assert defaults['checkpoint'] == str(latest / 'checkpoint.pt')
    for name in ('gpus', 'sample_steps', 'samples_per_trace', 'auto_optimize', 'seed', 'batch_size', 'diffusion_steps', 'window'):
        assert defaults[name] == request[name]
    assert defaults['profile_file'] == str(latest / 'profiles.json') and defaults['profile_name'] == 'bayesian'
    assert defaults['controller_file'] == ''
    assert defaults['auto_conditioning']
    with np.load(defaults['conditioning']) as scene:
        assert scene['range'].shape == (3, 4, 16)


def test_latest_defaults_do_not_invent_optional_files_or_use_failed_jobs(prepared, tmp_path):
    runs = tmp_path / 'runs'
    training_job(runs, 0, prepared, status='failed')
    assert not prepare_generation_inputs(runs)['defaults']
    folder = training_job(runs, 1, prepared)
    defaults = prepare_generation_inputs(runs, auto_conditioning=False)['defaults']
    assert defaults['checkpoint'] == str(folder / 'checkpoint.pt')
    assert defaults['profile_file'] == defaults['controller_file'] == defaults['conditioning'] == ''
    assert defaults['profile_name'] == 'uncalibrated'
    (folder / 'ppo_policy.pt').write_bytes(b'calibration policy is not an OT generation controller')
    assert prepare_generation_inputs(runs, auto_conditioning=False)['defaults']['controller_file'] == ''
    controller = folder / 'controller.pt'
    controller.write_bytes(b'controller explicitly recorded by this training run')
    result_path = folder / 'result.json'
    result = json.loads(result_path.read_text())
    result['generation_controller'] = str(controller)
    result_path.write_text(json.dumps(result))
    assert prepare_generation_inputs(runs, auto_conditioning=False)['defaults']['controller_file'] == str(controller)
