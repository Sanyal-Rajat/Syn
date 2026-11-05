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
    assert result == {'checkpoint': '', 'conditioning': '', 'checkpoints': []}
    moved = tmp_path / 'own-checkpoint.pt'
    moved.write_bytes(b'moved weights; no request metadata')
    result = prepare_generation_inputs(tmp_path / 'runs', prepared, str(moved))
    assert result['checkpoint'] == str(moved) and not result['conditioning']
