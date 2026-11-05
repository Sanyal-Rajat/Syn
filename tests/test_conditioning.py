import numpy as np
import pytest

from lidarpoc.config import RunConfig
from lidarpoc.data.conditioning import export_real_scene
from lidarpoc.data.split_manifest import digest, load_manifest, load_window, windows
from view_syn_npz import load_frame


def test_automatic_real_scene_matches_first_training_window_and_is_viewable(prepared, tmp_path):
    manifest = load_manifest(prepared)
    identity, records = windows(manifest, 'train', 2)[0]
    expected = load_window(records, manifest['sensor_meta'], identity)
    path, origin = export_real_scene(manifest, 2, tmp_path / 'generate')
    assert origin == {'kind': 'prepared', 'split': 'train', 'sequence_id': identity, 'frames': records}
    with np.load(path, allow_pickle=False) as scene:
        for key in ('range', 'valid', 'intensity', 'has_intensity', 'labels', 'timestamp_ns'):
            np.testing.assert_array_equal(scene[key], expected[key])
        np.testing.assert_array_equal(scene['elevations'], manifest['sensor_meta']['elevations'])
        assert str(scene['sequence_id']) == identity
    assert load_frame(path)[-1].shape[1] == 3
    assert all(digest(record['path']) == record['sha256'] for record in records)


def test_manual_scene_is_exported_without_using_prepared_arrays(prepared, tmp_path):
    manifest = load_manifest(prepared)
    sensor = manifest['sensor_meta']
    source = tmp_path / 'manual.npz'
    ranges = np.full((2, 4, 16), 17., dtype='float32')
    np.savez_compressed(source, range=ranges, valid=np.ones_like(ranges, dtype=bool),
                        elevations=sensor['elevations'])
    before = digest(source)
    manifest['splits']['train'] = []
    path, origin = export_real_scene(manifest, 2, tmp_path / 'generate', str(source))
    assert origin == {'kind': 'file', 'path': str(source)}
    with np.load(path, allow_pickle=False) as scene:
        np.testing.assert_array_equal(scene['range'], ranges)
    assert digest(source) == before


def test_automatic_scene_fails_clearly_without_a_complete_window(prepared, tmp_path):
    with pytest.raises(ValueError, match='complete 5-frame training window'):
        export_real_scene(load_manifest(prepared), 5, tmp_path / 'generate')


@pytest.mark.parametrize('problem', ['shape', 'beams', 'empty'])
def test_invalid_manual_scene_is_rejected_before_export(prepared, tmp_path, problem):
    manifest = load_manifest(prepared)
    ranges = np.ones((1 if problem == 'shape' else 2, 4, 16), dtype='float32')
    source = tmp_path / 'invalid.npz'
    np.savez_compressed(source, range=ranges, valid=np.full_like(ranges, problem != 'empty', dtype=bool),
                        elevations=np.asarray(manifest['sensor_meta']['elevations']) + (1 if problem == 'beams' else 0))
    with pytest.raises(ValueError):
        export_real_scene(manifest, 2, tmp_path / 'generate', str(source))
    assert not (tmp_path / 'generate' / 'real_scene.npz').exists()


def test_controller_accepts_automatic_conditioning_and_keeps_unconditional_guard():
    values = {'action': 'generate', 'manifest': '/manifest.json', 'checkpoint': '/checkpoint.pt',
              'controller_file': '/controller.pt'}
    assert RunConfig(**values).auto_conditioning
    with pytest.raises(ValueError, match='conditioning NPZ or automatic'):
        RunConfig(**values, auto_conditioning=False)
    assert RunConfig(**values, auto_conditioning=False, conditioning='/scene.npz').conditioning
