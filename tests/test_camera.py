import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from PIL import Image
import pytest

from lidarpoc.camera import draw_overlay, load_camera, project_points
from lidarpoc.camera_overlay import export_overlays
from lidarpoc.config import RunConfig
from lidarpoc.data.camera import matching_camera_frames, read_records
from lidarpoc.data.conditioning import export_real_scene
from lidarpoc.data.download import ensure_camera_files
from lidarpoc.data.split_manifest import load_manifest
from lidarpoc.provenance import license_gate


def test_optical_axes_scaling_depth_and_nearest_pixel(camera_inputs):
    _, calibration = camera_inputs
    camera = load_camera(calibration)
    points = np.array([[10, 0, 0], [10, -2, 0], [10, 0, -2], [10, 0, 2],
                       [-10, 0, 0], [10, 100, 0], [20, 0, 0], [np.nan, 0, 1]])
    pixels, depth, indices = project_points(points, camera, (160, 90))
    actual = {int(index): tuple(pixel) for index, pixel in zip(indices, pixels)}
    assert actual == {0: (80, 45), 1: (92, 45), 2: (80, 57), 3: (80, 33)}
    assert (depth == 10).all()
    small, _, indices = project_points(points[:1], camera, (80, 45))
    assert small.tolist() == [[40, 22]]
    assert indices.tolist() == [0]
    with pytest.raises(ValueError, match='aspect ratio'):
        project_points(points, camera, (80, 90))


def test_rational_distortion_and_explicit_calibration_validation(camera_inputs):
    _, calibration = camera_inputs
    values = json.loads(calibration.read_text())
    values.update(model='rational', distortion=[.1, -.01, .02, -.03, .001, .05, 0, 0])
    calibration.write_text(json.dumps(values))
    camera = load_camera(calibration)
    pixels, _, _ = project_points([[10, -2, -1]], camera, (160, 90))
    # Independent optical point (0.2,0.1): denominator K4, tangential P1/P2.
    radial = (1 + .1 * .05 - .01 * .05**2 + .001 * .05**3) / (1 + .05 * .05)
    expected = [int(60 * (.2 * radial + 2 * .02 * .2 * .1 - .03 * (.05 + 2 * .2**2)) + 80),
                int(60 * (.1 * radial + .02 * (.05 + 2 * .1**2) - 2 * .03 * .2 * .1) + 45)]
    assert pixels.tolist() == [expected]
    values['camera_from_sensor'][0][0] = 2
    calibration.write_text(json.dumps(values))
    with pytest.raises(ValueError, match='rigid'):
        load_camera(calibration)


def test_mount_translation_is_composed_before_optical_conversion(tmp_path):
    path = tmp_path / 'mounts.json'
    document = json.loads((Path(__file__).parents[1] / 'LICENSES/Pandar128-Configuration.json').read_text())
    for name, position in [('Pandar 128', [1, 0, 2]), ('ADAS Front Camera Wide', [2, 0, 1])]:
        document[name]['Mounting position'].update({f'{axis}_deg': '0' for axis in ('yaw', 'pitch', 'roll')})
        document[name]['Mounting position'].update({f'pos_{axis}_m': str(value) for axis, value in zip('xyz', position)})
    path.write_text(json.dumps(document))
    camera = load_camera(path, model='pinhole')
    np.testing.assert_allclose(camera['camera_from_sensor'] @ [10, 0, 0, 1], [0, -1, 9, 1])
    assert 'assumptions' in camera['convention']


@pytest.mark.parametrize('storage', ['stored', 'compressed', 'object'])
def test_camera_reader_supports_publisher_records_without_loading_numeric_images_into_ram(tmp_path, storage):
    records = np.zeros(3, dtype=[('t', 'f8'), ('data', object if storage == 'object' else 'u1',
                                           () if storage == 'object' else (4, 8, 3))])
    records['t'] = [0, .03, .06]
    for index in range(3):
        records['data'][index] = np.full((4, 8, 3), index, dtype='uint8')
    path = tmp_path / 'video.npz'
    (np.savez_compressed if storage == 'compressed' else np.savez)(path, data=records)
    with read_records(path) as actual:
        np.testing.assert_array_equal(actual['data'][2], np.full((4, 8, 3), 2, dtype='uint8'))
        if storage != 'object':
            assert isinstance(actual, np.memmap)


def test_timestamp_matching_uses_odometry_clock_and_rejects_skew(camera_inputs):
    scan, _ = camera_inputs
    frames = matching_camera_frames(scan, [0, 1], license_gate()['dataset'], lambda *_: None)
    assert [row['camera_frame'] for image, row in frames] == [2, 3]
    assert frames[0][0].getpixel((0, 0)) == (40, 0, 0)
    assert frames[1][1]['skew_ms'] == pytest.approx(16)
    with pytest.raises(ValueError, match='timestamp mismatch'):
        matching_camera_frames(scan, [0, 1], license_gate()['dataset'], lambda *_: None, max_skew_ms=2)
    odom = scan.with_name(scan.name.replace('_scan_list', '_odom_list'))
    np.savez(odom, data=np.zeros(4, dtype=[('transformation', 'f8', (4, 4))]))
    with pytest.raises(ValueError, match="'t' field"):
        matching_camera_frames(scan, [0], license_gate()['dataset'], lambda *_: None)


def test_camera_extraction_uses_only_matching_trace_and_reuses_files(tmp_path, monkeypatch):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', lambda **kwargs: pytest.fail('Cached archive must not download'))
    root = tmp_path / 'dataset'
    source = root / 'train/trace/trace_scan_list.npz'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'scan')
    (root / '.downloads').mkdir()
    with zipfile.ZipFile(root / '.downloads/train_part1.zip', 'w') as archive:
        archive.writestr('train/trace/trace_video_list.npz', b'camera')
        archive.writestr('train/trace/trace_odom_list.npz', b'odometry')
        archive.writestr('train/unrelated/unrelated_video_list.npz', b'other camera')
    video, odom = ensure_camera_files(source, license_gate()['dataset'], lambda *_: None)
    assert video.read_bytes() == b'camera' and odom.read_bytes() == b'odometry'
    assert not list(root.rglob('unrelated*'))
    (root / '.downloads/train_part1.zip').unlink()
    assert ensure_camera_files(source, license_gate()['dataset'], lambda *_: None) == [video, odom]


def test_export_synthetic_overlays_preserves_data_and_recovers_old_scene_metadata(prepared, camera_inputs, tmp_path):
    _, calibration = camera_inputs
    manifest = load_manifest(prepared)
    directory = tmp_path / 'generate'
    real_scene, _ = export_real_scene(manifest, 2, directory)
    with np.load(real_scene) as archive:
        scene = {key: archive[key] for key in archive.files}
    # Exercise files exported before camera provenance was added.
    scene.pop('source_scan_file')
    scene.pop('source_frame_indices')
    np.savez_compressed(real_scene, **scene)
    sample = directory / 'sample_000.npz'
    shutil.copy2(real_scene, sample)
    from lidarpoc.data.split_manifest import digest
    before = digest(sample)
    config = RunConfig(action='generate', manifest=prepared, checkpoint='/checkpoint.pt', camera_calibration=str(calibration))
    report = export_overlays(config, manifest, directory, real_scene)
    assert report['status'] == 'completed', report
    assert [row['camera_frame'] for row in report['frames']] == [2, 3]
    for row in report['frames']:
        real, synthetic = row['real'], row['synthetic'][0]
        assert real['projected_pixels'] == synthetic['projected_pixels'] > 0
        for record in (real, synthetic):
            with Image.open(directory / 'overlays' / record['file']) as image:
                assert image.size == (160, 90)
    assert digest(sample) == before
    assert (directory / 'overlays/calibration.json').is_file()
    assert json.loads((directory / 'overlays/manifest.json').read_text())['status'] == 'completed'
    report = export_overlays(config, manifest, directory, None)
    assert report['status'] == 'unavailable' and 'unconditional' in report['reason']
    assert export_overlays(config.model_copy(update={'camera_overlay': False}), manifest, directory, None)['status'] == 'disabled'
    sample.unlink()
    report = export_overlays(config, manifest, directory, real_scene)
    assert report['status'] == 'unavailable' and 'No sample_' in report['reason']


def test_overlay_changes_only_drawing_not_rgb_or_geometry(camera_inputs):
    _, calibration = camera_inputs
    image = Image.new('RGB', (160, 90), (30, 50, 70))
    points = np.array([[10., 0, 0]])
    original = points.copy()
    result, count = draw_overlay(image, points, load_camera(calibration), 'Synthetic LiDAR')
    assert count == 1 and result.getpixel((80, 45)) != image.getpixel((80, 45))
    assert image.getpixel((80, 45)) == (30, 50, 70)
    np.testing.assert_array_equal(points, original)


def test_existing_job_cli_adds_overlays_without_model_loading(prepared, camera_inputs, tmp_path):
    import subprocess
    import sys

    _, calibration = camera_inputs
    directory = tmp_path / 'old_generate'
    real_scene, _ = export_real_scene(load_manifest(prepared), 2, directory)
    shutil.copy2(real_scene, directory / 'sample_000.npz')
    config = RunConfig(action='generate', manifest=prepared, checkpoint='/deliberately/missing/checkpoint.pt',
                       camera_calibration=str(calibration))
    (directory / 'request.json').write_text(config.model_dump_json())
    (directory / 'result.json').write_text(json.dumps({'action': 'generate', 'samples': 1,
        'real_scene': real_scene, 'checkpoint': config.checkpoint}))
    result = subprocess.run([sys.executable, '-m', 'lidarpoc.camera_overlay', '--job', str(directory)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((directory / 'result.json').read_text())
    assert report['checkpoint'] == config.checkpoint
    assert report['camera_overlays']['status'] == 'completed'
    assert report['camera_overlays']['calibration_source'] == str(calibration)
    assert (directory / 'result.pdf').read_bytes().startswith(b'%PDF')
