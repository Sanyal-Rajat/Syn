from pathlib import Path
import numpy as np
import pytest

from lidarpoc.config import RunConfig
from lidarpoc.data.annotations import load_poses
from lidarpoc.data.split_manifest import load_manifest, load_window, prepare
from lidarpoc.data.synthetic_scene import road_scene, scene_splits
from lidarpoc.ot.config import ExperimentConfig
from lidarpoc.provenance import license_gate


def test_preparation_retains_native_labels_and_confirmed_pose_convention(prepared, tmp_path):
    raw = Path(prepared).parents[1] / 'raw'
    for source in raw.glob('**/*_scan_list.npz'):
        records = np.zeros(4, dtype=[('transformation', 'f4', (4, 4))])
        records['transformation'] = np.eye(4)
        records['transformation'][:, 0, 3] = np.arange(4) * .05
        np.savez_compressed(str(source).replace('_scan_list.npz', '_odom_list.npz'), data=records)
    config = RunConfig(action='prepare', dataset_root=str(raw), input_frame='sensor',
        elevations_file=str(tmp_path / 'beams.txt'), height=4, width=16, max_frames=4, odometry_mode='world_from_vehicle')
    result = prepare(config, tmp_path / 'with_annotations', license_gate())
    manifest = load_manifest(result, verify_files=True)
    trace = manifest['splits']['train'][0]
    sample = load_window(trace['frames'][:2], manifest['sensor_meta'], trace['sequence_id'])
    assert sample['labels'].shape == (2, 4, 16)
    assert (sample['labels'][sample['valid']] == 0).all()
    assert (sample['labels'][~sample['valid']] == -1).all()
    assert sample['world_from_sensor'].shape == (2, 4, 4)
    assert trace['odometry_sha256'] and trace['odometry_mode'] == 'world_from_vehicle'
    source = next(raw.glob('**/*_scan_list.npz'))
    poses = load_poses(source, 'previous_from_current', np.eye(4))
    assert poses[2, 0, 3] == pytest.approx(.15)


def test_external_canonical_scene_overlap_and_unconfirmed_labels_are_rejected(prepared, tmp_path):
    sensor = load_manifest(prepared)['sensor_meta']
    path = tmp_path / 'same_scene.npz'
    np.savez_compressed(path, **road_scene(sensor, 2))
    settings = ExperimentConfig(scene_files=[str(path)], development_scene_files=[str(path)], test_scene_files=[str(path)])
    with pytest.raises(ValueError, match='overlaps'):
        scene_splits(sensor, 2, settings, tmp_path / 'scenes')
    with pytest.raises(ValueError, match='Confirm the semantic label source'):
        ExperimentConfig(downstream=True)
