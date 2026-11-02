import json

import numpy as np
import pytest
import torch

from lidarpoc.calibration import apply, calibrate
from lidarpoc.config import RunConfig
from lidarpoc.data.pandar_reader import unpack_scan
from lidarpoc.data.scan_to_range import project, unproject
from lidarpoc.data.split_manifest import load_manifest, load_window
from lidarpoc.eval.domain_metrics import compare, describe
from lidarpoc.models.lidardm_adapter import LidarDMAdapter
from lidarpoc.models.masking import masked_mse
from lidarpoc.provenance import license_gate, validate_checkpoint
from lidarpoc.runner import read_examples, run

torch.set_num_threads(1)


def test_published_npz_schema_and_split_integrity(prepared):
    manifest = load_manifest(prepared, verify_files=True)
    assert len(manifest['splits']['train']) == 2
    trace = manifest['splits']['train'][0]
    batch = load_window(trace['frames'][:2], manifest['sensor_meta'], trace['sequence_id'])
    assert batch['range'].shape == (2, 4, 16)
    assert batch['valid'].dtype == bool
    assert batch['timestamp_ns'].dtype == np.int64
    assert (batch['range'][~batch['valid']] == 0).all()
    assert batch['has_intensity'].all()


def test_legacy_vector_xyz_schema():
    points = np.zeros(2, dtype=[('x', 'f4', (3,)), ('i', 'f4')])
    points['x'] = [[1, 2, 3], [4, 5, 6]]
    points['i'] = [8, 9]
    xyz, intensity = unpack_scan(points)
    np.testing.assert_array_equal(xyz, [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(intensity, [8, 9])
    with pytest.raises(ValueError, match='Unrecognized'):
        unpack_scan(np.zeros((3, 4)))
    published = np.array([(0, 1., 2., 3., 50.), (1, 4., 5., 6., 60.)],
                         dtype=[('Classification', 'u1'), ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'f4')])
    lossy_object_frames = np.array([published, published], dtype=object)
    xyz, intensity = unpack_scan(lossy_object_frames[0])
    np.testing.assert_array_equal(xyz, [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(intensity, [50, 60])


def test_nearest_return_projection_and_masks():
    elevations = np.array([-.1, .1])
    points = np.array([[10 * np.cos(.1), 0, 10 * np.sin(.1)],
                       [5 * np.cos(.1), 0, 5 * np.sin(.1)], [0., 0., 0.]])
    result = project(points, np.array([100, 200, 0]), elevations, width=8)
    assert result['valid'].sum() == 1
    assert result['range'][result['valid']][0] == pytest.approx(5.)
    assert np.linalg.norm(unproject(result['range'], result['valid'], elevations)[0]) == pytest.approx(5.)
    prediction = torch.tensor([3., 100.], requires_grad=True)
    loss = masked_mse(prediction, torch.zeros(2), torch.tensor([True, False]))
    loss.backward()
    assert prediction.grad.tolist() == [6., 0.]


def test_leakage_and_provenance_gates(prepared, tmp_path):
    manifest = load_manifest(prepared)
    changed = json.loads(json.dumps(manifest))
    changed['splits']['test'][0]['sequence_id'] = changed['splits']['train'][0]['sequence_id']
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='leakage'):
        load_manifest(bad)
    changed['revision'] = 'unapproved'
    with pytest.raises(ValueError, match='revision'):
        license_gate(changed)
    with pytest.raises(ValueError, match='provenance'):
        validate_checkpoint({'provenance': {'initialization': 'pretrained'}}, prepared)
    with pytest.raises(ValueError, match='random initialization'):
        RunConfig(manifest=prepared, checkpoint='/some/other/weights.pt')


@pytest.mark.parametrize('method', ['ppo', 'random_search', 'bayesian', 'cma_es'])
def test_rl_and_baselines_are_real_budgeted_optimizers(prepared, method):
    manifest = load_manifest(prepared)
    real = read_examples(manifest, 'train', 2, 1)
    val = read_examples(manifest, 'validation', 2, 1)
    generated = apply(real, [.1, .1, .1, 1.2], 0)
    result = calibrate(method, real, generated, val, generated, 120., 6)
    assert len(result['history']) == 6
    assert result['evaluations'] == 6
    assert np.isfinite(result['theta']).all()
    if method == 'ppo':
        assert result['policy']


def test_random_diffusion_training_sampling_and_conditioning(prepared, tmp_path):
    config = RunConfig(manifest=prepared, gpus=0, height=4, width=16, channels=8, epochs=1,
                       batch_size=2, window=2, diffusion_steps=10, sample_steps=2,
                       samples_per_trace=1, budget=6, methods=['ppo', 'random_search', 'bayesian'])
    output = tmp_path / 'run'
    report = run(config, output)
    assert report['pretrained_weights_used'] is False
    assert 'ppo' in report['profiles']
    assert (output / 'ppo_policy.pt').is_file()
    assert {'training_resampling', 'per_ring_empirical', 'uncalibrated', 'ppo'} <= report['evaluations'].keys()
    checkpoint = torch.load(output / 'checkpoint.pt', weights_only=True)
    validate_checkpoint(checkpoint, prepared)
    model = LidarDMAdapter(checkpoint['config'], checkpoint['sensor_meta'])
    model.load_state_dict(checkpoint['state_dict'])
    geometry = torch.full((1, 2, 4, 16), 10.)
    valid = torch.zeros_like(geometry, dtype=torch.bool)
    valid[..., :2, :] = True
    sample = model.sample(1, {'range': geometry, 'valid': valid, 'complete_scene': True}, 3, 2)
    assert not sample['valid'][~valid].any()
    assert ((sample['range'][sample['valid']] - 10.).abs() <= .10001).all()
    assert torch.isfinite(sample['range']).all()
