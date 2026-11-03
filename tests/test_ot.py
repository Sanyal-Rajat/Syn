import json
from pathlib import Path

import numpy as np
import pytest
import torch

from lidarpoc.config import RunConfig
from lidarpoc.data.synthetic_scene import road_scene, scene_splits
from lidarpoc.data.split_manifest import digest, load_manifest
from lidarpoc.eval import downstream
from lidarpoc.eval.gates import decisions
from lidarpoc.eval.temporal import real_persistence
from lidarpoc.methods.sac import SACAgent
from lidarpoc.methods.registry import CONTROLLER_METHODS
from lidarpoc.rl import RL_METHODS
from lidarpoc.ot.config import ExperimentConfig, OTConfig
from lidarpoc.ot.constraints import physical_loss, pose_temporal_loss, scene_metrics, warp_depth
from lidarpoc.ot.features import conditional_ot, return_mass
from lidarpoc.ot.reference import build_reference, load_reference
from lidarpoc.ot.sinkhorn import kl, sinkhorn_divergence, transport
from lidarpoc.runner import read_examples, run


def test_sinkhorn_debiasing_gradient_and_detached_real_reference():
    source = torch.tensor([[0.], [.5], [1.]], dtype=torch.float64, requires_grad=True)
    same = sinkhorn_divergence(source, source.detach(), [1.], [1.], iterations=100)
    assert abs(float(same.detach())) < 1e-9
    shifted = source.detach() + .3
    loss = sinkhorn_divergence(source, shifted, [1.], [1.], iterations=100)
    assert loss > .02
    loss.backward()
    assert torch.isfinite(source.grad).all() and source.grad.mean() < 0
    a = torch.zeros((4, 5), requires_grad=True)
    b = torch.ones((4, 5), requires_grad=True)
    value, unsupported = conditional_ot({0: a, 1: a[:1]}, {0: b}, OTConfig(max_points=4, min_samples=2))
    value.backward()
    assert a.grad is not None and b.grad is None and unsupported == [1]
    with pytest.raises(ValueError, match='No supported'):
        conditional_ot({0: a[:1]}, {0: b}, OTConfig())


def test_unbalanced_unequal_mass_matches_scalar_solution_and_gradcheck():
    a = torch.tensor([.4], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([.8], dtype=torch.float64)
    cost = torch.zeros((1, 1), dtype=torch.float64)
    epsilon, tau = .2, 1.
    mass = (((epsilon + tau) * (a.log() + b.log())) / (epsilon + 2 * tau)).exp()
    expected = epsilon * kl(mass, a * b) + tau * kl(mass, a) + tau * kl(mass, b)
    actual = transport(cost, a, b, epsilon, 200, tau, tau)
    assert torch.allclose(actual, expected, atol=1e-9)
    assert torch.autograd.gradcheck(lambda x: transport(cost, x, b, epsilon, 200, tau, tau), (a,))
    actual.backward()
    assert a.grad.item() < 0
    assert transport(cost, torch.zeros_like(a), b, epsilon, 20, tau, tau) == pytest.approx(.8)
    with pytest.raises(ValueError, match='equal total masses'):
        transport(cost, a, b)


def test_return_mass_is_not_normalized_away():
    sensor = {'elevations': [-.1, .1], 'width': 4}
    config = OTConfig(ring_groups=2, azimuth_sectors=2)
    sample = {'range': torch.full((1, 2, 4), 10.), 'valid': torch.ones((1, 2, 4))}
    _, full = return_mass(sample, sensor, config)
    sample['valid'][:, 0] = 0
    _, half = return_mass(sample, sensor, config)
    assert full.sum() == 1 and half.sum() == .5


def test_train_only_reference_and_frozen_units(prepared, tmp_path):
    manifest = load_manifest(prepared)
    training = read_examples(manifest, 'train', 2, 1)
    validation = read_examples(manifest, 'validation', 2, 1)
    config = OTConfig(max_points=8, min_samples=2)
    path = build_reference(training, prepared, config, tmp_path / 'reference.npz')
    data, meta = load_reference(path, prepared, config)
    assert meta['split'] == 'train' and data['features'].shape[1] == 5
    with pytest.raises(ValueError, match='training traces only'):
        build_reference(validation, prepared, config, tmp_path / 'invalid.npz')
    with pytest.raises(ValueError, match='specification differs'):
        load_reference(path, prepared, config.model_copy(update={'feature_scales': [60., 1., 1., 1., 1.]}))


def test_scene_preservation_and_pose_compensated_noise(prepared, tmp_path):
    sensor = load_manifest(prepared)['sensor_meta']
    scene = road_scene(sensor, 2, 42)
    assert scene['range'].shape == (2, 4, 16)
    assert scene['valid'].any()
    assert scene_metrics(scene, scene, sensor, .1)['geometry_pass']
    broken = {**scene, 'range': scene['range'] + 1}
    assert not scene_metrics(broken, scene, sensor, .1)['geometry_pass']
    ranges = torch.full((4, 16), 10.)
    valid = torch.ones_like(ranges, dtype=torch.bool)
    depth, mask = warp_depth(ranges, valid, torch.eye(4), torch.eye(4), sensor)
    assert mask.all() and torch.allclose(depth, ranges, atol=1e-4)
    # Move the sensor one metre toward one known point along its calibrated ray.
    one_return = torch.zeros_like(valid)
    one_return[0, 8] = True
    elevation, azimuth = sensor['elevations'][0], sensor['azimuths'][8]
    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor([np.cos(elevation) * np.cos(azimuth),
                              np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
    moved, moved_mask = warp_depth(ranges, one_return, torch.eye(4), pose, sensor)
    assert moved_mask.sum() == 1 and moved[0, 8] == pytest.approx(9., abs=1e-4)
    aligned = {'range': np.stack([ranges.numpy(), moved.numpy()]),
               'valid': np.stack([one_return.numpy(), moved_mask.numpy()]),
               'world_from_sensor': np.stack([np.eye(4), pose.numpy()])}
    assert real_persistence(aligned, sensor, .1)['matched_return_fraction'] == 1.
    prediction = torch.full((1, 2, 4, 16), 10., requires_grad=True)
    batch = {'range': prediction.detach().clone(), 'valid': torch.ones_like(prediction, dtype=torch.bool),
             'world_from_sensor': torch.eye(4).repeat(1, 2, 1, 1)}
    loss, supported = pose_temporal_loss(prediction, batch, sensor, .1)
    assert supported and abs(float(loss.detach())) < 1e-4
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    settings = ExperimentConfig(synthetic_scenes=2)
    splits = scene_splits(sensor, 2, settings, tmp_path / 'scenes')
    assert set(splits) == {'train', 'validation', 'test'}


def test_sac_updates_both_critics_actor_and_targets():
    torch.manual_seed(0)
    agent = SACAgent(5, 2)
    before = {key: value.clone() for key, value in agent.q.state_dict().items()}
    transitions = [(np.zeros(5), np.array([.1, -.2]), .2, np.ones(5), 0.) for _ in range(4)]
    values = agent.update(transitions)
    assert agent.updates == 1 and all(np.isfinite(value) for value in values.values())
    assert any(not torch.equal(value, before[key]) for key, value in agent.q.state_dict().items())
    assert all(not parameter.requires_grad for parameter in agent.target.parameters())
    assert np.all(abs(agent.action(np.ones(5))) <= 1)


def test_downstream_probe_has_identical_explicit_label_budgets(prepared):
    sensor = load_manifest(prepared)['sensor_meta']
    examples = []
    for index in range(3):
        labels = np.indices((2, 4, 16))[-1] % 2
        examples.append({'range': np.full(labels.shape, 10., dtype='float32'), 'valid': np.ones(labels.shape, dtype=bool),
                         'intensity': labels.astype('float32'), 'labels': labels, 'sequence_id': f'trace_{index}'})
    model, report = downstream.fit(examples[:1], examples[1:2], examples[2:], sensor, 16, 16, 10, 0)
    assert report['real_labels'] == report['synthetic_labels'] == 16
    assert report['development'][0]['sequence_id'] == 'trace_2'
    assert downstream.metric(model, examples[2:], sensor, report['classes'])[0]['confusion']


def test_task_and_geometry_gates_never_claim_success_without_evidence():
    settings = ExperimentConfig(seeds=[0])
    pending = decisions([{'geometry_pass': True, 'pose_supported_windows': 0}], [], 'E4', settings)
    assert pending['task_transfer'] == 'not_evaluated' and pending['decision'].startswith('pending')
    rejected = decisions([{'geometry_pass': False, 'pose_supported_windows': 1}], [], 'E4', settings)
    assert rejected['decision'] == 'reject_geometry_failure'
    values = [{'E0': {'8': [{'sequence_id': 'test', 'mean_iou': .5}]},
               'E4': {'8': [{'sequence_id': 'test', 'mean_iou': .4}]}}]
    assert decisions([{'geometry_pass': True, 'pose_supported_windows': 1}], values, 'E4', settings)['decision'].startswith('reject_transfer')


def test_full_ot_matrix_controllers_and_no_test_fit_leakage(prepared, tmp_path, monkeypatch):
    import lidarpoc.runner as runner
    output = tmp_path / 'experiment'
    original = runner.read_examples

    def guarded(manifest, split, *args):
        if split == 'test':
            selection = json.loads((output / 'selection.json').read_text())
            assert selection['controllers_frozen'] and not selection['test_arrays_used_for_fitting']
        return original(manifest, split, *args)

    monkeypatch.setattr(runner, 'read_examples', guarded)
    config = RunConfig(action='experiment', manifest=prepared, gpus=0, channels=8, height=4, width=16,
        epochs=1, batch_size=2, window=2, diffusion_steps=10, sample_steps=2, samples_per_trace=1, budget=6,
        ot=OTConfig(max_points=8, min_samples=2, iterations=10),
        experiment=ExperimentConfig(seeds=[0], synthetic_scenes=2, sac_warmup=1,
            controllers=list(CONTROLLER_METHODS)))
    report = run(config, output)
    assert report['action'] == 'experiment' and report['legacy_workflows_retained']
    assert report['gates']['task_transfer'] == 'not_evaluated'
    assert not report['pretrained_weights_used']
    for name in ['E0', 'E1', 'E2', 'E3', 'E4', 'E5_sac', 'E5_ppo', 'E5_contextual_bandit']:
        assert name in report['session_intervals']
    for name, cost in report['compute'][0].items():
        if name.startswith('E5_'):
            assert cost['generation_calls'] == 6
            assert cost['denoiser_evaluations'] == 24
    checkpoint = torch.load(output / 'experiments/seed_0/E4/checkpoint.pt', weights_only=True)
    assert checkpoint['provenance']['ot_reference_split'] == 'train'
    assert 'ot/sinkhorn.py' in checkpoint['provenance']['library_sha256']
    policy_path = output / 'experiments/seed_0/E5_sac/controller.pt'
    policy = torch.load(policy_path, weights_only=True)
    assert policy['policy']['updates'] > 0
    assert set(RL_METHODS) <= set(report['gates']['rl_comparisons'])
    synthetic = output / 'scenes/test/000.npz'
    generated = run(RunConfig(action='generate', manifest=prepared,
        checkpoint=str(output / 'experiments/seed_0/E4/checkpoint.pt'), controller_file=str(policy_path),
        conditioning=str(synthetic), gpus=0, sample_steps=2, samples_per_trace=1), tmp_path / 'controlled_generation')
    assert generated['geometry']['new_surface_returns'] == 0
    assert generated['generation_controller'] == str(policy_path)
    for method in RL_METHODS:
        if method in ('ppo', 'sac'):
            continue
        path = output / f'experiments/seed_0/E5_{method}/controller.pt'
        bundle = torch.load(path, weights_only=True)
        assert bundle['policy']['updates'] > 0
        generated = run(RunConfig(action='generate', manifest=prepared,
            checkpoint=str(output / 'experiments/seed_0/E4/checkpoint.pt'), controller_file=str(path),
            conditioning=str(synthetic), gpus=0, sample_steps=2, samples_per_trace=1),
            tmp_path / f'generate_{method}')
        assert generated['geometry']['new_surface_returns'] == 0


def test_disabling_experimental_rl_excludes_every_rl_controller(prepared, tmp_path):
    config = RunConfig(action='experiment', manifest=prepared, gpus=0, channels=8, height=4, width=16,
        epochs=1, batch_size=2, window=2, diffusion_steps=10, sample_steps=2,
        samples_per_trace=1, budget=6, rl_enabled=False,
        ot=OTConfig(max_points=8, min_samples=2, iterations=10),
        experiment=ExperimentConfig(ablations=['E0', 'E4', 'E5'], seeds=[0], synthetic_scenes=1,
                                    controllers=list(CONTROLLER_METHODS)))
    result = run(config, tmp_path / 'no_rl')
    assert not any(f'E5_{method}' in result['session_intervals'] for method in (*RL_METHODS, 'contextual_bandit'))
    assert 'E5_fixed' in result['session_intervals'] and 'E5_bayesian' in result['session_intervals']


def test_labeled_experiment_uses_fixed_budgets_and_held_out_probes(prepared, tmp_path):
    manifest = load_manifest(prepared)
    for traces in manifest['splits'].values():
        for trace in traces:
            for frame in trace['frames']:
                with np.load(frame['path'], allow_pickle=False) as archive:
                    values = {key: archive[key] for key in archive.files}
                values['labels'] = np.where(values['valid'], np.indices(values['valid'].shape)[-1] % 2, -1)
                np.savez_compressed(frame['path'], **values)
                frame['sha256'] = digest(frame['path'])
    Path(prepared).write_text(json.dumps(manifest))
    files = []
    for seed in (0, 10000, 20000):
        scene = road_scene(manifest['sensor_meta'], 2, seed)
        scene['labels'] = np.where(scene['valid'], np.indices(scene['valid'].shape)[-1] % 2, -1)
        path = tmp_path / f'canonical_{seed}.npz'
        np.savez_compressed(path, **scene)
        files.append(str(path))
    settings = ExperimentConfig(ablations=['E0', 'E4'], seeds=[0], downstream=True,
        label_source_confirmed=True, label_budgets=[8, 16], downstream_steps=2,
        scene_files=files[:1], development_scene_files=files[1:2], test_scene_files=files[2:])
    config = RunConfig(action='experiment', manifest=prepared, gpus=0, channels=8, height=4, width=16,
        epochs=1, batch_size=2, window=2, diffusion_steps=10, sample_steps=2, samples_per_trace=1,
        ot=OTConfig(max_points=8, min_samples=2, iterations=10), experiment=settings)
    report = run(config, tmp_path / 'labeled_experiment')
    assert report['selection']['primary_metric'] == 'development_point_segmentation_mean_iou'
    for arm in ('E0', 'E4'):
        for budget in ('8', '16'):
            task = report['data_efficiency'][0][arm][budget]
            assert task['real_labels'] == task['synthetic_labels'] == int(budget)
            assert all(row['sequence_id'].startswith('fixture_test_') for row in report['task_evaluations'][0][arm][budget])
        cost = report['compute'][0][arm]
        assert cost['total_generation_calls'] == 6 and cost['allocated_total_gpu_hours'] == 0
        assert cost['downstream_cpu_seconds'] > 0
    assert report['gates']['task_transfer'] != 'not_evaluated'
