"""Additive E0–E5 experiment runner; freeze all choices before decoding real test arrays."""
from pathlib import Path
import time
import json

import numpy as np
import torch

from .data.split_manifest import digest, load_manifest
from .data.synthetic_scene import geometry_digest, scene_splits
from .eval import downstream
from .eval.gates import decisions
from .eval.ot_metrics import session_intervals
from .io import write_json
from .gpu import launch_training, resolve_gpus
from .logs import progress
from .models.lidardm_adapter import LidarDMAdapter
from .ot.reference import build_reference, load_reference
from .provenance import license_gate, provenance, validate_checkpoint
from .rl import RL_METHODS
from .rl.comparison import compare as compare_rl


def cpu_policy(policy):
    if isinstance(policy, np.ndarray):
        return torch.from_numpy(policy.copy())
    if isinstance(policy, dict):
        return {key: cpu_policy(value) for key, value in policy.items()}
    return policy


def save_bank(path, samples):
    path.mkdir(parents=True, exist_ok=True)
    for index, sample in enumerate(samples):
        np.savez_compressed(path / f'sample_{index:03d}.npz', **sample)


def load_model(path, manifest, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = LidarDMAdapter(checkpoint['config'], manifest['sensor_meta']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def arm_config(config, arm, reference, sensor, seed):
    order = int(arm[1])
    weights = config.experiment
    ot = config.ot.model_copy(update={'reference': reference,
        'conditional_weight': weights.conditional_weight if order >= 2 else 0.,
        'mass_weight': weights.mass_weight if order >= 3 else 0.,
        'physics_weight': weights.physics_weight if order >= 4 else 0.,
        'regularization_weight': weights.regularization_weight if order >= 4 else 0.})
    return config.model_copy(update={'action': 'train', 'checkpoint': '', 'warm_start': config.checkpoint,
        'seed': seed, 'height': len(sensor['elevations']), 'width': sensor['width'],
        'max_range_m': sensor['max_range_m'], 'ot': ot,
        'geometry_weight': config.geometry_weight if order >= 4 else 0.,
        'temporal_weight': config.temporal_weight if order >= 4 else 0.})


def train_arm(config, directory, manifest, arm):
    directory.mkdir(parents=True, exist_ok=True)
    request = directory / 'request.json'
    write_json(request, config.model_dump())
    started = time.perf_counter()
    if arm == 'E0':
        torch.manual_seed(config.seed)
        model = LidarDMAdapter(config.model_dump(), manifest['sensor_meta'])
        if config.warm_start:
            parent = torch.load(config.warm_start, map_location='cpu', weights_only=True)
            validate_checkpoint(parent, config.manifest)
            model.load_state_dict(parent['state_dict'])
        proof = provenance(config.manifest, config.model_dump())
        if config.warm_start:
            proof['stage_initialization'] = 'own_checkpoint'
            proof['parent_checkpoint_sha256'] = digest(config.warm_start)
        torch.save({'state_dict': model.state_dict(), 'config': config.model_dump(), 'sensor_meta': manifest['sensor_meta'],
                    'provenance': proof, 'epoch': 0}, directory / 'checkpoint.pt')
    else:
        launch_training(config, directory)
    elapsed = time.perf_counter() - started
    resources = directory / 'resources.json'
    return {'training_seconds': elapsed, 'allocated_training_gpu_hours': elapsed * config.gpus / 3600 if arm != 'E0' else 0.,
            'epochs': 0 if arm == 'E0' else config.epochs, 'seed': config.seed,
            'training_resources': json.loads(resources.read_text()) if resources.is_file() else {}}


def run_experiment(config, directory):
    from .runner import read_examples

    directory = Path(directory).resolve()
    config = config.model_copy(update={'gpus': resolve_gpus(config.gpus)})
    manifest = load_manifest(config.manifest, verify_files=True)
    license_gate(manifest)
    torch.set_num_threads(4 if config.gpus else 1)
    sensor, settings = manifest['sensor_meta'], config.experiment
    if config.checkpoint:
        validate_checkpoint(torch.load(config.checkpoint, map_location='cpu', weights_only=True), config.manifest)
    progress(directory, 'freezing training-only OT reference and canonical scene splits')
    training = read_examples(manifest, 'train', config.window, config.samples_per_trace)
    reference_path = build_reference(training, config.manifest, config.ot, directory / 'ot_reference.npz')
    _, reference_metadata = load_reference(reference_path, config.manifest, config.ot)
    scenes = scene_splits(sensor, config.window, settings, directory / 'scenes')
    plan = {'specification': 'sim_to_real_OT_diffusion_RL_updated_report.pdf',
            'specification_sha256': '3cb5b0d8d6cae8d83264a73684a86753b7b1c75cd1ab58523f61591a28ab9ffa',
            'config': config.model_dump(),
            'reference_sha256': digest(reference_path), 'reference_metadata': reference_metadata,
            'canonical_geometry': {split: [geometry_digest(scene) for scene in values] for split, values in scenes.items()},
            'scene_source': 'external_simulator_npz' if settings.scene_files else 'procedural_road_plane_boxes_lane_markings',
            'primary_metric': 'development_point_segmentation_mean_iou' if settings.downstream else 'development_OT_gap_surrogate',
            'task_probe': 'small point-wise semantic MLP, not a production perception benchmark',
            'test_arrays_used_for_fitting': False, 'mass_objective': 'raw KL-regularized unbalanced OT',
            'strata_scope': 'range and ring proxies; not matched road type/material/incidence labels',
            'compute_scheduling': 'independent jobs on disjoint GPU groups; DDP within diffusion jobs',
            'automatic_throughput_tuning': config.auto_optimize}
    write_json(directory / 'experiment_plan.json', plan)
    from . import experiment_tasks as tasks
    from .scheduling import map_tasks

    seeds = {seed: {} for seed in settings.seeds}
    validations = {seed: {} for seed in settings.seeds}
    computations = {seed: {} for seed in settings.seeds}
    task_models, task_development = {}, {}
    base = {'root': str(directory), 'config': config.model_dump()}
    arms = [name for name in settings.ablations if name != 'E5']
    jobs = [{**base, 'seed': seed, 'name': name} for seed in settings.seeds for name in arms]

    def collect(results):
        for result in results:
            seed, name = result['seed'], result['name']
            seeds[seed][name] = result['record']
            validations[seed][name] = result['validation']
            computations[seed][name] = result['cost']

    collect(map_tasks(tasks.development, jobs, config.gpus,
        on_done=lambda done, total: progress(directory, 'diffusion development jobs', done, total)))
    if 'E5' in settings.ablations:
        jobs = [{**base, 'seed': seed, 'name': f'E5_{method}'} for seed in settings.seeds
                for method in settings.controllers
                if config.rl_enabled or method not in (*RL_METHODS, 'contextual_bandit')]
        collect(map_tasks(tasks.development, jobs, config.gpus,
            on_done=lambda done, total: progress(directory, 'independent frozen-generator controllers', done, total)))
    if settings.downstream:
        jobs = []
        available_real = len(downstream.labeled_points(training, sensor)[1])
        for seed, records in seeds.items():
            common = min(len(downstream.labeled_points(tasks.load_bank(
                directory / 'experiments' / f'seed_{seed}' / name / 'training_samples'), sensor)[1]) for name in records)
            task_models[seed], task_development[seed] = {}, {}
            for name in records:
                task_models[seed][name], task_development[seed][name] = {}, {}
                for budget in settings.label_budgets:
                    jobs.append({**base, 'seed': seed, 'name': name, 'budget': budget,
                                 'real_count': min(budget, available_real), 'synthetic_count': min(budget, common)})
        results = map_tasks(tasks.probe, jobs, config.gpus,
            on_done=lambda done, total: progress(directory, 'identical-budget downstream probes', done, total))
        for result in results:
            seed, name, budget = result['seed'], result['name'], result['budget']
            task_models[seed][name][budget] = result['path']
            task_development[seed][name][budget] = result['report']
            cost = computations[seed][name]
            cost['downstream_seconds'] = cost.get('downstream_seconds', 0.) + result['seconds']
            cost['downstream_cpu_seconds'] = cost.get('downstream_cpu_seconds', 0.) + result['cpu_seconds']
            cost['allocated_probe_gpu_hours'] = cost.get('allocated_probe_gpu_hours', 0.) + result['gpu_hours']
    write_json(directory / 'development.json', {'sensor': validations, 'task': task_development, 'compute': computations})
    candidates = sorted(set.intersection(*(set(rows) for rows in validations.values())))
    def selection_score(name):
        if settings.downstream:
            return -float(np.mean([row['mean_iou'] for rows in task_development.values()
                                  for probe in rows[name].values() for row in probe['development']]))
        return float(np.mean([rows[name]['gap']['mean'] for rows in validations.values()]))
    selected = min(candidates, key=selection_score)
    comparators = [name for name in candidates if name in ('E5_fixed', 'E5_random_search', 'E5_bayesian', 'E5_cma_es')]
    rl_comparator = min(comparators, key=selection_score) if comparators else None
    selection = {'selected_arm': selected, 'scores': {name: selection_score(name) for name in candidates},
                 'primary_metric': plan['primary_metric'], 'seeds': settings.seeds,
                 'test_arrays_used_for_fitting': False, 'controllers_frozen': True, 'rl_comparator': rl_comparator}
    write_json(directory / 'selection.json', selection)
    progress(directory, 'selection frozen; decoding final real test sessions')
    # No test window or semantic test label is decoded before the immutable selection above.
    jobs = [{**base, 'seed': seed, 'name': name, 'record': record,
             'probes': task_models.get(seed, {}).get(name, {})}
            for seed, records in seeds.items() for name, record in records.items()]
    results = map_tasks(tasks.test, jobs, config.gpus,
        on_done=lambda done, total: progress(directory, 'frozen held-out evaluation jobs', done, total))
    tests, task_rows = {seed: {} for seed in settings.seeds}, {seed: {} for seed in settings.seeds}
    for result in results:
        seed, name = result['seed'], result['name']
        tests[seed][name] = result['metrics']
        if settings.downstream:
            task_rows[seed][name] = result['task']
        cost = computations[seed][name]
        cost['test_seconds'] = result['seconds']
        training_cost = computations[seed][cost.get('shared_training_arm', name)]['allocated_training_gpu_hours']
        inference_hours = ((cost.get('development_seconds', 0.) + cost.get('synthetic_training_bank_seconds', 0.))
                           * cost['inference_gpus'] + result['seconds'] * result['gpus']) / 3600
        cost['allocated_total_gpu_hours'] = (training_cost + cost.get('allocated_controller_gpu_hours', 0.)
                                             + cost.get('allocated_probe_gpu_hours', 0.) + inference_hours)
        gain = selection_score('E0') - selection_score(name) if 'E0' in candidates else None
        cost['development_gain_over_E0'] = gain
        cost['gpu_hours_per_positive_development_gain'] = cost['allocated_total_gpu_hours'] / gain if gain is not None and gain > 0 else None
        count = cost['evaluation_calls_per_phase']
        cost['total_generation_calls'] = cost.get('generation_calls', 0) + count * (3 if settings.downstream else 2)
        cost['calls_per_positive_development_gain'] = cost['total_generation_calls'] / gain if gain is not None and gain > 0 else None
    task_tests = [task_rows[seed] for seed in settings.seeds]
    gates = decisions([rows[selected] for rows in tests.values()], task_tests, selected, settings)
    comparisons = compare_rl(candidates, rl_comparator, selection_score, task_tests, tests,
                             computations, config.budget, settings)
    if comparisons:
        gates['rl_comparisons'] = comparisons
    if 'sac' in comparisons:
        gates['rl_development_comparison'] = comparisons['sac']
    report = {'action': 'experiment', 'specification': plan['specification'], 'selected_arm': selected,
              'selection': selection, 'selection_sha256': digest(directory / 'selection.json'),
              'reference': reference_path, 'reference_sha256': digest(reference_path),
              'development': validations, 'evaluations': tests, 'task_evaluations': task_tests,
              'data_efficiency': task_development, 'compute': computations, 'gates': gates,
              'session_intervals': {name: session_intervals([rows[name] for rows in tests.values()]) for name in candidates},
              'pretrained_weights_used': False, 'legacy_workflows_retained': True,
              'limitations': ['Fixed range/ring strata are confounding proxies, not true scene-condition matching.',
                  'Procedural road/box scenes are simple simulator ground truth, not complete driving simulation.',
                  'The downstream probe is a small point-wise semantic classifier, not a production detector.',
                  'Ray-mask occlusion checks do not certify full volumetric scene reconstruction.',
                  'Controller time caps are checked between actions; measured elapsed GPU allocation is reported.',
                  'Commercial provenance and dependency/collection-rights review remain pending.']}
    write_json(directory / 'result.json', report)
    progress(directory, 'experiment completed', 1)
    return report
