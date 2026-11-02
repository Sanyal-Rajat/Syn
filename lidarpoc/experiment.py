"""Additive E0–E5 experiment runner; freeze all choices before decoding real test arrays."""
from collections import defaultdict
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from .data.split_manifest import digest, load_manifest
from .data.synthetic_scene import geometry_digest, scene_splits
from .eval import downstream
from .eval.gates import decisions
from .eval.ot_metrics import evaluate, session_intervals
from .eval.domain_metrics import bootstrap
from .io import write_json
from .logs import progress
from .models.lidardm_adapter import LidarDMAdapter
from .ot.controllers import Renderer, controlled_bank, fit_controller
from .ot.reference import build_reference, load_reference
from .provenance import license_gate, provenance, validate_checkpoint


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
        command = [sys.executable, '-u', '-m', 'lidarpoc.train']
        if config.gpus > 1:
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={config.gpus}',
                       '-m', 'lidarpoc.train']
        subprocess.run(command + ['--request', str(request), '--output', str(directory)], check=True,
                       env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    elapsed = time.perf_counter() - started
    return {'training_seconds': elapsed, 'allocated_training_gpu_hours': elapsed * config.gpus / 3600 if arm != 'E0' else 0.,
            'epochs': 0 if arm == 'E0' else config.epochs, 'seed': config.seed}


def run_experiment(config, directory):
    from .runner import read_examples

    directory = Path(directory).resolve()
    manifest = load_manifest(config.manifest, verify_files=True)
    license_gate(manifest)
    if config.gpus and (not torch.cuda.is_available() or config.gpus > torch.cuda.device_count()):
        raise ValueError('Requested experiment GPUs are unavailable; choose gpus=0 for CPU')
    device = 'cuda:0' if config.gpus else 'cpu'
    torch.set_num_threads(4 if config.gpus else 1)
    sensor, settings = manifest['sensor_meta'], config.experiment
    if config.checkpoint:
        validate_checkpoint(torch.load(config.checkpoint, map_location='cpu', weights_only=True), config.manifest)
    progress(directory, 'freezing training-only OT reference and canonical scene splits')
    training = read_examples(manifest, 'train', config.window, config.samples_per_trace)
    development = read_examples(manifest, 'validation', config.window, config.samples_per_trace)
    reference_path = build_reference(training, config.manifest, config.ot, directory / 'ot_reference.npz')
    reference, reference_metadata = load_reference(reference_path, config.manifest, config.ot)
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
            'strata_scope': 'range and ring proxies; not matched road type/material/incidence labels'}
    write_json(directory / 'experiment_plan.json', plan)
    seeds, validations, computations, task_models, task_development = {}, {}, {}, {}, {}
    count = max(1, config.samples_per_trace * len(manifest['splits']['validation']))
    for seed in settings.seeds:
        seed_directory = directory / 'experiments' / f'seed_{seed}'
        records, banks, validation_rows, costs = {}, {}, {}, {}
        seed_directory.mkdir(parents=True, exist_ok=True)
        for arm in [value for value in settings.ablations if value != 'E5']:
            progress(directory, f'seed {seed}: {arm} diffusion ablation')
            arm_directory = seed_directory / arm
            current = arm_config(config, arm, reference_path, sensor, seed)
            costs[arm] = train_arm(current, arm_directory, manifest, arm)
            checkpoint = arm_directory / 'checkpoint.pt'
            model = load_model(checkpoint, manifest, device)
            model.config['sample_steps'] = config.sample_steps
            renderer = Renderer(model, scenes['validation'])
            renderer.reference, renderer.ot = reference, config.ot
            started = time.perf_counter()
            bank, canonical = controlled_bank(renderer, scenes['validation'], count, seed + 100000)
            validation_rows[arm] = evaluate(development, bank, canonical, sensor, config.ot, config.geometry_tolerance_m)
            costs[arm]['development_seconds'] = time.perf_counter() - started
            records[arm] = {'checkpoint': str(checkpoint), 'profile': None, 'policy': None}
            save_bank(arm_directory / 'development_samples', bank)
            if settings.downstream:
                started = time.perf_counter()
                training_renderer = Renderer(model, scenes['train'])
                training_renderer.reference, training_renderer.ot = reference, config.ot
                banks[arm], _ = controlled_bank(training_renderer, scenes['train'], count, seed + 300000)
                costs[arm]['synthetic_training_bank_seconds'] = time.perf_counter() - started
            if arm == 'E4' and 'E5' in settings.ablations:
                for method in settings.controllers:
                    if not config.rl_enabled and method in ('sac', 'ppo', 'contextual_bandit'):
                        continue
                    progress(directory, f'seed {seed}: frozen E4 + {method} controller')
                    train_renderer = Renderer(model, scenes['train'])
                    train_renderer.reference, train_renderer.ot = reference, config.ot
                    profile, policy = fit_controller(method, train_renderer, scenes['train'], reference, sensor,
                        config.ot, settings, config.budget, seed, device)
                    name = f'E5_{method}'
                    folder = seed_directory / name
                    folder.mkdir(parents=True, exist_ok=True)
                    write_json(folder / 'controller.json', profile)
                    bundle = {'profile': profile, 'policy': cpu_policy(policy), 'checkpoint_sha256': digest(checkpoint),
                              'manifest_sha256': digest(config.manifest), 'algorithm': method, 'reference_sha256': digest(reference_path),
                              'root_initialization': 'random', 'ot_config': config.ot.model_dump()}
                    bundle['reference'] = {key: torch.as_tensor(value) for key, value in reference.items()}
                    torch.save(bundle, folder / 'controller.pt')
                    started = time.perf_counter()
                    bank, canonical = controlled_bank(renderer, scenes['validation'], count, seed + 100000, profile, policy)
                    validation_rows[name] = evaluate(development, bank, canonical, sensor, config.ot, config.geometry_tolerance_m)
                    records[name] = {'checkpoint': str(checkpoint), 'profile': profile, 'policy': policy,
                                     'controller': str(folder / 'controller.pt')}
                    costs[name] = {**{key: profile[key] for key in ('generation_calls', 'call_budget', 'elapsed_s', 'seconds_budget')},
                                   'denoiser_evaluations': profile['denoiser_evaluations'],
                                   'allocated_controller_gpu_hours': profile['elapsed_s'] * min(config.gpus, 1) / 3600,
                                   'shared_training_arm': 'E4',
                                   'development_seconds': time.perf_counter() - started,
                                   'frozen_generator': True}
                    save_bank(folder / 'development_samples', bank)
                    if settings.downstream:
                        started = time.perf_counter()
                        banks[name], _ = controlled_bank(train_renderer, scenes['train'], count, seed + 300000, profile, policy)
                        costs[name]['synthetic_training_bank_seconds'] = time.perf_counter() - started
            model.cpu()
            del renderer, model
        for name in records:
            costs[name]['evaluation_calls_per_phase'] = count
            costs[name]['evaluation_denoiser_evaluations_per_phase'] = 2 * config.sample_steps * count
        if settings.downstream:
            progress(directory, f'seed {seed}: identical-budget point-segmentation probes')
            available_real = len(downstream.labeled_points(training, sensor)[1])
            common_synthetic = min(len(downstream.labeled_points(bank, sensor)[1]) for bank in banks.values())
            fitted, rows = {}, {}
            for name, bank in banks.items():
                started = time.perf_counter()
                fitted[name], rows[name] = {}, {}
                for budget in settings.label_budgets:
                    real_count, synthetic_count = min(budget, available_real), min(budget, common_synthetic)
                    probe, probe_report = downstream.fit(training, bank, development, sensor, real_count,
                                                         synthetic_count, settings.downstream_steps, seed)
                    fitted[name][str(budget)] = (probe, probe_report['classes'])
                    rows[name][str(budget)] = probe_report
                    torch.save({'state_dict': probe.state_dict(), 'report': probe_report},
                               seed_directory / name / f'probe_{budget}.pt')
                costs[name]['downstream_cpu_seconds'] = time.perf_counter() - started
            task_models[seed], task_development[seed] = fitted, rows
        seeds[seed], validations[seed], computations[seed] = records, validation_rows, costs
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
    test = read_examples(manifest, 'test', config.window, config.samples_per_trace)
    tests, task_tests = {}, []
    for seed, records in seeds.items():
        rows, task_rows = {}, {}
        for name, record in records.items():
            model = load_model(record['checkpoint'], manifest, device)
            model.config['sample_steps'] = config.sample_steps
            renderer = Renderer(model, scenes['test'])
            renderer.reference, renderer.ot = reference, config.ot
            started = time.perf_counter()
            bank, canonical = controlled_bank(renderer, scenes['test'], count, seed + 200000,
                                               record['profile'], record['policy'])
            rows[name] = evaluate(test, bank, canonical, sensor, config.ot, config.geometry_tolerance_m)
            cost = computations[seed][name]
            cost['test_seconds'] = time.perf_counter() - started
            training_cost = computations[seed][cost.get('shared_training_arm', name)]['allocated_training_gpu_hours']
            inference_seconds = sum(cost.get(key, 0.) for key in (
                'development_seconds', 'test_seconds', 'synthetic_training_bank_seconds'))
            cost['allocated_total_gpu_hours'] = training_cost + cost.get('allocated_controller_gpu_hours', 0.) + inference_seconds * min(config.gpus, 1) / 3600
            gain = selection_score('E0') - selection_score(name) if 'E0' in candidates else None
            cost['development_gain_over_E0'] = gain
            cost['gpu_hours_per_positive_development_gain'] = cost['allocated_total_gpu_hours'] / gain if gain is not None and gain > 0 else None
            cost['total_generation_calls'] = cost.get('generation_calls', 0) + count * (3 if settings.downstream else 2)
            cost['calls_per_positive_development_gain'] = cost['total_generation_calls'] / gain if gain is not None and gain > 0 else None
            save_bank(directory / 'experiments' / f'seed_{seed}' / name / 'test_samples', bank)
            if settings.downstream:
                task_rows[name] = {budget: downstream.metric(probe, test, sensor, classes)
                                   for budget, (probe, classes) in task_models[seed][name].items()}
            model.cpu()
            del renderer, model
        tests[seed] = rows
        task_tests.append(task_rows)
    gates = decisions([rows[selected] for rows in tests.values()], task_tests, selected, settings)
    if 'E5_sac' in candidates and rl_comparator:
        baseline = rl_comparator
        improvement = selection_score(baseline) - selection_score('E5_sac')
        by_session = defaultdict(list)
        if settings.downstream:
            for rows in task_tests:
                for budget, values in rows['E5_sac'].items():
                    base = {row['sequence_id']: row['mean_iou'] for row in rows[baseline][budget]}
                    for row in values:
                        by_session[row['sequence_id']].append(row['mean_iou'] - base[row['sequence_id']])
        else:
            for rows in tests.values():
                base = {row['sequence_id']: row['gap'] for row in rows[baseline]['traces']}
                for row in rows['E5_sac']['traces']:
                    by_session[row['sequence_id']].append(base[row['sequence_id']] - row['gap'])
        paired = bootstrap([float(np.mean(values)) for values in by_session.values()])
        metered = [cost for rows in computations.values() for name, cost in rows.items() if name.startswith('E5_')]
        matched = all(row['generation_calls'] == config.budget and row['elapsed_s'] <= settings.controller_seconds for row in metered)
        material = improvement > settings.rl_min_improvement and paired['ci95'][0] > settings.rl_min_improvement and matched
        gates['rl_development_comparison'] = {'comparator': baseline, 'primary_metric_improvement': improvement,
                                             'material': material, 'paired_test_session_gain': paired,
                                             'matched_call_and_time_caps': matched,
                                             'recommendation': 'retain_as_experimental' if material else 'no_incremental_value_demonstrated',
                                             'methods_removed': False}
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
