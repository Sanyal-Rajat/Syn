"""Independent development, controller, probe and frozen-test jobs."""
from pathlib import Path
import time

import numpy as np
import torch

from .config import RunConfig
from .data.split_manifest import digest, load_manifest
from .data.synthetic_scene import load_scene
from .eval import downstream
from .eval.ot_metrics import evaluate
from .experiment import arm_config, cpu_policy, load_model, save_bank, train_arm
from .gpu import resolve_gpus
from .io import write_json
from .logs import get_logger
from .ot.controllers import Renderer, controlled_bank, fit_controller
from .ot.reference import load_reference
from .sampling import SamplingPool

_CACHE = {}


def load_bank(path):
    result = []
    for file in sorted(Path(path).glob('sample_*.npz')):
        with np.load(file, allow_pickle=False) as archive:
            result.append({key: archive[key].copy() for key in archive.files})
    return result


def context(task):
    config = RunConfig(**task['config'])
    count = min(config.gpus, resolve_gpus(-1)) if config.gpus > 0 else resolve_gpus(config.gpus)
    config = config.model_copy(update={'gpus': count})
    root = Path(task['root'])
    key = (str(root), config.manifest)
    if key not in _CACHE:
        manifest = load_manifest(config.manifest)
        sensor = manifest['sensor_meta']
        scenes = {split: [load_scene(path, sensor, config.window) for path in sorted((root / 'scenes' / split).glob('*.npz'))]
                  for split in ('train', 'validation', 'test')}
        reference, _ = load_reference(root / 'ot_reference.npz', config.manifest, config.ot)
        _CACHE[key] = (manifest, scenes, reference)
    torch.set_num_threads(1)
    return config, root, *_CACHE[key]


def examples(config, root, manifest, split):
    from .runner import read_examples

    key = (str(root), config.manifest, split, config.window, config.samples_per_trace)
    if key not in _CACHE:
        _CACHE[key] = read_examples(manifest, split, config.window, config.samples_per_trace)
    return _CACHE[key]


def development(task):
    config, root, manifest, scenes, reference = context(task)
    real = examples(config, root, manifest, 'validation')
    seed, name = task['seed'], task['name']
    folder = root / 'experiments' / f'seed_{seed}' / name
    folder.mkdir(parents=True, exist_ok=True)
    sensor, settings = manifest['sensor_meta'], config.experiment
    count = max(1, config.samples_per_trace * len(manifest['splits']['validation']))
    device = 'cuda:0' if config.gpus else 'cpu'
    get_logger().info('Development job: seed=%d, arm=%s, GPUs=%d', seed, name, config.gpus)
    if name.startswith('E5_'):
        checkpoint = folder.parent / 'E4' / 'checkpoint.pt'
        costs = {'shared_training_arm': 'E4', 'frozen_generator': True}
    else:
        current = arm_config(config, name, str(root / 'ot_reference.npz'), sensor, seed)
        costs = train_arm(current, folder, manifest, name)
        checkpoint = folder / 'checkpoint.pt'
    model = load_model(checkpoint, manifest, device)
    model.config.update(sample_steps=config.sample_steps, auto_optimize=config.auto_optimize, batch_size=config.batch_size)
    sampler = SamplingPool(model, [f'cuda:{index}' for index in range(config.gpus)])
    record = {'checkpoint': str(checkpoint), 'profile': None, 'policy': None}
    profile = policy = None
    try:
        if name.startswith('E5_'):
            renderer = Renderer(model, scenes['train'], sampler)
            renderer.reference, renderer.ot = reference, config.ot
            profile, policy = fit_controller(name[3:], renderer, scenes['train'], reference, sensor,
                config.ot, settings, config.budget, seed, device, config.rl)
            write_json(folder / 'controller.json', profile)
            bundle = {'profile': profile, 'policy': cpu_policy(policy), 'checkpoint_sha256': digest(checkpoint),
                'manifest_sha256': digest(config.manifest), 'algorithm': name[3:],
                'reference_sha256': digest(root / 'ot_reference.npz'), 'root_initialization': 'random',
                'ot_config': config.ot.model_dump(), 'reference': {key: torch.as_tensor(value) for key, value in reference.items()}}
            torch.save(bundle, folder / 'controller.pt')
            record.update(profile=profile, controller=str(folder / 'controller.pt'))
            costs.update({key: profile[key] for key in ('generation_calls', 'call_budget', 'elapsed_s',
                                                       'seconds_budget', 'denoiser_evaluations')})
            costs.update({key: profile[key] for key in ('learning_updates', 'training_transitions') if key in profile})
            costs['allocated_controller_gpu_hours'] = profile['elapsed_s'] * config.gpus / 3600
        started = time.perf_counter()
        renderer = Renderer(model, scenes['validation'], sampler)
        renderer.reference, renderer.ot = reference, config.ot
        bank, canonical = controlled_bank(renderer, scenes['validation'], count, seed + 100000, profile, policy)
        validation = evaluate(real, bank, canonical, sensor, config.ot, config.geometry_tolerance_m, device=device)
        costs.update(development_seconds=time.perf_counter() - started, inference_gpus=config.gpus,
                     evaluation_calls_per_phase=count, evaluation_denoiser_evaluations_per_phase=2 * config.sample_steps * count)
        save_bank(folder / 'development_samples', bank)
        if settings.downstream:
            started = time.perf_counter()
            renderer = Renderer(model, scenes['train'], sampler)
            renderer.reference, renderer.ot = reference, config.ot
            bank, _ = controlled_bank(renderer, scenes['train'], count, seed + 300000, profile, policy)
            save_bank(folder / 'training_samples', bank)
            costs['synthetic_training_bank_seconds'] = time.perf_counter() - started
        return {'seed': seed, 'name': name, 'record': record, 'validation': validation, 'cost': costs}
    finally:
        model.cpu()
        sampler.close()


def probe(task):
    config, root, manifest, scenes, reference = context(task)
    training = examples(config, root, manifest, 'train')
    real = examples(config, root, manifest, 'validation')
    seed, name, budget = task['seed'], task['name'], task['budget']
    folder = root / 'experiments' / f'seed_{seed}' / name
    started, cpu_started = time.perf_counter(), time.process_time()
    device = 'cuda:0' if config.gpus else 'cpu'
    bank = load_bank(folder / 'training_samples')
    model, report = downstream.fit(training, bank, real, manifest['sensor_meta'], task['real_count'],
        task['synthetic_count'], config.experiment.downstream_steps, seed, device=device)
    path = folder / f'probe_{budget}.pt'
    torch.save({'state_dict': model.cpu().state_dict(), 'report': report}, path)
    return {'seed': seed, 'name': name, 'budget': str(budget), 'report': report, 'path': str(path),
            'seconds': time.perf_counter() - started, 'cpu_seconds': time.process_time() - cpu_started,
            'gpu_hours': (time.perf_counter() - started) * min(1, config.gpus) / 3600}


def test(task):
    config, root, manifest, scenes, reference = context(task)
    if not (root / 'selection.json').is_file():
        raise ValueError('Test jobs require frozen global selection')
    seed, name, record = task['seed'], task['name'], task['record']
    device = 'cuda:0' if config.gpus else 'cpu'
    real = examples(config, root, manifest, 'test')
    count = max(1, config.samples_per_trace * len(manifest['splits']['validation']))
    model = load_model(record['checkpoint'], manifest, device)
    model.config.update(sample_steps=config.sample_steps, auto_optimize=config.auto_optimize, batch_size=config.batch_size)
    sampler = SamplingPool(model, [f'cuda:{index}' for index in range(config.gpus)])
    profile, policy = record['profile'], None
    if record.get('controller'):
        bundle = torch.load(record['controller'], map_location='cpu', weights_only=True)
        profile, policy = bundle['profile'], bundle['policy']
    started = time.perf_counter()
    try:
        renderer = Renderer(model, scenes['test'], sampler)
        renderer.reference, renderer.ot = reference, config.ot
        bank, canonical = controlled_bank(renderer, scenes['test'], count, seed + 200000, profile, policy)
        metrics = evaluate(real, bank, canonical, manifest['sensor_meta'], config.ot, config.geometry_tolerance_m, device=device)
        save_bank(root / 'experiments' / f'seed_{seed}' / name / 'test_samples', bank)
        task_rows = {}
        for budget, path in task.get('probes', {}).items():
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            classes = checkpoint['report']['classes']
            classifier = downstream.make_model(classes, device)
            classifier.load_state_dict(checkpoint['state_dict'])
            classifier.eval()
            task_rows[budget] = downstream.metric(classifier, real, manifest['sensor_meta'], classes)
        return {'seed': seed, 'name': name, 'metrics': metrics, 'task': task_rows,
                'seconds': time.perf_counter() - started, 'gpus': config.gpus}
    finally:
        model.cpu()
        sampler.close()
