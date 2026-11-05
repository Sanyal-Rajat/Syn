"""Remote prepare/train/generate workflows and held-out LiDAR comparison."""
import argparse
from functools import partial
import json
from pathlib import Path

import numpy as np

from .calibration import DEFAULT, NAMES, apply, calibration_task
from .config import RunConfig
from .data.download import ensure_dataset
from .data.scan_to_range import unproject
from .data.split_manifest import load_manifest, load_window, prepare, windows
from .eval.domain_metrics import bootstrap, compare, describe
from .io import write_json
from .logs import get_logger, progress as log_progress
from .methods.empirical import sample as empirical_sample
from .provenance import license_gate, runtime_lock, validate_checkpoint

LOGGER = get_logger()


def read_examples(manifest, split, window, per_trace):
    examples, grouped = [], {}
    for identity, records in windows(manifest, split, window):
        grouped.setdefault(identity, []).append(records)
    for identity, records in grouped.items():
        indices = np.linspace(0, len(records) - 1, min(per_trace, len(records)), dtype=int)
        examples.extend(load_window(records[index], manifest['sensor_meta'], identity) for index in indices)
    return examples


def generate(model, count, seed=0, conditioning=None, has_intensity=True, sampler=None):
    from .sampling import SamplingPool

    result = []
    model.eval()
    sampler = sampler or SamplingPool(model)
    requests = [{'batch_size': 1, 'conditioning': conditioning, 'seed': seed + index} for index in range(count)]
    for generated in sampler.sample_many(requests):
        sample = {key: value[0].cpu().numpy() for key, value in generated.items()}
        sample['has_intensity'] = np.asarray([has_intensity] * model.config['window'])
        sample['max_range_m'] = float(model.sensor_meta['max_range_m'])
        result.append(sample)
    return result


def evaluate(real_by_trace, generated_by_trace, max_range):
    rows = []
    for identity, real in real_by_trace.items():
        rows.append({'sequence_id': identity, **compare(describe(real), describe(generated_by_trace[identity]), max_range)})
    return {'traces': rows, 'loss': bootstrap([row['loss'] for row in rows])}


def by_trace(examples):
    result = {}
    for example in examples:
        result.setdefault(example['sequence_id'], []).append(example)
    return result


def run(config, directory):
    report = _run(config, directory)
    from .report import write_pdf
    path = write_pdf(Path(directory) / 'result.json')
    LOGGER.info('PDF results report: %s', path)
    return report


def _run(config, directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if config.action != 'prepare':
        from .gpu import resolve_gpus
        config = config.model_copy(update={'gpus': resolve_gpus(config.gpus)})
    write_json(directory / 'request.json', config.model_dump())
    evidence = license_gate()
    write_json(directory / 'dependency-lock.json', runtime_lock())
    progress = partial(log_progress, directory)
    LOGGER.info('Starting %s; artifacts: %s', config.action, directory)
    if config.action == 'prepare':
        config = config.model_copy(update={'dataset_root': ensure_dataset(config.dataset_root, evidence['dataset'], progress)})
        write_json(directory / 'request.json', config.model_dump())
        progress('inspecting and projecting Pandar128 traces')
        manifest_path = prepare(config, directory / 'prepared', evidence)
        prepared = load_manifest(manifest_path)
        inventory = [{'split': split, 'sequence_id': trace['sequence_id'], 'frames': len(trace['frames']),
                      'source_frames': trace['source_frames'], 'schema': trace['schema']}
                     for split, traces in prepared['splits'].items() for trace in traces]
        report = {'action': 'prepare', 'dataset_root': config.dataset_root, 'manifest': manifest_path, 'inventory': inventory,
                  'sensor_meta': load_manifest(manifest_path)['sensor_meta']}
        write_json(directory / 'result.json', report)
        progress('prepared', 1)
        return report
    if config.action == 'experiment':
        from .experiment import run_experiment
        return run_experiment(config, directory)
    import torch
    from .gpu import hardware

    manifest = load_manifest(config.manifest, verify_files=True)
    license_gate(manifest)
    if config.action == 'train':
        config = config.model_copy(update={'height': len(manifest['sensor_meta']['elevations']),
                                            'width': manifest['sensor_meta']['width'],
                                            'max_range_m': manifest['sensor_meta']['max_range_m']})
        write_json(directory / 'request.json', config.model_dump())
        progress('training random-initialized diffusion model', 0, config.epochs)
        from .gpu import launch_training
        launch_training(config, directory)
        checkpoint_path = directory / 'checkpoint.pt'
    else:
        checkpoint_path = Path(config.checkpoint).expanduser().resolve()
    from .models.lidardm_adapter import LidarDMAdapter

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    validate_checkpoint(checkpoint, config.manifest)
    device = 'cuda:0' if config.gpus and torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(1 if device == 'cpu' else 4)
    model = LidarDMAdapter(checkpoint['config'], checkpoint['sensor_meta']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.config.update(auto_optimize=config.auto_optimize, batch_size=config.batch_size)
    from .sampling import SamplingPool
    sampler = SamplingPool(model, [f'cuda:{index}' for index in range(config.gpus)])
    window = checkpoint['config']['window']
    real_scene, conditioning_source = None, None
    if config.action == 'generate' and (config.conditioning or config.auto_conditioning):
        from .data.conditioning import export_real_scene
        progress('extracting real conditioning scene')
        real_scene, conditioning_source = export_real_scene(manifest, window, directory, config.conditioning)
        config = config.model_copy(update={'conditioning': real_scene})
        write_json(directory / 'request.json', config.model_dump())
    conditioning = None
    if config.conditioning:
        with np.load(config.conditioning, allow_pickle=False) as scene:
            conditioning = {key: torch.as_tensor(scene[key])[None] for key in ('range', 'valid')}
        conditioning['complete_scene'] = True
    if config.action == 'generate':
        progress('generating LiDAR samples')
        if config.sample_steps > model.config['diffusion_steps']:
            raise ValueError('Sampling steps exceed the checkpoint diffusion schedule')
        model.config['sample_steps'] = config.sample_steps
        if config.controller_file:
            from .ot.generate import controlled_generate
            generated = controlled_generate(model, config, sampler)
        else:
            generated = generate(model, config.samples_per_trace, config.seed, conditioning, sampler=sampler)
        if config.profile_file:
            profile = json.loads(Path(config.profile_file).read_text())['profiles'][config.profile_name]
            generated = apply(generated, profile['theta'], config.seed)
        geometry = None
        if conditioning is not None:
            reference = conditioning['range'][0].numpy()
            known = conditioning['valid'][0].numpy().astype(bool)
            tolerance = model.config['geometry_tolerance_m']
            for sample in generated:
                sample['valid'] &= known
                sample['range'] = np.where(sample['valid'], np.clip(sample['range'], reference - tolerance, reference + tolerance), 0)
            differences = [abs(sample['range'][sample['valid']] - reference[sample['valid']]) for sample in generated]
            geometry = {'new_surface_returns': 0,
                        'max_surface_deviation_m': max((float(value.max()) for value in differences if len(value)), default=0.)}
        for index, sample in enumerate(generated):
            np.savez_compressed(directory / f'sample_{index:03d}.npz', **sample,
                                elevations=np.asarray(manifest['sensor_meta']['elevations']))
        report = {'action': 'generate', 'samples': len(generated), 'checkpoint': str(checkpoint_path),
                  'conditioning': config.conditioning or None, 'pretrained_weights_used': False}
        report['real_scene'] = real_scene
        report['conditioning_source'] = conditioning_source
        report['geometry'] = geometry
        report['calibration_profile'] = config.profile_name if config.profile_file else 'uncalibrated'
        report['generation_controller'] = config.controller_file or None
        write_json(directory / 'result.json', report)
        progress('completed', 1)
        return report
    progress('generating frozen calibration and validation samples')
    training = read_examples(manifest, 'train', window, config.samples_per_trace)
    validation = read_examples(manifest, 'validation', window, config.samples_per_trace)
    has_intensity = all(bool(np.asarray(row['has_intensity']).all()) for row in training)
    calibration_generated = generate(model, len(training), config.seed + 10_000, has_intensity=has_intensity, sampler=sampler)
    validation_generated = generate(model, len(validation), config.seed + 20_000, has_intensity=has_intensity, sampler=sampler)
    profiles = {'uncalibrated': {'theta': DEFAULT.tolist(),
                'validation': compare(describe(validation), describe(validation_generated), manifest['sensor_meta']['max_range_m'])}}
    if config.rl_enabled:
        from .scheduling import map_tasks
        model.cpu()
        sampler.close()
        jobs = [{'method': method, 'real': training, 'generated': calibration_generated,
                 'validation_real': validation, 'validation_generated': validation_generated,
                 'max_range': manifest['sensor_meta']['max_range_m'], 'budget': config.budget,
                 'seed': config.seed, 'gpus': config.gpus, 'rl': config.rl.model_dump(),
                 'output': str(directory)} for method in config.methods]
        profiles.update(map_tasks(calibration_task, jobs, config.gpus,
            on_done=lambda done, total: progress('independent sensor calibration methods', done, total)))
        model.to(device)
        sampler = SamplingPool(model, [f'cuda:{index}' for index in range(config.gpus)])
    write_json(directory / 'profiles.json', {'parameter_names': list(NAMES), 'profiles': profiles})
    selected_profile = min(profiles, key=lambda name: profiles[name]['validation']['loss'])
    # Final real test arrays are loaded only after training and calibration are frozen.
    progress('held-out evaluation')
    test = read_examples(manifest, 'test', window, config.samples_per_trace)
    real = by_trace(test)
    generator_results = {}
    for identity, examples in real.items():
        # Same fixed synthetic seed bank for every test trace and every calibration arm.
        generator_results[identity] = generate(model, len(examples), config.seed + 30_000,
                                                has_intensity=has_intensity, sampler=sampler)
    maximum = manifest['sensor_meta']['max_range_m']
    evaluations = {}
    rng = np.random.default_rng(config.seed)
    resampled = {identity: [training[int(rng.integers(len(training)))] for _ in examples]
                 for identity, examples in real.items()}
    empirical = {identity: empirical_sample(training, len(examples), config.seed) for identity, examples in real.items()}
    evaluations['training_resampling'] = evaluate(real, resampled, maximum)
    evaluations['per_ring_empirical'] = evaluate(real, empirical, maximum)
    for name, profile in profiles.items():
        generated = {identity: apply(examples, profile['theta'], config.seed) for identity, examples in generator_results.items()}
        evaluations[name] = evaluate(real, generated, maximum)
    preview = apply([next(iter(generator_results.values()))[0]], profiles[selected_profile]['theta'], config.seed)[0]
    np.savez_compressed(directory / 'sample_000.npz', **preview,
                        elevations=np.asarray(manifest['sensor_meta']['elevations']))
    points = unproject(preview['range'][0], preview['valid'][0], manifest['sensor_meta']['elevations'])
    report = {'action': 'train', 'checkpoint': str(checkpoint_path), 'evaluations': evaluations,
              'profiles': profiles, 'selected_profile_on_validation': selected_profile,
              'training': json.loads((directory / 'training.json').read_text()),
              'resources': hardware(), 'preview': {'range': preview['range'][0].tolist(),
                                                   'points': points[::max(1, len(points) // 3000)].tolist()},
              'license_evidence': evidence, 'geometry_evaluated': False,
              'status': 'held_out_sensor_statistics_only', 'pretrained_weights_used': False,
              'limitations': ['Range-image adaptation of LidarDM; no SDF/VAE world-generation claim.',
                             'RL calibrates a postprocessor; it does not replace diffusion training.',
                             'Unconditional fidelity does not establish scene geometry or downstream perception transfer.',
                             'Dependency/collection-rights audit and cross-sensor generalization remain outside this PoC.']}
    model_losses = {row['sequence_id']: row['loss'] for row in evaluations[selected_profile]['traces']}
    report['fidelity_lift_vs_baselines'] = {
        name: bootstrap([row['loss'] - model_losses[row['sequence_id']] for row in evaluations[name]['traces']])
        for name in ('training_resampling', 'per_ring_empirical')}
    write_json(directory / 'result.json', report)
    progress('completed', 1)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    run(RunConfig.model_validate_json(arguments.request.read_text()), arguments.output)
