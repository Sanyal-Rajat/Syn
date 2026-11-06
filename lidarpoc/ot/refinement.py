"""Bounded output optimization on each sampling GPU; model weights stay frozen."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import torch

from .config import OTConfig
from .features import conditional_ot, features, mass_ot, return_mass, select
from .reference import load_reference
from ..data.split_manifest import digest
from ..logs import get_logger


def _objective(sample, support, reference, sensor, config):
    vectors, bins, _ = features(sample, sensor, config)
    source = select(vectors, bins, support.reshape(-1).float(), config.max_points)
    target = {int(key): reference['features'][reference['bins'] == key]
              for key in reference['bins'].unique().tolist()}
    try:
        conditional, unsupported = conditional_ot(source, target, config)
    except ValueError as error:
        if not str(error).startswith('No supported conditional OT bins'):
            raise
        conditional = sample['range'].sum() * 0 + 1.
        unsupported = sorted(set(source) | set(target))
    centers, mass = return_mass(sample, sensor, config)
    unbalanced = mass_ot(mass, reference['mass'], centers, config)
    return conditional + unbalanced + len(unsupported) / max(1, len(target))


def refine(model, ranges, logits, intensity, conditioning=None, initial_valid=None):
    """Return the best hard-export objective, including the original output candidate."""
    config = OTConfig(**model.config.get('ot', {}))
    if not config.generation_enabled:
        valid = initial_valid.clone() if initial_valid is not None else (logits > 0) & (ranges >= .1)
        if conditioning is not None and conditioning.get('complete_scene', True):
            valid &= conditioning['valid'].to(valid.device).bool()
        return {'range': torch.where(valid, ranges, 0).float(), 'valid': valid,
                'intensity': torch.where(valid, intensity, 0).float()}
    if not config.reference:
        raise ValueError('Generation alignment requires a training-only reference; prepare it before sampling')
    # SamplingPool uses inference_mode: explicitly leave it before creating optimization tensors.
    with torch.inference_mode(False), torch.enable_grad(), torch.autocast(ranges.device.type, enabled=False):
        cache = getattr(model, '_generation_reference', None)
        if cache is None or cache[0] != config.reference:
            data, metadata = load_reference(config.reference, model.config['manifest'], config)
            cache = model._generation_reference = (config.reference, data, metadata)
        reference = {key: torch.as_tensor(value, device=ranges.device) for key, value in cache[1].items()}
        maximum = float(model.sensor_meta['max_range_m'])
        results = []
        for index in range(len(ranges)):
            original = ranges[index:index + 1].detach().clone().float()
            allowed = torch.isfinite(original) & (original >= .1) & (original <= maximum)
            lower, upper = torch.full_like(original, .1), torch.full_like(original, maximum)
            if conditioning is not None:
                known = conditioning['valid'][index:index + 1].to(original.device).clone().bool()
                geometry = conditioning['range'][index:index + 1].to(original.device).clone().float()
                tolerance = model.config['geometry_tolerance_m']
                lower = torch.where(known, (geometry - tolerance).clamp(.1, maximum), lower)
                upper = torch.where(known, (geometry + tolerance).clamp(.1, maximum), upper)
                if conditioning.get('complete_scene', True):
                    allowed &= known
            normalized = torch.nn.Parameter(original.maximum(lower).minimum(upper) / maximum)
            initial_logits = logits[index:index + 1].detach().clone().float()
            if initial_valid is not None:
                valid = initial_valid[index:index + 1].to(original.device).clone().bool()
                initial_logits = torch.where(valid, initial_logits, initial_logits.clamp_max(-.25))
            probabilities = torch.nn.Parameter(initial_logits)
            reflectance = torch.nn.Parameter(intensity[index:index + 1].detach().clone().float())
            parameters = [normalized, probabilities, reflectance]
            optimizer = torch.optim.Adam(parameters, lr=config.generation_lr)

            def output(hard):
                valid = (probabilities > 0) & allowed if hard else probabilities.sigmoid() * allowed
                depth = (normalized * maximum).maximum(lower).minimum(upper)
                return {'range': torch.where(valid, depth, 0) if hard else depth,
                        'valid': valid,
                        'intensity': torch.where(valid, reflectance, 0) if hard else reflectance}

            with torch.no_grad():
                best = output(True)
                initial = best_value = float(_objective(best, best['valid'], reference, model.sensor_meta, config))
            if not np.isfinite(initial):
                raise ValueError('Initial generation alignment objective is nonfinite')
            get_logger().info('Starting OT refinement on %s, window %d/%d: objective=%.6f',
                              original.device, index + 1, len(ranges), initial)
            attempted = accepted = 0
            for step in range(config.generation_steps):
                attempted = step + 1
                optimizer.zero_grad(set_to_none=True)
                loss = _objective(output(False), allowed, reference, model.sensor_meta, config)
                if not torch.isfinite(loss):
                    get_logger().warning('Nonfinite refinement loss; retaining the best finite output')
                    break
                loss.backward()
                if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                       for parameter in parameters):
                    get_logger().warning('Nonfinite refinement gradient; retaining the best finite output')
                    break
                torch.nn.utils.clip_grad_norm_(parameters, 1.)
                optimizer.step()
                with torch.no_grad():
                    normalized.copy_(normalized.maximum(lower / maximum).minimum(upper / maximum))
                    probabilities.clamp_(-12, 12)
                    reflectance.clamp_(0, 1)
                    candidate = output(True)
                    value = float(_objective(candidate, candidate['valid'], reference, model.sensor_meta, config))
                    if np.isfinite(value) and value < best_value - 1e-7:
                        best, best_value = candidate, value
                        accepted += 1
                if attempted % max(1, config.generation_steps // 4) == 0:
                    get_logger().info('OT refinement on %s: step %d/%d, best objective=%.6f',
                                      original.device, attempted, config.generation_steps, best_value)
            best.update(ot_initial_objective=original.new_tensor([initial]),
                        ot_final_objective=original.new_tensor([best_value]),
                        ot_refinement_steps=original.new_tensor([attempted], dtype=torch.int32),
                        ot_accepted_updates=original.new_tensor([accepted], dtype=torch.int32),
                        has_intensity=torch.full(original.shape[:2], cache[2].get('intensity_observed',
                            bool(config.feature_weights[1])), dtype=torch.bool, device=original.device))
            results.append(best)
            get_logger().info('OT refinement on %s, window %d/%d: %.6f -> %.6f; %d steps, %d accepted',
                              original.device, index + 1, len(ranges), initial, best_value, attempted, accepted)
        return {key: torch.cat([result[key].detach() for result in results]) for key in results[0]}


def refine_samples(samples, model, config, conditioning, devices):
    """Re-align final arrays after an optional calibration profile/controller changes them."""
    devices = list(devices) or ['cpu']
    workers = min(len(devices), len(samples))
    if not workers:
        return samples

    def worker(rank):
        device = torch.device(devices[rank])
        context = torch.cuda.device(device) if device.type == 'cuda' else nullcontext()
        proxy = SimpleNamespace(config={**model.config, 'ot': config.model_dump()}, sensor_meta=model.sensor_meta)
        rows = []
        with context:
            for index in range(rank, len(samples), workers):
                sample = samples[index]
                ranges = torch.as_tensor(sample['range'], device=device)[None]
                valid = torch.as_tensor(sample['valid'], device=device)[None].bool()
                intensity = torch.as_tensor(sample['intensity'], device=device)[None]
                if conditioning is not None:
                    known = conditioning['valid'].to(device).bool()
                    ranges = torch.where(~valid & known, conditioning['range'].to(device), ranges)
                refined = refine(proxy, ranges, valid.float() * .5 - .25, intensity, conditioning)
                rows.append((index, {**sample, **{key: value[0].cpu().numpy() for key, value in refined.items()}}))
        return rows

    if workers == 1:
        rows = worker(0)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = [row for group in executor.map(worker, range(workers)) for row in group]
    return [sample for _, sample in sorted(rows)]


def describe_refinement(samples, config, final_stage=False):
    """Persist the actual measured improvement, including unchanged outputs."""
    rows = []
    for index, sample in enumerate(samples):
        if 'ot_initial_objective' in sample:
            rows.append({'sample': index, 'initial_objective': float(sample['ot_initial_objective']),
                         'final_objective': float(sample['ot_final_objective']),
                         'steps': int(sample['ot_refinement_steps']), 'accepted_updates': int(sample['ot_accepted_updates'])})
    return {'enabled': bool(rows), 'reference': config.reference or None,
            'reference_sha256': digest(config.reference) if rows else None, 'reference_split': 'train',
            'samples': rows, 'objective': 'conditional Sinkhorn + unbalanced KL-OT + unsupported-bin penalty',
            'scope': ('Final output after calibration/controller changes.' if final_stage else 'Generated sensor output.') +
                     ' Geometry-constrained LiDAR refinement; not an RGB image loss.',
            'selection': 'Best exported hard-mask objective including the original sample; other metrics may differ.'}
