"""Bounded output optimization on each sampling GPU; model weights stay frozen."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import torch

from .config import OTConfig
from .features import conditional_ot, features, mass_ot, return_mass, select
from .reference import load_reference
from .range_distance import finite_number, frame_distances
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
    """Retain non-worsening frame W1 on actual exported masks, starting with the original."""
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
            range_target = None
            conditioned_target = conditioning is not None and bool(known.any())
            if config.generation_range_weight:
                if conditioned_target:
                    range_target = {'range': geometry, 'valid': known}
                else:
                    if 'range_values' not in reference:
                        raise ValueError('Rebuild the training reference to include range-distance support')
                    range_target = {'range': reference['range_values'],
                                    'valid': torch.ones_like(reference['range_values'], dtype=torch.bool)}
                target_frames = (known.flatten(2).any(-1).reshape(-1) if conditioned_target else
                                 torch.ones(original.shape[1], dtype=torch.bool, device=original.device))
            normalized = torch.nn.Parameter(original.maximum(lower).minimum(upper) / maximum)
            initial_logits = logits[index:index + 1].detach().clone().float()
            if initial_valid is not None:
                valid = initial_valid[index:index + 1].to(original.device).clone().bool()
                initial_logits = torch.where(valid, initial_logits, initial_logits.clamp_max(-.25))
            if range_target is not None:
                # Preserve the initial hard mask while allowing the bounded optimizer to change returns.
                initial_logits.clamp_(-.25, .25)
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

            def distances(sample, soft=False):
                values = frame_distances(sample, range_target, soft=soft)
                return torch.stack([value if value is not None else original.new_tensor(float('nan')).double()
                                    for value in values])

            with torch.no_grad():
                best = output(True)
                initial = best_value = float(_objective(best, best['valid'], reference, model.sensor_meta, config))
                initial_counts = best['valid'].flatten(2).sum(-1)
                initial_distances = best_distances = distances(best) if range_target is not None else None
            if not np.isfinite(initial):
                raise ValueError('Initial generation alignment objective is nonfinite')
            get_logger().info('Starting OT refinement on %s, window %d/%d: objective=%.6f',
                              original.device, index + 1, len(ranges), initial)
            attempted = accepted = 0
            for step in range(config.generation_steps):
                attempted = step + 1
                optimizer.zero_grad(set_to_none=True)
                soft = output(False)
                loss = _objective(soft, allowed, reference, model.sensor_meta, config)
                if range_target is not None:
                    gap = distances(soft, soft=True)
                    if torch.isfinite(gap).any():
                        loss = loss + config.generation_range_weight * gap[torch.isfinite(gap)].mean().float()
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
                    candidate_distances = distances(candidate) if range_target is not None else None
                    better = value < best_value - 1e-7
                    if range_target is not None:
                        proposed, retained = candidate_distances[target_frames], best_distances[target_frames]
                        finite = bool(torch.isfinite(proposed).all())
                        comparable = bool(torch.isfinite(retained).all())
                        closer = finite and (not comparable or (
                            bool((proposed <= retained + 1e-9).all()) and
                            (float(proposed.mean()) < float(retained.mean()) - 1e-9 or better)))
                        coverage = (not conditioned_target or
                                    bool((candidate['valid'].flatten(2).sum(-1) >= initial_counts).all()))
                        better = closer and coverage
                    if np.isfinite(value) and better:
                        best, best_value = candidate, value
                        best_distances = candidate_distances
                        accepted += 1
                if attempted % max(1, config.generation_steps // 4) == 0:
                    get_logger().info('OT refinement on %s: step %d/%d, retained secondary objective=%.6f',
                                      original.device, attempted, config.generation_steps, best_value)
            best.update(ot_initial_objective=original.new_tensor([initial]),
                        ot_final_objective=original.new_tensor([best_value]),
                        ot_refinement_steps=original.new_tensor([attempted], dtype=torch.int32),
                        ot_accepted_updates=original.new_tensor([accepted], dtype=torch.int32),
                        has_intensity=torch.full(original.shape[:2], cache[2].get('intensity_observed',
                            bool(config.feature_weights[1])), dtype=torch.bool, device=original.device))
            if range_target is not None:
                best.update(ot_range_initial_m=initial_distances[None], ot_range_final_m=best_distances[None],
                            ot_range_conditioned=torch.tensor([conditioned_target], device=original.device))
                get_logger().info('Range-distance refinement on %s: %s -> %s m per frame; conditioned target=%s',
                                  original.device, initial_distances.tolist(), best_distances.tolist(), conditioned_target)
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
            if 'ot_range_initial_m' in sample:
                rows[-1].update(range_initial_m=[finite_number(value) for value in np.asarray(sample['ot_range_initial_m']).reshape(-1)],
                                range_final_m=[finite_number(value) for value in np.asarray(sample['ot_range_final_m']).reshape(-1)],
                                range_target='conditioning_scene' if bool(sample['ot_range_conditioned']) else 'training_reference_quantiles')
    return {'enabled': bool(rows), 'reference': config.reference or None,
            'reference_sha256': digest(config.reference) if rows else None, 'reference_split': 'train',
            'samples': rows, 'range_weight': config.generation_range_weight,
            'objective': ('Frame-wise range W1 in metres plus ' if config.generation_range_weight else '') +
                         'conditional Sinkhorn, unbalanced KL-OT and unsupported-bin penalty',
            'objective_fields': 'initial_objective/final_objective store secondary OT; range_initial_m/range_final_m store per-frame W1.',
            'scope': ('Final output after calibration/controller changes.' if final_stage else 'Generated sensor output.') +
                     ' Geometry-constrained LiDAR refinement; not an RGB image loss.',
            'selection': ('No frame range W1 may worsen; smaller mean W1 is primary, secondary OT breaks ties. '
                          'Conditioned frame return counts cannot fall below the original. Original sample is a candidate.'
                          if config.generation_range_weight else 'Best exported secondary OT objective including the original sample.'),
            'limits': 'Per-sample refinement does not guarantee that a trained model outperforms an untrained model.'}
