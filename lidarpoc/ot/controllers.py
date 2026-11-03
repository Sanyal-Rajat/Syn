"""Freeze generator parameters; compare controls under the same metered budget."""
import importlib
import time

import numpy as np
import torch

from .environment import BudgetExhausted, SensorAdaptationEnv, decode
from ..methods import contextual_bandit
from ..methods.ppo import ActorCriticNetwork
from ..methods.sac import Actor
from ..logs import get_logger
from ..sampling import SamplingPool
from .environment import score


class Renderer:
    def __init__(self, model, scenes, sampler=None):
        self.model, self.scenes = model.eval().requires_grad_(False), scenes
        self.sampler = sampler or SamplingPool(model)
        self.tolerance = model.config['geometry_tolerance_m']
        self.denoiser_evaluations = 0

    def __call__(self, controls, seed):
        return self.render_many([(controls, seed)])[0]

    def render_many(self, requests):
        inputs = []
        for controls, seed in requests:
            scene = self.scenes[controls['scenario']]
            conditioning = {key: torch.as_tensor(scene[key])[None] for key in ('range', 'valid')}
            conditioning['complete_scene'] = True
            inputs.append({'batch_size': 1, 'conditioning': conditioning, 'seed': seed,
                           'guidance': controls['guidance'], 'force_cfg': True})
        samples = self.sampler.sample_many(inputs)
        self.denoiser_evaluations += len(samples) * 2 * self.model.config['sample_steps']
        results = []
        for (controls, seed), sampled in zip(requests, samples):
            scene = self.scenes[controls['scenario']]
            result = {key: value[0].numpy() for key, value in sampled.items()}
            result['valid'] &= np.random.default_rng(seed).random(result['valid'].shape) >= controls['dropout']
            for key in ('range', 'intensity'):
                result[key] = np.where(result['valid'], result[key], 0).astype('float32')
            for key in ('labels', 'has_intensity', 'world_from_sensor', 'timestamp_ns'):
                if key in scene:
                    result[key] = np.asarray(scene[key]).copy()
            results.append(result)
        return results


def fit_controller(method, renderer, scenes, reference, sensor, ot, settings, budget, seed, device):
    env = SensorAdaptationEnv(renderer, scenes, reference, sensor, ot, settings, budget, seed)
    env.reset(seed)
    policy = None
    try:
        if method == 'fixed':
            while env.remaining:
                env(env.initial)
        else:
            policy = importlib.import_module(f'lidarpoc.methods.{method}').optimize(
                env, seed=seed, device=device, warmup=settings.sac_warmup, batch_size=settings.sac_batch_size)
    except BudgetExhausted:
        pass  # BO/CMA/random batches stop at the shared generation-call/time boundary.
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    result = {'method': method, 'unit': min(env.history, key=lambda row: row['loss'])['unit'],
              'history': env.history, 'generation_calls': env.calls, 'call_budget': budget,
              'elapsed_s': time.perf_counter() - env.started, 'seconds_budget': settings.controller_seconds,
              'time_budget_checked': 'between actions; final action/optimizer update can overrun',
              'frozen_generator': True, 'last_context': env.context(env.initial).tolist()}
    result['denoiser_evaluations'] = renderer.denoiser_evaluations
    if method == 'ppo' and policy:
        policy = {'algorithm': 'ppo', 'state': policy}
    return result, policy


def controlled_bank(renderer, scenes, count, seed, profile=None, policy=None):
    unit = np.asarray(profile['unit'] if profile else [1 / 3, 0., 0., 0.], dtype='float32')
    actor = None
    if policy and policy['algorithm'] == 'sac':
        actor = Actor(policy['states'], policy['actions'])
        actor.load_state_dict(policy['actor'])
        actor.eval()
    if policy and policy['algorithm'] == 'ppo':
        actor = ActorCriticNetwork(9, 9)
        actor.load_state_dict(policy['state'])
        actor.eval()
    samples, canonicals, requests = [], [], []
    inference_context = np.asarray(profile['last_context'], dtype='float32').copy() if profile else None
    for index in range(count):
        scenario = min(len(scenes) - 1, index * len(scenes) // count)
        chosen = unit.copy()
        if policy:
            context = inference_context.copy()
            context[:4] = chosen
            context[7] = (count - index) / count
            known = np.asarray(scenes[scenario]['valid']).astype(bool)
            context[9] = float(known.mean())
            context[10] = float(scenes[scenario]['range'][known].mean() / renderer.model.sensor_meta['max_range_m']) if known.any() else 0.
            context[11] = scenario / max(1, len(scenes) - 1)
            if policy['algorithm'] == 'sac':
                with torch.no_grad():
                    delta, _ = actor(torch.from_numpy(context)[None], deterministic=True)
                chosen = np.clip(chosen + .2 * delta[0].numpy(), 0, 1)
            elif policy['algorithm'] == 'contextual_bandit':
                chosen = np.asarray(policy['profiles'][contextual_bandit.choose(policy, context)]).copy()
            elif policy['algorithm'] == 'ppo':
                with torch.no_grad():
                    probabilities, _ = actor(torch.from_numpy(context[:9])[None])
                action = int(probabilities.argmax())
                if action < 8:
                    chosen[action // 2] = np.clip(chosen[action // 2] + (.1 if action % 2 else -.1), 0, 1)
        controls = decode(chosen, len(scenes), 1)
        # Evaluation uses a common, fixed scenario mixture and one sample per call.
        controls.update(scenario=scenario, samples=1)
        if policy:
            samples.append(renderer(controls, seed + index))
        else:
            requests.append((controls, seed + index))
        canonicals.append(scenes[scenario])
        if policy and hasattr(renderer, 'reference'):
            metrics = score([samples[-1]], renderer.reference, renderer.model.sensor_meta, renderer.ot,
                            next(renderer.model.parameters()).device)
            previous = float(inference_context[4])
            inference_context[4:7] = [metrics['gap'], metrics['conditional_sinkhorn'] or 0., metrics['return_mass_l1']]
            inference_context[8] = previous - metrics['gap']
            count_bins = renderer.ot.ring_groups * (len(renderer.ot.range_edges_m) - 1)
            inference_context[13:13 + count_bins] = [metrics['conditional_by_bin'].get(str(k), 0.) for k in range(count_bins)]
            inference_context[13 + count_bins:] = metrics['mass_error_by_bin']
            inference_context = inference_context.clip(-10, 10)
        unit = chosen
    if not policy:
        samples = renderer.render_many(requests)
    return samples, canonicals
