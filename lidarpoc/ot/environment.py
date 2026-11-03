"""Frozen-generator control with explicit generation-call and elapsed-time budgets."""
import time

import numpy as np
import torch

from ..logs import get_logger
from ..rl import BudgetExhausted
from .constraints import scene_metrics
from .features import conditional_ot, features, mass_ot, return_mass, select
from .reference import stack_samples


def score(samples, reference, sensor, config, device='cpu'):
    sample = stack_samples(samples, device)
    vector, bins, valid = features(sample, sensor, config)
    selected = select(vector, bins, valid, config.max_points)
    target = {int(key): torch.as_tensor(reference['features'][reference['bins'] == key], device=device)
              for key in np.unique(reference['bins'])}
    try:
        conditional, unsupported, per_bin = conditional_ot(selected, target, config, return_bins=True)
        cot, supported = float(conditional), True
    except ValueError:
        cot, unsupported, supported, per_bin = None, sorted(set(target) | set(selected)), False, {}
    centers, mass = return_mass(sample, sensor, config)
    target_mass = torch.as_tensor(reference['mass']).to(mass)
    uot = float(mass_ot(mass, target_mass, centers, config))
    count_error = float((mass - target_mass).abs().sum())
    return {'conditional_sinkhorn': cot, 'unbalanced_ot': uot, 'return_mass_l1': count_error,
            'return_probability': float(mass.sum()), 'reference_return_probability': float(target_mass.sum()),
            'point_count_per_frame': float(mass.sum()) * sensor['width'] * len(sensor['elevations']),
            'reference_point_count_per_frame': float(target_mass.sum()) * sensor['width'] * len(sensor['elevations']),
            'unsupported_bins': unsupported, 'supported': supported,
            'conditional_by_bin': per_bin,
            'mass_error_by_bin': (mass - target_mass).detach().tolist(),
            'gap': (cot if supported else 1.) + uot + count_error + len(unsupported) / max(1, len(target))}


def decode(unit, scenarios, max_samples):
    unit = np.clip(np.asarray(unit, dtype='float64'), 0, 1)
    return {'guidance': float(3 * unit[0]), 'dropout': float(.5 * unit[1]),
            'scenario': min(scenarios - 1, int(unit[2] * scenarios)),
            'samples': 1 + int(round(unit[3] * (max_samples - 1)))}


class SensorAdaptationEnv:
    def __init__(self, renderer, scenes, reference, sensor, ot, settings, budget, seed=0):
        self.renderer, self.scenes, self.reference, self.sensor = renderer, scenes, reference, sensor
        self.ot, self.settings, self.budget, self.seed = ot, settings, budget, seed
        self.dim, self.initial = 4, np.array([1 / 3, 0., 0., 0.], dtype='float32')
        self.history, self.calls, self.last_reward, self.previous_gap = [], 0, 0., None
        self.started = time.perf_counter()
        self.last_components = {'loss': 0., 'range_wasserstein_normalized': 0., 'return_rate_mae': 0.,
                                'intensity_wasserstein': 0., 'temporal_discrepancy': 0.}

    @property
    def remaining(self):
        return max(0, self.budget - self.calls) if time.perf_counter() - self.started < self.settings.controller_seconds else 0

    def observation(self, unit):
        row = self.last_components
        return np.r_[unit, [row['loss'], row['range_wasserstein_normalized'], row['return_rate_mae'],
                            self.remaining / self.budget, self.last_reward]].astype('float32').clip(-10, 10)

    def context(self, unit, scenario=None):
        selected = decode(unit, len(self.scenes), self.settings.controller_max_samples)['scenario'] if scenario is None else scenario
        scene = self.scenes[selected]
        valid = np.asarray(scene['valid']).astype(bool)
        ranges = np.asarray(scene['range'])
        descriptors = [float(valid.mean()), float(ranges[valid].mean() / self.sensor['max_range_m']) if valid.any() else 0.,
                       selected / max(1, len(self.scenes) - 1), 1.]
        row = self.history[-1] if self.history else {}
        gaps = row.get('conditional_by_bin', {})
        count = self.ot.ring_groups * (len(self.ot.range_edges_m) - 1)
        vector = [gaps.get(str(index), 0.) for index in range(count)]
        mass = row.get('mass_error_by_bin', [0.] * len(self.reference['mass']))
        return np.r_[self.observation(unit), descriptors, vector, mass].astype('float32').clip(-10, 10)

    def __call__(self, unit):
        if not self.remaining:
            raise BudgetExhausted('Controller generation/time budget exhausted')
        controls = decode(unit, len(self.scenes), self.settings.controller_max_samples)
        count = min(controls['samples'], self.remaining)
        requests = [(controls, self.seed + self.calls + index) for index in range(count)]
        samples = self.renderer.render_many(requests)
        self.calls += count
        metrics = score(samples, self.reference, self.sensor, self.ot, next(self.renderer.model.parameters()).device)
        geometry = [scene_metrics(sample, self.scenes[controls['scenario']], self.sensor,
                                   self.renderer.tolerance) for sample in samples]
        physics = sum(not row['geometry_pass'] for row in geometry) / len(geometry)
        cost = count / self.budget
        improvement = 0. if self.previous_gap is None else self.previous_gap - metrics['gap']
        self.last_reward = improvement - self.settings.physics_penalty * physics - self.settings.compute_penalty * cost
        self.previous_gap = metrics['gap']
        loss = metrics['gap'] + self.settings.physics_penalty * physics + self.settings.compute_penalty * cost
        self.last_components = {'loss': loss, 'range_wasserstein_normalized': metrics['conditional_sinkhorn'] or 0.,
                                'return_rate_mae': metrics['return_mass_l1'], 'intensity_wasserstein': 0.,
                                'temporal_discrepancy': 0.}
        self.history.append({'unit': np.clip(unit, 0, 1).tolist(), 'loss': loss, 'generation_calls': self.calls,
                             'controls': controls, 'reward': self.last_reward, 'physics': physics,
                             'elapsed_s': time.perf_counter() - self.started, **metrics})
        get_logger().info('Controller calls %d/%d: gap=%.6f reward=%.6f', self.calls, self.budget, metrics['gap'], self.last_reward)
        return loss

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed = seed
        self.calls, self.history, self.previous_gap = 0, [], None
        self.started = time.perf_counter()
        self(self.initial)
        return self.context(self.initial), {}

    def step(self, action):
        previous = np.asarray(self.history[-1]['unit'] if self.history else self.initial)
        unit = np.clip(previous + .2 * np.clip(action, -1, 1), 0, 1)
        self(unit)
        return self.context(unit), self.last_reward, self.calls >= self.budget, self.remaining == 0 and self.calls < self.budget, self.history[-1]
