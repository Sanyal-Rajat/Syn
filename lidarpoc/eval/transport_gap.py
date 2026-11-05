"""Finite-support OT reporting; independent of diffusion training and RL losses."""
from dataclasses import asdict, dataclass
import warnings

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import kl_div
from scipy.stats import wasserstein_distance


@dataclass(frozen=True)
class TransportSettings:
    max_points: int = 128
    projections: int = 64
    epsilon: float = .05
    mass_penalty: float = 1.
    iterations: int = 1000
    gw_iterations: int = 100
    fgw_alpha: float = .5
    seed: int = 0

    def __post_init__(self):
        if not 8 <= self.max_points <= 512 or not 8 <= self.projections <= 512:
            raise ValueError('OT support and projection counts must be in 8..512')
        if (not np.isfinite([self.epsilon, self.mass_penalty, self.fgw_alpha]).all()
                or self.epsilon <= 0 or self.mass_penalty <= 0 or self.iterations < 1
                or self.gw_iterations < 1 or not 0 <= self.fgw_alpha <= 1):
            raise ValueError('Invalid transport regularization, iteration limit or FGW alpha')


LABELS = {'wasserstein_1': 'Wasserstein-1', 'wasserstein_2': 'Wasserstein-2',
    'sliced_wasserstein_2': 'Sliced Wasserstein-2', 'sinkhorn_divergence': 'Debiased Sinkhorn',
    'gromov_wasserstein': 'Gromov-Wasserstein sqrt loss',
    'fused_gromov_wasserstein': 'Fused GW objective', 'unbalanced_ot': 'Unbalanced KL-OT objective'}


def scalar_gap(a, b, unit):
    """Exact 1D empirical W1/W2 on every valid value, rather than histogram errors."""
    import ot

    a, b = np.asarray(a, dtype=float).reshape(-1), np.asarray(b, dtype=float).reshape(-1)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    values = {'wasserstein_1': None, 'wasserstein_2': None}
    if len(a) and len(b):
        values.update(wasserstein_1=float(wasserstein_distance(a, b)),
                      wasserstein_2=float(np.sqrt(max(0., ot.wasserstein_1d(a, b, p=2)))))
    return {'values': values, 'units': {key: unit for key in values}, 'count_a': len(a), 'count_b': len(b),
            'method': 'Exact one-dimensional empirical transport on all finite values',
            'diagnostics': {} if len(a) and len(b) else {'reason': 'An input distribution is empty'}}


def empirical_cdf(values):
    values = np.sort(np.asarray(values, dtype=float).reshape(-1))
    values = values[np.isfinite(values)]
    if not len(values):
        return {'x': [], 'cdf': [], 'count': 0}
    index = np.unique(np.linspace(0, len(values) - 1, min(len(values), 256), dtype=int))
    return {'x': [float(values[0]), *values[index].tolist()], 'cdf': [0., *((index + 1) / len(values)).tolist()],
            'count': len(values)}


def _nonnegative(value):
    value = float(value)
    if not np.isfinite(value) or value < -1e-8:
        raise ValueError(f'Invalid numerical OT objective: {value}')
    return max(0., value)


def multivariate_gap(x, y, scale, unit, mass_a, mass_b, attributes_a, attributes_b, settings):
    """Shared physical scale; sampled empirical supports; feasible GW upper bounds."""
    import ot

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    result = {'values': {key: None for key in LABELS},
        'units': {key: unit for key in LABELS}, 'count_a': len(x), 'count_b': len(y),
        'normalization_scale': float(scale), 'mass_a': float(mass_a), 'mass_b': float(mass_b),
        'settings': asdict(settings), 'diagnostics': {}}
    result['units'].update(sinkhorn_divergence=unit + '^2', fused_gromov_wasserstein='normalized objective',
                           unbalanced_ot='normalized objective')
    if not len(x) or not len(y):
        result['diagnostics']['reason'] = 'An input distribution is empty; no geometry metric is fabricated'
        return result, {}
    if (not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(scale) or scale <= 0
            or x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]):
        raise ValueError('OT requires finite point matrices with shared dimensions and a positive common scale')
    ix = np.sort(np.random.default_rng(settings.seed).choice(len(x), min(len(x), settings.max_points), replace=False))
    iy = np.sort(np.random.default_rng(settings.seed).choice(len(y), min(len(y), settings.max_points), replace=False))
    x, y = x[ix] / scale, y[iy] / scale
    a, b = np.full(len(x), 1 / len(x)), np.full(len(y), 1 / len(y))
    result.update(support_a=len(x), support_b=len(y),
                  method='Uniform seeded subsample; exact LP only on this support; GW/FGW nonconvex local solutions')
    cost = cdist(x, y)
    squared = cost**2
    arrays = {'source_indices': ix, 'target_indices': iy, 'source_support_normalized': x,
              'target_support_normalized': y, 'cross_cost_normalized': cost}

    def estimate(name, function):
        try:
            with warnings.catch_warnings(record=True) as messages:
                warnings.simplefilter('always')
                value, diagnostic = function()
            result['values'][name] = _nonnegative(value)
            result['diagnostics'][name] = {**diagnostic, 'warnings': [str(message.message) for message in messages]}
        except (ValueError, FloatingPointError, RuntimeError) as error:
            result['diagnostics'][name] = {'reason': str(error)}

    plans = {}

    def linear(name, matrix, power):
        plan, log = ot.emd(a, b, matrix, log=True, numItermax=100000, numThreads=1)
        residual = max(abs(plan.sum(1) - a).max(), abs(plan.sum(0) - b).max())
        if log.get('warning') or residual > 1e-7 or not np.isfinite(plan).all():
            raise ValueError(f'LP transport failed: {log.get("warning")}; marginal residual={residual}')
        plans[name] = arrays[name + '_plan'] = plan
        return _nonnegative((plan * matrix).sum()) ** (1 / power) * scale, {'marginal_residual': float(residual)}

    estimate('wasserstein_1', lambda: linear('wasserstein_1', cost, 1))
    estimate('wasserstein_2', lambda: linear('wasserstein_2', squared, 2))
    estimate('sliced_wasserstein_2', lambda: (float(ot.sliced_wasserstein_distance(
        x, y, n_projections=settings.projections, p=2, seed=settings.seed)) * scale,
        {'projections': settings.projections, 'seed': settings.seed}))

    def regularized(source, target, matrix):
        warmstart, attempts, total = None, [], 0
        for limit in (settings.iterations, settings.iterations * 9):
            plan, log = ot.bregman.sinkhorn_log(source, target, matrix, settings.epsilon,
                numItermax=limit, stopThr=1e-9, log=True, warmstart=warmstart)
            residual = max(abs(plan.sum(1) - source).max(), abs(plan.sum(0) - target).max())
            total += int(log['niter']) + 1
            attempts.append({'iterations': int(log['niter']) + 1, 'marginal_residual': float(residual)})
            if np.isfinite(plan).all() and residual <= 1e-7:
                break
            warmstart = (log['log_u'], log['log_v'])
        if not np.isfinite(plan).all() or residual > 1e-6:
            raise ValueError(f'Sinkhorn marginal residual={residual}; increase iterations or regularization')
        objective = (plan * matrix).sum() + settings.epsilon * kl_div(plan, source[:, None] * target[None]).sum()
        return float(objective), {'marginal_residual': float(residual), 'iterations': total, 'attempts': attempts}

    def sinkhorn():
        cross, error = regularized(a, b, squared)
        self_a, error_a = regularized(a, a, cdist(x, x)**2)
        self_b, error_b = regularized(b, b, cdist(y, y)**2)
        raw = cross - .5 * (self_a + self_b)
        return _nonnegative(raw) * scale**2, {'marginal_residual': max(value['marginal_residual'] for value in (error, error_a, error_b)),
            'cross_solver': error, 'self_a_solver': error_a, 'self_b_solver': error_b,
            'normalized_raw_value': raw, 'epsilon_normalized': settings.epsilon,
            'epsilon_in_squared_units': settings.epsilon * scale**2}

    estimate('sinkhorn_divergence', sinkhorn)
    c1, c2 = cdist(x, x), cdist(y, y)
    attributes_a = np.asarray(attributes_a, dtype=float)[ix]
    attributes_b = np.asarray(attributes_b, dtype=float)[iy]
    feature_cost = cdist(attributes_a, attributes_b)**2

    def gromov(fused):
        candidates = []
        starts = [('product', a[:, None] * b[None])]
        if 'wasserstein_2' in plans:
            starts.append(('coordinate_W2', plans['wasserstein_2']))
        for start_name, initial in starts:
            kwargs = dict(p=a, q=b, loss_fun='square_loss', symmetric=True, G0=initial.copy(), log=True,
                          max_iter=settings.gw_iterations, tol_abs=1e-9, tol_rel=1e-7)
            if fused:
                value, log = ot.gromov.fused_gromov_wasserstein2(feature_cost, c1, c2,
                                                               alpha=settings.fgw_alpha, **kwargs)
            else:
                value, log = ot.gromov.gromov_wasserstein2(c1, c2, **kwargs)
            plan = log['T']
            residual = max(abs(plan.sum(1) - a).max(), abs(plan.sum(0) - b).max())
            if np.isfinite(value) and value >= -1e-8 and residual <= 1e-7:
                losses = log['loss']
                change = abs(float(losses[-1]) - float(losses[-2])) if len(losses) > 1 else 0.
                stopped = change <= 1e-9 or change / max(abs(float(losses[-2])), 1e-15) <= 1e-7 if len(losses) > 1 else True
                candidates.append((float(value), plan, {'initialization': start_name, 'iterations': len(losses) - 1,
                    'stopping_criterion_met': bool(stopped), 'marginal_residual': float(residual)}))
        if not candidates:
            raise ValueError('No finite feasible GW solution was found')
        value, plan, diagnostic = min(candidates, key=lambda item: item[0])
        name = 'fused_gromov_wasserstein' if fused else 'gromov_wasserstein'
        arrays[name + '_plan'] = plan
        diagnostic.update(normalized_objective=value, starts=[name for name, _ in starts],
                          global_optimum_guaranteed=False)
        return (_nonnegative(value) if fused else np.sqrt(_nonnegative(value)) * scale), diagnostic

    estimate('gromov_wasserstein', lambda: gromov(False))
    estimate('fused_gromov_wasserstein', lambda: gromov(True))

    def unbalanced():
        source, target = a * mass_a, b * mass_b
        plan, log = ot.unbalanced.sinkhorn_unbalanced(source, target, squared, settings.epsilon,
            settings.mass_penalty, reg_type='kl', method='sinkhorn_stabilized', numItermax=settings.iterations,
            stopThr=1e-8, log=True)
        if not np.isfinite(plan).all():
            raise ValueError('Unbalanced transport returned a nonfinite coupling')
        # Generalized KL includes -mass(P)+mass(Q); POT's log cost omits these terms.
        objective = ((plan * squared).sum()
            + settings.epsilon * kl_div(plan, source[:, None] * target[None]).sum()
            + settings.mass_penalty * (kl_div(plan.sum(1), source).sum() + kl_div(plan.sum(0), target).sum()))
        arrays['unbalanced_ot_plan'] = plan
        errors = log.get('err', [])
        return objective, {'transported_mass': float(plan.sum()),
            'fixed_point_error': float(errors[-1]) if len(errors) and np.isfinite(errors[-1]) else None,
            'stopping_criterion_met': bool(len(errors) and errors[-1] <= 1e-8),
            'identity_zero_guaranteed': False}

    estimate('unbalanced_ot', unbalanced)
    return result, arrays
