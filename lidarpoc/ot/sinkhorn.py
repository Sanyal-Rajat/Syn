"""Log-domain balanced/unequal-mass scaling for KL(gamma | a tensor b).

Chizat et al., arXiv:1607.05816. Balanced debiasing: Feydy et al.,
arXiv:1810.08278. Iterations remain differentiable; real inputs are detached by
the conditional-loss caller. Unbalanced output is the full regularized objective.
"""
import torch


def physical_cost(source, target, scales, weights):
    scale = source.new_tensor(scales)
    weight = source.new_tensor(weights)
    return (((source[:, None] - target[None]) / scale).square() * weight).sum(-1)


def kl(measure, reference):
    tiny = torch.finfo(measure.dtype).tiny
    return (torch.where(measure > 0, measure * (measure.clamp_min(tiny).log()
             - reference.clamp_min(tiny).log()), 0) - measure + reference).sum()


def transport(cost, source_mass, target_mass, epsilon=.1, iterations=60, tau_source=None, tau_target=None):
    if cost.ndim != 2 or cost.shape != (len(source_mass), len(target_mass)):
        raise ValueError('Transport cost and marginal dimensions do not match')
    if epsilon <= 0 or iterations < 1 or any(tau is not None and tau <= 0 for tau in (tau_source, tau_target)):
        raise ValueError('Transport regularization, mass penalties and iteration count must be positive')
    if (not torch.isfinite(cost).all() or not torch.isfinite(source_mass).all()
            or not torch.isfinite(target_mass).all() or (source_mass < 0).any() or (target_mass < 0).any()):
        raise ValueError('Transport requires finite costs and nonnegative finite masses')
    balanced = tau_source is None and tau_target is None
    if (tau_source is None) != (tau_target is None):
        raise ValueError('Specify both unbalanced mass penalties')
    if balanced and not torch.allclose(source_mass.sum(), target_mass.sum(), rtol=1e-5, atol=1e-7):
        raise ValueError('Balanced transport requires equal total masses')
    active_source, active_target = source_mass > 0, target_mass > 0
    if not active_source.any() or not active_target.any():
        value = cost.sum() * 0 + source_mass.sum() * 0 + target_mass.sum() * 0
        if not balanced:
            value = value + tau_source * source_mass.sum() + tau_target * target_mass.sum()
        return value
    a, b = source_mass[active_source], target_mass[active_target]
    c = cost[active_source][:, active_target]
    loga, logb = a.log(), b.log()
    log_kernel = loga[:, None] + logb[None] - c / epsilon
    u, v = torch.zeros_like(a), torch.zeros_like(b)
    rho_a = 1. if balanced else tau_source / (tau_source + epsilon)
    rho_b = 1. if balanced else tau_target / (tau_target + epsilon)
    for _ in range(iterations):
        u = rho_a * (loga - torch.logsumexp(log_kernel + v[None], dim=1))
        v = rho_b * (logb - torch.logsumexp(log_kernel + u[:, None], dim=0))
    plan = (log_kernel + u[:, None] + v[None]).exp()
    value = (plan * c).sum() + epsilon * kl(plan, a[:, None] * b[None])
    if not balanced:
        value = value + tau_source * kl(plan.sum(1), a) + tau_target * kl(plan.sum(0), b)
    return value


def sinkhorn_divergence(source, target, scales, weights, epsilon=.1, iterations=60):
    a = source.new_full((len(source),), 1 / len(source))
    b = target.new_full((len(target),), 1 / len(target))
    cross = transport(physical_cost(source, target, scales, weights), a, b, epsilon, iterations)
    self_source = transport(physical_cost(source, source, scales, weights), a, a, epsilon, iterations)
    self_target = transport(physical_cost(target, target, scales, weights), b, b, epsilon, iterations)
    return cross - .5 * (self_source + self_target)
