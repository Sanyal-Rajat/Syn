"""n_step_actor_critic: independent method entry point backed by the shared policy_gradient library."""
from .policy_gradient import optimize as fit


def optimize(objective, seed=0, device="cpu", config=None, **_):
    return fit(objective, "n_step_actor_critic", seed=seed, device=device, config=config)
