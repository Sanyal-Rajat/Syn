"""c51: independent method entry point backed by the shared q_learning library."""
from .q_learning import optimize as fit


def optimize(objective, seed=0, device="cpu", config=None, **_):
    return fit(objective, "c51", seed=seed, device=device, config=config)
