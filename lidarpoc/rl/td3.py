"""td3: independent method entry point backed by the shared deterministic library."""
from .deterministic import optimize as fit


def optimize(objective, seed=0, device="cpu", config=None, **_):
    return fit(objective, "td3", seed=seed, device=device, config=config)
