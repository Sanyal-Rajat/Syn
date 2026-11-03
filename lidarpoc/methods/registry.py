"""Shared selection and lazy dispatch; configuration/UI imports never load Torch."""
from importlib import import_module
from typing import Literal, Union

from ..rl import RL_METHODS, RLMethod

CalibrationMethod = Union[RLMethod, Literal['random_search', 'bayesian', 'cma_es']]
ControllerMethod = Union[CalibrationMethod, Literal['fixed', 'contextual_bandit']]
CALIBRATION_METHODS = (*RL_METHODS, 'random_search', 'bayesian', 'cma_es')
CONTROLLER_METHODS = ('fixed', 'random_search', 'bayesian', 'cma_es', *RL_METHODS, 'contextual_bandit')


def optimize(method, objective, *, rl=None, **kwargs):
    if method not in CALIBRATION_METHODS and method != 'contextual_bandit':
        raise ValueError(f'Unknown optimization method: {method}')
    package = 'rl' if method in RL_METHODS else 'methods'
    if package == 'rl' and method not in ('ppo', 'sac'):
        kwargs['config'] = rl
    return import_module(f'lidarpoc.{package}.{method}').optimize(objective, **kwargs)
