"""Linear UCB over fixed bounded controller profiles, one choice per context."""
import numpy as np


PROFILES = np.array([[1 / 3, 0, .1, 0], [.5, .1, .3, .5], [.8, .2, .6, 1], [1 / 3, .4, .9, 0]], dtype='float32')


def choose(policy, context, explore=False):
    inverses = np.linalg.inv(policy['covariance'])
    coefficients = np.einsum('aij,aj->ai', inverses, policy['rewards'])
    values = coefficients @ context
    if explore:
        values += np.sqrt(np.einsum('i,aij,j->a', context, inverses, context).clip(0))
    return int(values.argmax())


def optimize(objective, seed=0, **_):
    size = len(objective.context(objective.initial))
    policy = {'algorithm': 'contextual_bandit', 'covariance': np.repeat(np.eye(size)[None], len(PROFILES), axis=0),
              'rewards': np.zeros((len(PROFILES), size)), 'profiles': PROFILES.copy()}
    while objective.remaining:
        scenario = len(objective.history) % len(objective.scenes)
        context = objective.context(objective.initial, scenario=scenario)
        arm = choose(policy, context, explore=True)
        unit = PROFILES[arm].copy()
        unit[2] = (scenario + .5) / len(objective.scenes)
        objective(unit)
        policy['covariance'][arm] += np.outer(context, context)
        policy['rewards'][arm] -= objective.history[-1]['loss'] * context
    return policy
