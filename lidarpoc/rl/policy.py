"""Reconstruct frozen policies without importing training orchestration or LiDAR models."""
import numpy as np
import torch

from .common import discrete_unit


def load_actor(policy):
    algorithm = policy['algorithm']
    if algorithm == 'ppo':
        from .ppo import ActorCriticNetwork
        actor = ActorCriticNetwork(9, 9)
        state = policy['state']
    elif algorithm == 'sac':
        from .sac import Actor
        actor = Actor(policy['states'], policy['actions'])
        state = policy['actor']
    elif algorithm in ('ddpg', 'td3'):
        from .deterministic import Actor
        actor = Actor(policy['states'], policy['actions'])
        state = policy['actor']
    elif algorithm in ('a2c', 'reinforce', 'n_step_actor_critic'):
        from .policy_gradient import PolicyNetwork
        actor = PolicyNetwork(policy['states'], policy['actions'], algorithm != 'reinforce')
        state = policy['network']
    elif algorithm in ('dqn', 'enhanced_dqn', 'n_step_dqn', 'c51', 'rainbow'):
        from .q_learning import QNetwork
        actor = QNetwork(policy['states'], policy['actions'], **policy['network_config'])
        state = policy['network']
    else:
        raise ValueError(f'Unknown frozen RL policy: {algorithm}')
    actor.load_state_dict(state)
    return actor.eval().requires_grad_(False)


def apply_policy(policy, actor, context, unit):
    state = torch.as_tensor(np.asarray(context)[:policy.get('states', 9)], dtype=torch.float32)[None]
    with torch.no_grad():
        if policy['algorithm'] == 'sac':
            action, _ = actor(state, deterministic=True)
            return np.clip(unit + .2 * action[0].numpy(), 0, 1)
        if policy.get('action_kind') == 'continuous':
            return np.clip(unit + .2 * actor(state)[0].numpy(), 0, 1)
        output = actor.values(state) if hasattr(actor, 'values') else actor(state)[0]
        return discrete_unit(unit, int(output.argmax(-1).item()))
