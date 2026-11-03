"""RL library: algorithm names are available without importing PyTorch."""
from typing import Literal, get_args

RLMethod = Literal['ppo', 'sac', 'a2c', 'reinforce', 'n_step_actor_critic', 'ddpg', 'td3',
                   'dqn', 'enhanced_dqn', 'n_step_dqn', 'c51', 'rainbow']
RL_METHODS = get_args(RLMethod)


class BudgetExhausted(RuntimeError):
    """A controller has reached its generation-call or wall-time boundary."""
