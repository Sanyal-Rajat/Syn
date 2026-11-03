"""Bounded actions, a budget-aware training loop and independent replay buffers."""
from collections import deque

import numpy as np
import torch
from torch import nn

from . import BudgetExhausted
from .config import RLConfig
from ..logs import get_logger


def mlp(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, outputs))


def state_of(objective, unit):
    observe = getattr(objective, 'context', None) or objective.observation
    return np.asarray(observe(unit), dtype='float32')


def discrete_unit(unit, action):
    candidate = np.asarray(unit, dtype='float32').copy()
    if action < 2 * len(candidate):
        candidate[action // 2] += .1 if action % 2 else -.1
    return candidate.clip(0, 1)


def cpu_state(network):
    return {name: value.detach().cpu().clone() for name, value in network.state_dict().items()}


class Replay:
    def __init__(self, capacity=10000, prioritized=False):
        self.rows = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.prioritized = prioritized

    def add(self, row):
        self.rows.append(row)
        self.priorities.append(max(self.priorities, default=1.))

    def sample(self, size, rng, device, beta=.4):
        size = min(size, len(self.rows))
        probability = np.asarray(self.priorities, dtype='float64') ** .6
        probability = probability / probability.sum() if self.prioritized else np.full(len(self.rows), 1 / len(self.rows))
        indices = rng.choice(len(self.rows), size, replace=self.prioritized, p=probability)
        weights = (len(self.rows) * probability[indices]) ** -beta
        weights /= weights.max()
        batch = [torch.as_tensor(np.asarray(values), dtype=torch.float32, device=device)
                 for values in zip(*(self.rows[int(index)] for index in indices))]
        return batch, torch.as_tensor(weights, dtype=torch.float32, device=device), indices

    def update(self, indices, errors):
        for index, error in zip(indices, errors):
            self.priorities[int(index)] = float(abs(error)) + 1e-5


class NStep:
    def __init__(self, steps=3, gamma=.99):
        self.steps, self.gamma = steps, gamma
        self.pending = deque()

    def _pop(self):
        state, action = self.pending[0][:2]
        reward, discount = 0., 1.
        for _, _, value, next_state, done in list(self.pending)[:self.steps]:
            reward += discount * value
            discount *= self.gamma
            if done:
                break
        self.pending.popleft()
        return state, action, reward, next_state, done, discount

    def add(self, transition):
        self.pending.append(transition)
        if transition[4]:
            return self.finish()
        return [self._pop()] if len(self.pending) >= self.steps else []

    def finish(self):
        if self.pending:
            last = self.pending[-1]
            self.pending[-1] = (*last[:4], True)
        return [self._pop() for _ in range(len(self.pending))]


def train(objective, make_agent, algorithm, *, seed=0, device='cpu', config=None):
    config = RLConfig.model_validate(config or {})
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    unit = np.asarray(objective.initial, dtype='float32').copy()
    if not objective.history:
        objective(unit)
    agent = make_agent(len(state_of(objective, unit)), objective.dim, device, config)
    agent.warmup = min(config.warmup, max(1, objective.remaining // 2)) if hasattr(agent, 'warmup') else 0
    transitions = 0
    while objective.remaining:
        state = state_of(objective, unit)
        action = agent.action(state, rng, transitions)
        candidate = (np.clip(unit + .2 * action, 0, 1) if agent.continuous else discrete_unit(unit, action))
        previous = objective.history[-1]['loss']
        try:
            loss = objective(candidate)
        except BudgetExhausted:
            break
        reward = float(getattr(objective, 'last_reward', previous - loss))
        agent.observe((state, action, reward, state_of(objective, candidate), not objective.remaining), rng)
        unit = candidate
        transitions += 1
    agent.finish(rng)
    policy = agent.export()
    policy.update(algorithm=algorithm, transitions=transitions, resolved_warmup=agent.warmup)
    get_logger().info('%s: %d transitions, %d learning updates, assigned device=%s',
                      algorithm, transitions, policy['updates'], device)
    return policy
