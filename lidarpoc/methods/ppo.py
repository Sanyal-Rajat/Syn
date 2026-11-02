"""Discrete bounded-update PPO, adapted from the user's on_policy_RL_Agents.py.

The actor/critic, clipped policy objective and GAE follow the existing implementation.
Actions select one parameter to increase/decrease, or leave the profile unchanged.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..gpu import DistributedHandler


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_size, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.actor, self.critic = nn.Linear(64, action_size), nn.Linear(64, 1)

    def forward(self, state):
        hidden = self.shared(state)
        return F.softmax(self.actor(hidden), dim=-1), self.critic(hidden)


class PPOAgent:
    def __init__(self, state_size, action_size, device='cpu', data_parallel=False):
        self.device = device
        self.network = ActorCriticNetwork(state_size, action_size).to(device)
        self.network = DistributedHandler.distribute_model(self.network, data_parallel)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=3e-4)
        self.buffer = []

    def select_action(self, state):
        with torch.no_grad():
            probabilities, value = self.network(torch.as_tensor(state, device=self.device).float()[None])
            distribution = torch.distributions.Categorical(probabilities)
            action = distribution.sample()
        return int(action.item()), float(distribution.log_prob(action).item()), float(value.item())

    def learn(self):
        if not self.buffer:
            return
        states, actions, log_probs, values, rewards, dones = map(np.asarray, zip(*self.buffer))
        advantage, accumulated = np.zeros(len(rewards)), 0.
        for i in range(len(rewards) - 1, -1, -1):
            next_value = values[i + 1] if i + 1 < len(values) else 0.
            delta = rewards[i] + .99 * next_value * (1 - dones[i]) - values[i]
            accumulated = delta + .99 * .95 * accumulated * (1 - dones[i])
            advantage[i] = accumulated
        returns = advantage + values
        if len(advantage) > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        tensor = lambda value, dtype=torch.float32: torch.as_tensor(value, dtype=dtype, device=self.device)
        states, actions, old_log = tensor(states), tensor(actions, torch.long), tensor(log_probs)
        advantage, returns = tensor(advantage), tensor(returns)
        for _ in range(4):
            for indices in torch.randperm(len(states), device=self.device).split(64):
                probabilities, predicted = self.network(states[indices])
                distribution = torch.distributions.Categorical(probabilities)
                ratio = (distribution.log_prob(actions[indices]) - old_log[indices]).exp()
                unclipped = ratio * advantage[indices]
                clipped = ratio.clamp(.8, 1.2) * advantage[indices]
                loss = -torch.minimum(unclipped, clipped).mean()
                loss += .5 * F.mse_loss(predicted[:, 0], returns[indices]) - .01 * distribution.entropy().mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), .5)
                self.optimizer.step()
        self.buffer.clear()


def optimize(objective, seed=0, device='cpu', data_parallel=False, **_):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    agent = PPOAgent(objective.dim + 5, 2 * objective.dim + 1, device, data_parallel)
    unit = objective.initial.copy()
    loss = objective.history[0]['loss']
    for step in range(objective.remaining):
        if step and step % 10 == 0 and objective.remaining > 1:
            unit = rng.uniform(size=objective.dim)
            loss = objective(unit)
        if not objective.remaining:
            break
        components = objective.last_components
        state = objective.observation(unit)
        action, log_prob, value = agent.select_action(state)
        candidate = unit.copy()
        if action < 2 * objective.dim:
            candidate[action // 2] += .1 if action % 2 else -.1
        candidate = np.clip(candidate, 0, 1)
        next_loss = objective(candidate)
        done = (step + 1) % 10 == 0 or objective.remaining == 0
        agent.buffer.append((state, action, log_prob, value, loss - next_loss - 1e-4, done))
        unit, loss = candidate, next_loss
        if done:
            agent.learn()
    agent.learn()
    network = agent.network.module if isinstance(agent.network, nn.DataParallel) else agent.network
    return {key: value.detach().cpu() for key, value in network.state_dict().items()}
