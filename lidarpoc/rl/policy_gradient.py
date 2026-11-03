"""A2C/GAE, REINFORCE and synchronous n-step actor-critic adapted from the source agents."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import cpu_state, mlp, train


class PolicyNetwork(nn.Module):
    def __init__(self, states, actions, critic=True):
        super().__init__()
        self.shared = mlp(states, 64)
        self.actor = nn.Linear(64, actions)
        self.critic = nn.Linear(64, 1) if critic else None

    def forward(self, states):
        hidden = self.shared(states)
        return self.actor(hidden), self.critic(hidden).squeeze(-1) if self.critic else hidden.new_zeros(len(hidden))


class PolicyGradientAgent:
    continuous = False

    def __init__(self, states, dimensions, device, config, algorithm):
        self.device, self.states, self.actions = device, states, 2 * dimensions + 1
        self.config, self.algorithm = config, algorithm
        self.network = PolicyNetwork(states, self.actions, algorithm != 'reinforce').to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=3e-4)
        self.rows, self.updates = [], 0

    def action(self, state, rng, step):
        with torch.no_grad():
            logits, _ = self.network(torch.as_tensor(state, device=self.device)[None])
            return int(torch.distributions.Categorical(logits=logits).sample().item())

    def observe(self, transition, rng):
        self.rows.append(transition)
        if transition[4] or (self.algorithm != 'reinforce' and len(self.rows) >= self.config.rollout_steps):
            self.learn()

    def learn(self, terminal=False):
        if not self.rows:
            return
        states, actions, rewards, next_states, dones = zip(*self.rows)
        states = torch.as_tensor(np.asarray(states), device=self.device)
        actions = torch.as_tensor(actions, device=self.device)
        logits, values = self.network(states)
        with torch.no_grad():
            _, final_value = self.network(torch.as_tensor(next_states[-1], device=self.device)[None])
            bootstrap = 0. if terminal or dones[-1] or self.algorithm == 'reinforce' else float(final_value.item())
            targets, advantage, running = [], 0., bootstrap
            if self.algorithm == 'a2c':
                advantages = []
                for index in reversed(range(len(rewards))):
                    next_value = float(values[index + 1]) if index + 1 < len(values) else bootstrap
                    delta = rewards[index] + .99 * (1 - dones[index]) * next_value - float(values[index])
                    advantage = delta + .99 * .95 * (1 - dones[index]) * advantage
                    advantages.insert(0, advantage)
                targets = torch.as_tensor(advantages, device=self.device) + values.detach()
            else:
                for reward, done in zip(reversed(rewards), reversed(dones)):
                    running = reward + .99 * (1 - done) * running
                    targets.insert(0, running)
                targets = torch.as_tensor(targets, dtype=torch.float32, device=self.device)
        distribution = torch.distributions.Categorical(logits=logits)
        advantage = targets - values.detach() if self.algorithm != 'reinforce' else targets
        if self.algorithm == 'reinforce' and len(advantage) > 1:
            advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
        loss = -(distribution.log_prob(actions) * advantage).mean()
        if self.algorithm != 'reinforce':
            loss += .5 * F.mse_loss(values, targets) - .01 * distribution.entropy().mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), .5)
        self.optimizer.step()
        self.updates += 1
        self.rows.clear()

    def finish(self, rng):
        self.learn(terminal=True)

    def export(self):
        return {'states': self.states, 'actions': self.actions, 'updates': self.updates,
                'action_kind': 'discrete', 'network': cpu_state(self.network)}


def optimize(objective, algorithm, **kwargs):
    return train(objective, lambda s, d, dev, c: PolicyGradientAgent(s, d, dev, c, algorithm), algorithm, **kwargs)
