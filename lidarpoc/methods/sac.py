"""Continuous SAC: squashed Gaussian actor, twin Qs, replay and Polyak targets.

Equations follow Haarnoja et al. and the Spinning Up SAC algorithm. This is
independent of the disabled SAC stub in the original application source.
"""
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def network(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, outputs))


class Actor(nn.Module):
    def __init__(self, states, actions):
        super().__init__()
        self.net = network(states, 2 * actions)

    def forward(self, state, deterministic=False):
        mean, log_std = self.net(state).chunk(2, dim=-1)
        distribution = torch.distributions.Normal(mean, log_std.clamp(-5, 2).exp())
        raw = mean if deterministic else distribution.rsample()
        action = raw.tanh()
        log_prob = (distribution.log_prob(raw) - 2 * (np.log(2) - raw - F.softplus(-2 * raw))).sum(-1)
        return action, log_prob


class SACAgent:
    def __init__(self, states, actions, device='cpu'):
        self.device, self.states, self.actions = device, states, actions
        self.actor = Actor(states, actions).to(device)
        self.q = nn.ModuleList([network(states + actions, 1), network(states + actions, 1)]).to(device)
        self.target = deepcopy(self.q).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=3e-4)
        self.log_alpha = torch.zeros((), device=device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.updates = 0

    def action(self, state, deterministic=False):
        with torch.no_grad():
            action, _ = self.actor(torch.as_tensor(state, dtype=torch.float32, device=self.device)[None], deterministic)
        return action[0].cpu().numpy()

    def update(self, transitions):
        state, action, reward, next_state, done = (
            torch.as_tensor(np.asarray(value), dtype=torch.float32, device=self.device)
            for value in zip(*transitions))
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            next_action, next_log = self.actor(next_state)
            target_input = torch.cat((next_state, next_action), -1)
            target_q = torch.minimum(self.target[0](target_input), self.target[1](target_input)).squeeze(-1)
            backup = reward + .99 * (1 - done) * (target_q - alpha * next_log)
        inputs = torch.cat((state, action), -1)
        q_loss = sum(F.mse_loss(critic(inputs).squeeze(-1), backup) for critic in self.q)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        self.q.requires_grad_(False)
        chosen, log_prob = self.actor(state)
        actor_input = torch.cat((state, chosen), -1)
        actor_loss = (alpha * log_prob - torch.minimum(self.q[0](actor_input), self.q[1](actor_input)).squeeze(-1)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.q.requires_grad_(True)
        alpha_loss = -(self.log_alpha * (log_prob.detach() - self.actions)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            for target, parameter in zip(self.target.parameters(), self.q.parameters()):
                target.mul_(.995).add_(parameter, alpha=.005)
        self.updates += 1
        return {'critic_loss': float(q_loss.detach()), 'actor_loss': float(actor_loss.detach()),
                'alpha': float(self.log_alpha.exp().detach())}

    def export(self):
        return {'algorithm': 'sac', 'states': self.states, 'actions': self.actions, 'updates': self.updates,
                'actor': {key: value.detach().cpu() for key, value in self.actor.state_dict().items()},
                'critics': {key: value.detach().cpu() for key, value in self.q.state_dict().items()},
                'target_critics': {key: value.detach().cpu() for key, value in self.target.state_dict().items()},
                'alpha': float(self.log_alpha.exp().detach())}


def optimize(objective, seed=0, device='cpu', warmup=4, batch_size=16, **_):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    unit = objective.initial.copy()
    agent = SACAgent(len(objective.context(unit)), objective.dim, device)
    replay = []
    while objective.remaining:
        state = objective.context(unit)
        action = rng.uniform(-1, 1, objective.dim) if len(replay) < warmup else agent.action(state)
        candidate = np.clip(unit + .2 * action, 0, 1)
        old_loss = objective.history[-1]['loss']
        objective(candidate)
        next_state = objective.context(candidate)
        reward = getattr(objective, 'last_reward', old_loss - objective.history[-1]['loss'])
        replay.append((state, action, reward, next_state, float(objective.remaining == 0)))
        if len(replay) > 10000:
            replay.pop(0)
        if len(replay) >= 2:
            indices = rng.choice(len(replay), min(batch_size, len(replay)), replace=False)
            agent.update([replay[int(index)] for index in indices])
        unit = candidate
    return agent.export()
