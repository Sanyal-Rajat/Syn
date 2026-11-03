"""DDPG and TD3: explicit devices, small-budget replay and deterministic frozen actors."""
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import Replay, cpu_state, mlp, train


class Actor(nn.Module):
    def __init__(self, states, actions):
        super().__init__()
        self.net = mlp(states, actions)

    def forward(self, state):
        return self.net(state).tanh()


class DeterministicAgent:
    continuous = True

    def __init__(self, states, actions, device, config, twin=False):
        self.states, self.actions, self.device, self.config, self.twin = states, actions, device, config, twin
        self.actor = Actor(states, actions).to(device)
        self.q = nn.ModuleList([mlp(states + actions, 1) for _ in range(2 if twin else 1)]).to(device)
        self.target_actor, self.target_q = deepcopy(self.actor).requires_grad_(False), deepcopy(self.q).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=3e-4)
        self.replay = Replay(config.replay_size)
        self.updates, self.actor_updates, self.warmup = 0, 0, config.warmup

    def action(self, state, rng, step):
        if step < self.warmup:
            return rng.uniform(-1, 1, self.actions).astype('float32')
        with torch.no_grad():
            action = self.actor(torch.as_tensor(state, device=self.device)[None])[0].cpu().numpy()
        return np.clip(action + rng.normal(0, .1, self.actions), -1, 1).astype('float32')

    def observe(self, transition, rng):
        self.replay.add((*transition, .99))
        if len(self.replay.rows) >= 2:
            self.learn(rng)

    def learn(self, rng):
        (state, action, reward, next_state, done, discount), _, _ = self.replay.sample(self.config.batch_size, rng, self.device)
        with torch.no_grad():
            next_action = self.target_actor(next_state)
            if self.twin:
                next_action = (next_action + (torch.randn_like(next_action) * .2).clamp(-.5, .5)).clamp(-1, 1)
            inputs = torch.cat((next_state, next_action), -1)
            target_q = torch.stack([q(inputs).squeeze(-1) for q in self.target_q]).min(0).values
            backup = reward + discount * (1 - done) * target_q
        inputs = torch.cat((state, action), -1)
        loss = sum(F.mse_loss(q(inputs).squeeze(-1), backup) for q in self.q)
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()
        self.updates += 1
        if not self.twin or self.updates % 2 == 0:
            self.q.requires_grad_(False)
            actor_loss = -self.q[0](torch.cat((state, self.actor(state)), -1)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.q.requires_grad_(True)
            self.actor_updates += 1
            with torch.no_grad():
                for target, source in [(self.target_actor, self.actor), (self.target_q, self.q)]:
                    for a, b in zip(target.parameters(), source.parameters()):
                        a.lerp_(b, .005)

    def finish(self, rng):
        pass  # Every eligible transition already performs one critic update.

    def export(self):
        return {'states': self.states, 'actions': self.actions, 'updates': self.updates,
                'actor_updates': self.actor_updates, 'action_kind': 'continuous', 'actor': cpu_state(self.actor)}


def optimize(objective, algorithm, **kwargs):
    return train(objective, lambda s, d, dev, c: DeterministicAgent(s, d, dev, c, algorithm == 'td3'), algorithm, **kwargs)
