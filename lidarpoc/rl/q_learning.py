"""DQN variants adapted from off_policy_dqn_RL_Agents.py; shared replay and C51 projection."""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from .common import NStep, Replay, cpu_state, mlp, train


class QNetwork(nn.Module):
    def __init__(self, states, actions, *, dueling=False, atoms=1, value_min=-20., value_max=20.):
        super().__init__()
        self.actions, self.atoms, self.dueling = actions, atoms, dueling
        self.shared = mlp(states, 64)
        self.advantage = nn.Linear(64, actions * atoms)
        self.value = nn.Linear(64, atoms) if dueling else None
        self.register_buffer('support', torch.linspace(value_min, value_max, atoms))

    def forward(self, state):
        hidden = self.shared(state)
        result = self.advantage(hidden).view(-1, self.actions, self.atoms)
        if self.value is not None:
            result = self.value(hidden)[:, None] + result - result.mean(1, keepdim=True)
        return result.softmax(-1) if self.atoms > 1 else result.squeeze(-1)

    def values(self, state):
        output = self(state)
        return (output * self.support).sum(-1) if self.atoms > 1 else output


def categorical_projection(probabilities, rewards, discounts, dones, support):
    """Project Bellman targets, retaining mass when a target lies exactly on an atom."""
    target = rewards[:, None] + discounts[:, None] * (1 - dones[:, None]) * support
    target = target.clamp(support[0], support[-1])
    spacing = (support[-1] - support[0]) / (len(support) - 1)
    position = ((target - support[0]) / spacing).clamp(0, len(support) - 1)
    lower, upper = position.floor().long(), position.ceil().long()
    projected = torch.zeros_like(probabilities)
    projected.scatter_add_(1, lower, probabilities * (upper - position + (lower == upper)))
    projected.scatter_add_(1, upper, probabilities * (position - lower))
    return projected


class QAgent:
    continuous = False

    def __init__(self, states, dimensions, device, config, algorithm):
        self.states, self.actions, self.device, self.config = states, 2 * dimensions + 1, device, config
        self.double = algorithm in ('enhanced_dqn', 'n_step_dqn', 'rainbow')
        self.network_config = {'dueling': algorithm in ('enhanced_dqn', 'rainbow'),
                               'atoms': 51 if algorithm in ('c51', 'rainbow') else 1,
                               'value_min': config.value_min, 'value_max': config.value_max}
        self.q = QNetwork(states, self.actions, **self.network_config).to(device)
        self.target = deepcopy(self.q).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=5e-4)
        self.replay = Replay(config.replay_size, algorithm in ('enhanced_dqn', 'rainbow'))
        self.nstep = NStep(config.n_steps if algorithm in ('n_step_dqn', 'rainbow') else 1)
        self.updates, self.warmup = 0, config.warmup

    def action(self, state, rng, step):
        epsilon = max(.05, .3 * .98 ** step)
        if step < self.warmup or rng.random() < epsilon:
            return int(rng.integers(self.actions))
        with torch.no_grad():
            return int(self.q.values(torch.as_tensor(state, device=self.device)[None]).argmax(-1).item())

    def observe(self, transition, rng):
        for row in self.nstep.add(transition):
            self.replay.add(row)
        if len(self.replay.rows) >= 2:
            self.learn(rng)

    def learn(self, rng):
        (state, action, reward, next_state, done, discount), weights, indices = self.replay.sample(
            self.config.batch_size, rng, self.device, beta=min(1., .4 + .0006 * self.updates))
        batch = torch.arange(len(state), device=self.device)
        action = action.long()
        with torch.no_grad():
            next_action = (self.q if self.double else self.target).values(next_state).argmax(-1)
            target_output = self.target(next_state)[batch, next_action]
        output = self.q(state)[batch, action]
        if self.q.atoms > 1:
            with torch.no_grad():
                projected = categorical_projection(target_output, reward, discount, done, self.q.support)
            errors = -(projected * output.clamp_min(1e-8).log()).sum(-1)
        else:
            backup = reward + discount * (1 - done) * target_output
            errors = F.smooth_l1_loss(output, backup, reduction='none')
        loss = (weights * errors).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.)
        self.optimizer.step()
        if self.replay.prioritized:
            priority = errors if self.q.atoms > 1 else (backup - output).abs()
            self.replay.update(indices, priority.detach().cpu().numpy())
        with torch.no_grad():
            for target, source in zip(self.target.parameters(), self.q.parameters()):
                target.lerp_(source, .005)
        self.updates += 1

    def finish(self, rng):
        pending = self.nstep.finish()
        for row in pending:
            self.replay.add(row)
        if pending and len(self.replay.rows) >= 2:
            self.learn(rng)

    def export(self):
        return {'states': self.states, 'actions': self.actions, 'updates': self.updates,
                'action_kind': 'discrete', 'network': cpu_state(self.q), 'network_config': self.network_config,
                'n_steps': self.nstep.steps, 'prioritized_replay': self.replay.prioritized, 'double_q': self.double}


def optimize(objective, algorithm, **kwargs):
    return train(objective, lambda s, d, dev, c: QAgent(s, d, dev, c, algorithm), algorithm, **kwargs)
