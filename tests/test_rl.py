import io

import numpy as np
import pytest
import torch

from lidarpoc.config import RunConfig
from lidarpoc.methods.registry import CALIBRATION_METHODS, CONTROLLER_METHODS, optimize
from lidarpoc.rl import BudgetExhausted, RL_METHODS
from lidarpoc.rl.common import NStep, Replay
from lidarpoc.rl.config import RLConfig
from lidarpoc.rl.deterministic import DeterministicAgent
from lidarpoc.rl.policy import apply_policy, load_actor
from lidarpoc.rl.q_learning import categorical_projection

NEW_METHODS = tuple(name for name in RL_METHODS if name not in ('ppo', 'sac'))


class Objective:
    dim = 4
    initial = np.full(4, .5, dtype='float32')

    def __init__(self, budget=6, stop=None):
        self.budget, self.stop, self.history = budget, stop, []

    @property
    def remaining(self):
        return self.budget - len(self.history)

    def observation(self, unit):
        return np.r_[unit, np.zeros(5)].astype('float32')

    def __call__(self, unit):
        if self.stop is not None and len(self.history) >= self.stop:
            raise BudgetExhausted('time cap')
        assert self.remaining > 0 and unit.shape == (4,)
        assert np.isfinite(unit).all() and (unit >= 0).all() and (unit <= 1).all()
        loss = float(np.square(unit - .2).sum())
        self.history.append({'loss': loss})
        return loss


@pytest.mark.parametrize('method', NEW_METHODS)
def test_new_methods_learn_with_small_budget_and_roundtrip_frozen_policies(method):
    torch.set_num_threads(1)
    objective = Objective()
    policy = optimize(method, objective, seed=7, device='cpu')
    assert len(objective.history) == 6 and policy['transitions'] == 5
    assert policy['updates'] > 0
    if method in ('a2c', 'reinforce', 'n_step_actor_critic'):
        assert policy['resolved_warmup'] == 0
    if method in ('ddpg', 'td3'):
        assert policy['actor_updates'] > 0
    buffer = io.BytesIO()
    torch.save(policy, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=True)
    actor = load_actor(restored)
    assert not any(parameter.requires_grad for parameter in actor.parameters())
    context = objective.observation(objective.initial)
    first = apply_policy(restored, actor, context, objective.initial)
    second = apply_policy(restored, actor, context, objective.initial)
    assert np.array_equal(first, second) and first.shape == (4,)
    assert (first >= 0).all() and (first <= 1).all()
    assert len(objective.history) == 6  # Frozen inference cannot spend fitting calls.


@pytest.mark.parametrize('method', NEW_METHODS)
def test_budget_interrupt_still_exports_policy_and_repeated_seed_is_reproducible(method):
    torch.set_num_threads(1)
    one, two = Objective(stop=4), Objective(stop=4)
    a = optimize(method, one, seed=3, device='cpu')
    b = optimize(method, two, seed=3, device='cpu')
    assert len(one.history) == len(two.history) == 4
    assert a['updates'] > 0
    state_key = 'actor' if a['action_kind'] == 'continuous' else 'network'
    assert all(torch.equal(value, b[state_key][name]) for name, value in a[state_key].items())


def test_n_step_terminal_flush_has_exact_discounted_returns_and_no_episode_leakage():
    buffer = NStep(3, gamma=.5)
    assert buffer.add((0, 0, 1., 1, False)) == []
    assert buffer.add((1, 0, 2., 2, False)) == []
    rows = buffer.add((2, 0, 3., 3, True))
    assert [row[2] for row in rows] == [2.75, 3.5, 3.]
    assert [row[5] for row in rows] == [.125, .25, .5]
    assert all(row[3] == 3 and row[4] for row in rows)
    assert not buffer.pending
    buffer.add((10, 0, 4., 11, False))
    assert buffer.finish() == [(10, 0, 4., 11, True, .5)]


def test_categorical_projection_preserves_mass_on_exact_atoms_and_clipped_bounds():
    support = torch.linspace(-20, 20, 51)
    probabilities = torch.rand(3, 51)
    probabilities /= probabilities.sum(-1, keepdim=True)
    projected = categorical_projection(probabilities, torch.zeros(3), torch.ones(3), torch.zeros(3), support)
    assert torch.allclose(projected, probabilities, atol=2e-6)
    clipped = categorical_projection(probabilities, torch.tensor([-100., 100., 0.]), torch.ones(3), torch.ones(3), support)
    assert torch.allclose(clipped.sum(-1), torch.ones(3))
    assert clipped[0, 0] == pytest.approx(1.) and clipped[1, -1] == pytest.approx(1.)
    assert clipped[2, 25] == pytest.approx(1.)


def test_prioritized_replay_prefers_high_priority_and_returns_finite_weights():
    memory = Replay(2, prioritized=True)
    for value in (0., 1.):
        memory.add((np.array([value]), 0, value, np.array([value]), False, .99))
    memory.update([0, 1], [1e-4, 100.])
    rng, counts = np.random.default_rng(0), [0, 0]
    for _ in range(100):
        _, weights, indices = memory.sample(2, rng, 'cpu')
        for index in indices:
            counts[index] += 1
        assert torch.isfinite(weights).all() and (weights > 0).all() and weights.max() <= 1
    assert counts[1] > 190
    memory.add((np.array([2.]), 0, 2., np.array([2.]), False, .99))
    assert len(memory.rows) == len(memory.priorities) == 2


@pytest.mark.parametrize('twin', [False, True])
def test_deterministic_actor_delay_critics_and_targets(twin):
    torch.set_num_threads(1)
    agent = DeterministicAgent(9, 4, 'cpu', RLConfig(), twin=twin)
    rng = np.random.default_rng(0)
    row = (np.zeros(9, dtype='float32'), np.zeros(4, dtype='float32'), 1., np.ones(9, dtype='float32'), False)
    actor = [p.detach().clone() for p in agent.actor.parameters()]
    critic = [p.detach().clone() for p in agent.q.parameters()]
    agent.observe(row, rng)
    agent.observe(row, rng)
    assert agent.updates == 1 and agent.actor_updates == (0 if twin else 1)
    assert any(not torch.equal(a, b) for a, b in zip(critic, agent.q.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(actor, agent.actor.parameters())) == twin
    agent.observe(row, rng)
    assert agent.actor_updates == (1 if twin else 2)
    assert not any(p.requires_grad for p in agent.target_q.parameters())
    assert not any(p.requires_grad for p in agent.target_actor.parameters())


def test_registry_configuration_and_compatibility_imports():
    from lidarpoc.methods.ppo import PPOAgent as original_ppo
    from lidarpoc.methods.sac import SACAgent as original_sac
    from lidarpoc.rl.ppo import PPOAgent
    from lidarpoc.rl.sac import SACAgent

    assert original_ppo is PPOAgent and original_sac is SACAgent
    config = RunConfig(manifest='/server/manifest.json', methods=list(CALIBRATION_METHODS),
                       experiment={'controllers': list(CONTROLLER_METHODS)})
    assert set(NEW_METHODS) <= set(config.methods) & set(config.experiment.controllers)
    with pytest.raises(ValueError):
        RunConfig(manifest='x', methods=['disabled_sac'])
    with pytest.raises(ValueError):
        RLConfig(value_min=20, value_max=-20)


def test_paired_rl_comparisons_preserve_session_units_and_check_each_pair_budget():
    from lidarpoc.ot.config import ExperimentConfig
    from lidarpoc.rl.comparison import compare

    candidates = ['E5_fixed', 'E5_sac', 'E5_td3']
    scores = {'E5_fixed': 1., 'E5_sac': .8, 'E5_td3': .6}
    tests = {seed: {name: {'traces': [{'sequence_id': session, 'gap': value} for session in ('a', 'b')]}
                    for name, value in scores.items()} for seed in (0, 1, 2)}
    costs = {seed: {name: {'generation_calls': 6, 'elapsed_s': 301 if name == 'E5_td3' else 1}
                    for name in candidates} for seed in (0, 1, 2)}
    result = compare(candidates, 'E5_fixed', scores.get, [], tests, costs, 6, ExperimentConfig())
    assert result['sac']['material'] and result['sac']['matched_call_and_time_caps']
    assert not result['td3']['material'] and not result['td3']['matched_call_and_time_caps']
    assert result['sac']['paired_test_session_gain']['traces'] == 2
    assert result['sac']['paired_test_session_gain']['mean'] == pytest.approx(.2)
