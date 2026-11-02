import cma


def optimize(objective, seed=0, **_):
    strategy = cma.CMAEvolutionStrategy(objective.initial.tolist(), .2,
                                       {'bounds': [0, 1], 'seed': seed + 1, 'verbose': -9})
    while objective.remaining:
        candidates = strategy.ask()
        values = []
        for candidate in candidates[:objective.remaining]:
            values.append(objective(candidate))
        if len(values) != len(candidates):
            break
        strategy.tell(candidates, values)
