"""Paired session comparisons for frozen RL controllers, selected without test fitting."""
from collections import defaultdict

import numpy as np

from . import RL_METHODS
from ..eval.domain_metrics import bootstrap


def compare(candidates, baseline, score, task_tests, tests, computations, budget, settings):
    if not baseline:
        return {}
    results = {}
    for method in (*RL_METHODS, 'contextual_bandit'):
        name = f'E5_{method}'
        if name not in candidates:
            continue
        by_session = defaultdict(list)
        if settings.downstream:
            for rows in task_tests:
                for label_budget, values in rows[name].items():
                    original = {row['sequence_id']: row['mean_iou'] for row in rows[baseline][label_budget]}
                    for row in values:
                        by_session[row['sequence_id']].append(row['mean_iou'] - original[row['sequence_id']])
        else:
            for rows in tests.values():
                original = {row['sequence_id']: row['gap'] for row in rows[baseline]['traces']}
                for row in rows[name]['traces']:
                    by_session[row['sequence_id']].append(original[row['sequence_id']] - row['gap'])
        paired = bootstrap([float(np.mean(values)) for values in by_session.values()])
        metered = [cost for rows in computations.values() for arm, cost in rows.items() if arm in (name, baseline)]
        matched = bool(metered) and all(row['generation_calls'] == budget and
                                        row['elapsed_s'] <= settings.controller_seconds for row in metered)
        improvement = score(baseline) - score(name)
        material = improvement > settings.rl_min_improvement and paired['ci95'][0] > settings.rl_min_improvement and matched
        results[method] = {'comparator': baseline, 'primary_metric_improvement': improvement,
                           'material': material, 'paired_test_session_gain': paired,
                           'matched_call_and_time_caps': matched,
                           'recommendation': 'retain_as_experimental' if material else 'no_incremental_value_demonstrated',
                           'methods_removed': False}
    return results
