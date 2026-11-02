"""Explicit research decisions; absent task/pose evidence can never imply success."""
import numpy as np

from .domain_metrics import bootstrap


def decisions(evaluations, task_results, selected, settings):
    geometry = all(result['geometry_pass'] for result in evaluations)
    pose = all(result['pose_supported_windows'] > 0 and result.get('real_pose_supported_windows', 0) > 0 for result in evaluations)
    grouped = {}
    for seed_result in task_results:
        baseline = seed_result.get('E0', {})
        candidate = seed_result.get(selected, {})
        for budget, rows in candidate.items():
            if budget not in baseline:
                continue
            a = {row['sequence_id']: row['mean_iou'] for row in baseline[budget]}
            for row in rows:
                if row['sequence_id'] in a:
                    grouped.setdefault((budget, row['sequence_id']), []).append(row['mean_iou'] - a[row['sequence_id']])
    # Bootstrap independent sessions after averaging repeated seeds/label budgets.
    sessions = {}
    for (_, identity), values in grouped.items():
        sessions.setdefault(identity, []).extend(values)
    gain = bootstrap([float(np.mean(values)) for values in sessions.values()]) if sessions else None
    task = 'not_evaluated' if gain is None else ('improved' if gain['ci95'][0] > settings.task_min_improvement else 'not_demonstrated')
    if not geometry:
        decision = 'reject_geometry_failure'
    elif task == 'not_demonstrated':
        decision = 'reject_transfer_hypothesis_at_tested_budget'
    elif task == 'not_evaluated' or not pose:
        decision = 'pending_task_or_pose_evidence'
    else:
        decision = 'supported_for_this_probe_and_test_sessions'
    return {'decision': decision, 'geometry_pass': geometry, 'pose_metrics_supported': pose,
            'task_transfer': task, 'paired_session_task_gain': gain,
            'commercial_rights_audit': 'pending', 'rl_methods_retained': True,
            'scope': 'ray-space generation and a small point-segmentation probe; not a release certification'}
