import json

import pytest

from lidarpoc.report import comparison_rows, component_means, findings, write_pdf


@pytest.fixture
def experiment_result():
    return {'action': 'experiment', 'selected_arm': 'E1',
            'selection': {'scores': {'E0': 1.4, 'E1': 1.1}, 'primary_metric': 'development_OT_gap_surrogate'},
            'session_intervals': {'E0': {'mean': 1.2, 'ci95': [1., 1.4], 'traces': 2},
                                  'E1': {'mean': .9, 'ci95': [.8, 1.], 'traces': 2}},
            'evaluations': {'0': {'E1': {'traces': [{'sequence_id': 'a', 'conditional_sinkhorn': .2},
                                                    {'sequence_id': 'b', 'conditional_sinkhorn': .6}]}},
                            '1': {'E1': {'traces': [{'sequence_id': 'a', 'conditional_sinkhorn': .4}]}}},
            'compute': {'0': {'E1': {'allocated_total_gpu_hours': .1, 'training_resources': {
                'batch_size_per_process': 4, 'world_size': 1}}, 'E2': {'training_resources': {
                'batch_size_per_process': 40, 'world_size': 1}}}},
            'gates': {'decision': 'pending_task_or_pose_evidence', 'geometry_pass': True,
                      'pose_metrics_supported': False, 'task_transfer': 'not_evaluated',
                      'rl_development_comparison': {'comparator': 'E5_bayesian',
                          'paired_test_session_gain': {'mean': -.08, 'ci95': [-.09, -.07]},
                          'matched_call_and_time_caps': True, 'recommendation': 'no_incremental_value_demonstrated',
                          'methods_removed': False}},
            'task_evaluations': [{'E1': {'512': [{'sequence_id': 'a', 'mean_iou': .4}]}}],
            'limitations': ['<img src="https://invalid.example/never-fetch.png"/> is untrusted result text.']}


def test_analysis_preserves_selection_ci_session_units_and_rl(experiment_result):
    rows = comparison_rows(experiment_result)
    assert rows[1]['ci95'] == [.8, 1.] and rows[1]['traces'] == 2
    assert rows[1]['development'] == 1.1
    text = '\n'.join(findings(experiment_result))
    assert '25.00% lower than E0' in text
    assert '2 independent sessions' in text
    assert 'Positive gain favors SAC' in text and '-0.0800' in text
    assert 'Methods removed: No' in text
    assert 'Equal epochs' in text and 'optimizer-step counts' in text
    # Average .2 and .4 within session a before averaging a with session b.
    assert component_means(experiment_result, 'E1')['conditional_sinkhorn'] == pytest.approx(.45)
    experiment_result['session_intervals']['E0']['mean'] = 0
    assert not any('% lower' in line for line in findings(experiment_result))


@pytest.mark.parametrize('action', ['prepare', 'train', 'generate', 'experiment'])
def test_pdf_export_for_every_workflow_is_valid_and_keeps_json(tmp_path, experiment_result, action):
    reports = {'experiment': experiment_result,
               'prepare': {'action': 'prepare', 'manifest': '/server/prepared/manifest.json',
                           'inventory': [{'split': 'train', 'sequence_id': 'trace', 'frames': 4, 'source_frames': 20}]},
               'train': {'action': 'train', 'selected_profile_on_validation': 'ppo',
                         'evaluations': {'ppo': {'loss': {'mean': .2, 'ci95': [.1, .3], 'traces': 2}}},
                         'training': [{'epoch': 1, 'train_loss': .3, 'validation_loss': .2}]},
               'generate': {'action': 'generate', 'samples': 4, 'geometry': {'new_surface_returns': 0}}}
    source = tmp_path / 'result.json'
    source.write_text(json.dumps(reports[action]))
    before = source.read_bytes()
    target = write_pdf(source)
    assert target == tmp_path / 'result.pdf'
    assert target.read_bytes().startswith(b'%PDF-')
    assert target.read_bytes().rstrip().endswith(b'%%EOF')
    assert source.read_bytes() == before
    assert not list(tmp_path.glob('.result-*'))


def test_pdf_failure_preserves_previous_export_and_source(tmp_path, monkeypatch):
    from reportlab.platypus import SimpleDocTemplate

    source, target = tmp_path / 'result.json', tmp_path / 'result.pdf'
    source.write_text('{"action":"generate","samples":2}')
    target.write_bytes(b'previous PDF')

    def fail(*args, **kwargs):
        raise RuntimeError('render failure')

    monkeypatch.setattr(SimpleDocTemplate, 'build', fail)
    with pytest.raises(RuntimeError, match='render failure'):
        write_pdf(source)
    assert target.read_bytes() == b'previous PDF'
    assert not list(tmp_path.glob('.result-*'))
    with pytest.raises(ValueError, match='overwrite'):
        write_pdf(source, source)
