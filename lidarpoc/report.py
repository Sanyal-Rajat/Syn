"""Standalone JSON-to-PDF results reporting; no model stack or network access."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import math
import os
from pathlib import Path
from statistics import mean
import tempfile


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def format_value(value):
    if value is None:
        return 'Not available'
    if isinstance(value, bool):
        return 'Yes' if value else 'No'
    if isinstance(value, float):
        return f'{value:.4f}' if math.isfinite(value) else 'Not available'
    if isinstance(value, list):
        return '[' + ', '.join(format_value(item) for item in value) + ']'
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def comparison_rows(report):
    """Use exported session intervals, preserving development-based selection."""
    if report.get('action') == 'experiment':
        scores = report.get('selection', {}).get('scores', {})
        return [{'name': name, 'development': scores.get(name), **stats}
                for name, stats in report.get('session_intervals', {}).items()]
    if report.get('action') == 'train':
        return [{'name': name, **value['loss']} for name, value in report.get('evaluations', {}).items()]
    return []


def component_means(report, arm):
    """Average repeated seeds within sessions before averaging sessions."""
    sessions = defaultdict(lambda: defaultdict(list))
    for arms in report.get('evaluations', {}).values():
        for trace in arms.get(arm, {}).get('traces', []):
            for key, value in trace.items():
                if number(value):
                    sessions[trace['sequence_id']][key].append(value)
    keys = {key for values in sessions.values() for key in values}
    return {key: mean(mean(values[key]) for values in sessions.values() if key in values) for key in keys}


def findings(report):
    action = report.get('action')
    selected = report.get('selected_arm', report.get('selected_profile_on_validation'))
    rows = {row['name']: row for row in comparison_rows(report)}
    text = []
    if selected:
        text.append(f'Selected on development/validation data: {selected}. Test scores did not choose the selection.')
    if selected in rows:
        score = rows[selected]
        text.append(f"Selected test score: {format_value(score.get('mean'))}; 95% CI "
                    f"{format_value(score.get('ci95'))}; {score.get('traces', 'unknown')} independent sessions.")
        baseline = rows.get('E0' if action == 'experiment' else 'training_resampling', {})
        a, b = baseline.get('mean'), score.get('mean')
        if number(a) and a > 0 and number(b):
            text.append(f"The selected mean discrepancy is {(a - b) / a * 100:.2f}% lower than "
                        f"{baseline['name']}. This is a surrogate-score comparison, not an accuracy gain.")
    if action == 'experiment':
        baseline_costs = [arms['E0'] for arms in report.get('compute', {}).values() if 'E0' in arms]
        if baseline_costs and all(cost.get('epochs') == 0 for cost in baseline_costs):
            text.append('E0 records zero additional training epochs. A gain over E0 alone does not establish '
                        'the value of OT losses or RL beyond the trained E1 baseline.')
        components = component_means(report, selected)
        actual, reference = components.get('return_probability'), components.get('reference_return_probability')
        if number(actual) and number(reference):
            text.append(f'Selected generated/real return probabilities: {actual:.4f} / {reference:.4f}; '
                        f'signed difference {(actual - reference) * 100:.2f} percentage points. '
                        'Aggregate agreement does not imply matching each range/ring context.')
    gates = report.get('gates', {})
    if gates:
        text.append(f"Evidence decision: {gates.get('decision', 'not reported')}. "
                    f"Geometry: {format_value(gates.get('geometry_pass'))}; real pose support: "
                    f"{format_value(gates.get('pose_metrics_supported'))}; task transfer: "
                    f"{gates.get('task_transfer', 'not reported')}.")
        rl = gates.get('rl_development_comparison', {})
        if rl:
            gain = rl.get('paired_test_session_gain', {})
            text.append(f"SAC versus {rl.get('comparator')}: paired test gain "
                        f"{format_value(gain.get('mean'))}, 95% CI {format_value(gain.get('ci95'))}. "
                        f"Positive gain favors SAC. Recommendation: {rl.get('recommendation')}. "
                        f"Matched call/time caps: {format_value(rl.get('matched_call_and_time_caps'))}. "
                        f"Methods removed: {format_value(rl.get('methods_removed'))}.")
        for method, value in gates.get('rl_comparisons', {}).items():
            if method == 'sac' and rl:
                continue
            gain = value.get('paired_test_session_gain', {})
            text.append(f"{method} versus {value.get('comparator')}: paired test gain "
                        f"{format_value(gain.get('mean'))}, 95% CI {format_value(gain.get('ci95'))}. "
                        f"Positive gain favors {method}. Recommendation: {value.get('recommendation')}.")
    batches = {cost.get('training_resources', {}).get('batch_size_per_process')
               for arms in report.get('compute', {}).values() for cost in arms.values()}
    batches.discard(None)
    if len(batches) > 1:
        text.append(f'Training batches differ across arms: {sorted(batches)} per process. Equal epochs '
                    'need not give equal optimizer-step counts; this can confound the ablation comparison.')
    if action == 'prepare':
        inventory = report.get('inventory', [])
        text.append(f"Prepared {len(inventory)} traces and {sum(row.get('frames', 0) for row in inventory)} frames. "
                    f"Manifest: {report.get('manifest', 'not reported')}.")
    elif action == 'generate':
        text.append(f"Generated {report.get('samples', 0)} LiDAR windows; calibration profile: "
                    f"{report.get('calibration_profile', 'not reported')}.")
        if report.get('real_scene'):
            text.append(f"Exported conditioning scene: {report['real_scene']}. "
                        'This contains the input geometry, not an independently generated real-world scene.')
    return text


def score_chart(rows, selected, width):
    """Vector bars and the original confidence intervals; no plotting dependency."""
    from reportlab.graphics.shapes import Drawing, Line, Rect, String
    from reportlab.lib.colors import HexColor

    rows = sorted((row for row in rows if number(row.get('mean'))), key=lambda row: row['mean'])
    height, left, right = 30 * len(rows) + 45, 112, width - 55
    drawing = Drawing(width, height)
    endpoints = [value for row in rows for value in [row['mean'], *(row.get('ci95') or [])] if number(value)]
    low, high = min([0, *endpoints]), max([0, *endpoints])
    span = high - low or 1
    scale = lambda value: left + (value - low) / span * (right - left)
    for tick in range(5):
        value = low + span * tick / 4
        x = scale(value)
        drawing.add(Line(x, 30, x, height - 8, strokeColor=HexColor('#dde3ea')))
        drawing.add(String(x, 12, f'{value:.2f}', fontSize=8, textAnchor='middle'))
    for index, row in enumerate(rows):
        y = height - 24 - index * 30
        color = HexColor('#12805c' if row['name'] == selected else '#426d9d')
        x, zero = scale(row['mean']), scale(0)
        drawing.add(String(left - 8, y, row['name'], fontSize=9, textAnchor='end'))
        drawing.add(Rect(min(zero, x), y - 4, abs(x - zero), 13, fillColor=color, strokeColor=None))
        interval = row.get('ci95') or []
        if len(interval) == 2 and all(number(value) for value in interval):
            a, b = map(scale, interval)
            drawing.add(Line(a, y + 2, b, y + 2, strokeWidth=1.2))
            for end in (a, b):
                drawing.add(Line(end, y - 2, end, y + 6, strokeWidth=1.2))
        drawing.add(String(right + 8, y, f"{row['mean']:.3f}", fontSize=8))
    return drawing


def write_pdf(result_path, output=None):
    """Render any completed result JSON atomically; retain the JSON as source data."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    result_path = Path(result_path)
    source = result_path.read_bytes()
    report = json.loads(source)
    if not isinstance(report, dict):
        raise ValueError('A results report must be a JSON object')
    output = Path(output) if output else result_path.with_suffix('.pdf')
    if output.resolve() == result_path.resolve():
        raise ValueError('PDF output must not overwrite the source JSON')
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles['Normal'].fontSize = 9
    styles['Normal'].leading = 13
    styles['Normal'].spaceAfter = 6
    styles['BodyText'].fontSize = 8
    styles['BodyText'].leading = 11
    story = []
    width = A4[0] - 84

    def paragraph(value, style='Normal'):
        # Result strings are data, never Paragraph XML, links, images or instructions.
        return Paragraph(escape(format_value(value)), styles[style])

    def section(title):
        story.append(paragraph(title, 'Heading2'))

    def table(headers, rows, widths=None):
        if not rows:
            return
        data = [[paragraph(cell, 'BodyText') for cell in row] for row in [headers, *rows]]
        grid = Table(data, colWidths=widths or [width / len(headers)] * len(headers),
                     repeatRows=1, hAlign='LEFT')
        grid.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e5edf5')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f7fa')]),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ('TOPPADDING', (0, 0), (-1, -1), 7),
            ('LINEBELOW', (0, 0), (-1, 0), .5, colors.HexColor('#9aaabd')),
        ]))
        story.extend([grid, Spacer(1, 9)])

    def key_values(values):
        table(['Field', 'Value'], [[key.replace('_', ' '), value] for key, value in values.items()], [170, width - 170])

    story.append(paragraph('Pandar128 LiDAR results', 'Title'))
    story.append(paragraph(f"Operation: {report.get('action', 'unknown')} | "
                           f"Generated UTC: {datetime.now(timezone.utc):%Y-%m-%d %H:%M}"))
    story.append(paragraph(f'Source: {result_path.name} | SHA-256: {hashlib.sha256(source).hexdigest()}', 'BodyText'))
    section('Findings')
    for line in findings(report):
        story.append(paragraph(line))
    rows = comparison_rows(report)
    selected = report.get('selected_arm', report.get('selected_profile_on_validation'))
    if rows:
        section('Held-out comparison')
        story.append(paragraph('Lower discrepancy is better. The exported 95% confidence intervals use independent '
                               'sessions; repeated experiment seeds are averaged within sessions, not counted as new sessions. '
                               'OT gaps are composite surrogate scores, not metres or task accuracy.'))
        headers = ['Arm / method', 'Development score', 'Test mean', '95% CI', 'Sessions']
        table(headers, [[row['name'], row.get('development'), row.get('mean'),
                         row.get('ci95'), row.get('traces')] for row in rows], [105, 85, 65, 180, width - 435])
        if report.get('selection', {}).get('primary_metric'):
            story.append(paragraph(f"Selection metric: {report['selection']['primary_metric']}. "
                                   'Development selection scores are exported as minimized scores; task-IoU scores are negated.'))
        for start in range(0, len(rows), 14):
            story.append(score_chart(rows[start:start + 14], selected, width))
        story.append(paragraph('Green marks the development/validation selection. Error bars show the exported 95% CIs.', 'BodyText'))
    if report.get('action') == 'experiment':
        section('OT components and return statistics')
        story.append(paragraph('Ablations: E0 adds no diffusion training; E1 trains diffusion; E2 adds conditional OT; '
                               'E3 adds return-mass OT; E4 adds physics/temporal constraints; E5 controls a frozen E4 generator.'))
        components = {row['name']: component_means(report, row['name']) for row in rows}
        table(['Arm', 'Conditional OT', 'Unbalanced OT', 'Return mass L1', 'Return probability', 'Real return probability'],
              [[name, *(values.get(key) for key in ('conditional_sinkhorn', 'unbalanced_ot', 'return_mass_l1',
                                                   'return_probability', 'reference_return_probability'))]
               for name, values in components.items()])
        table(['Arm', 'Range W1 / max range', 'Intensity W1', 'Temporal discrepancy', 'Unsupported contexts'],
              [[name, *(values.get(key) for key in ('range_wasserstein_normalized', 'intensity_wasserstein',
                                                   'temporal_discrepancy', 'unsupported_contexts'))]
               for name, values in components.items()])
        story.append(paragraph('Component values average repeated seeds within sessions, then sessions equally. '
                               'The composite gap also penalizes unsupported OT bins; components need not all improve.'))
        section('Evidence gates')
        key_values({key: value for key, value in report.get('gates', {}).items() if not isinstance(value, dict)})
        tasks = defaultdict(lambda: defaultdict(list))
        for seed_result in report.get('task_evaluations', []):
            for arm, budgets in seed_result.items():
                for budget, traces in budgets.items():
                    for trace in traces:
                        if number(trace.get('mean_iou')):
                            tasks[(arm, budget)][trace['sequence_id']].append(trace['mean_iou'])
        if tasks:
            section('Downstream point-segmentation probe')
            table(['Arm', 'Label budget', 'Mean IoU', 'Sessions'],
                  [[arm, budget, mean(mean(values) for values in sessions.values()), len(sessions)]
                   for (arm, budget), sessions in tasks.items()])
            story.append(paragraph('IoU averages seeds within sessions, then sessions equally; higher is better. '
                                   'These descriptive means have no new confidence intervals. See the exported paired task gate.'))
        section('Compute and assigned resources')
        costs = defaultdict(list)
        for arms in report.get('compute', {}).values():
            for name, cost in arms.items():
                costs[name].append(cost)
        table(['Arm', 'Seeds', 'Allocated GPU h', 'Training batch / process', 'Training processes'],
              [[name, len(values), sum(c.get('allocated_total_gpu_hours', 0) for c in values),
                sorted({c['training_resources']['batch_size_per_process'] for c in values
                        if c.get('training_resources', {}).get('batch_size_per_process') is not None}) or None,
                sorted({c['training_resources']['world_size'] for c in values
                        if c.get('training_resources', {}).get('world_size') is not None}) or None]
               for name, values in costs.items()])
        learned = [(name, values) for name, values in costs.items() if any('learning_updates' in c for c in values)]
        if learned:
            table(['Controller', 'Generation calls', 'Learning updates', 'Fitting transitions'],
                  [[name, sum(c.get('generation_calls', 0) for c in values),
                    sum(c.get('learning_updates', 0) for c in values),
                    sum(c.get('training_transitions') or 0 for c in values) if
                    all(c.get('training_transitions') is not None for c in values) else None]
                   for name, values in learned])
        story.append(paragraph('GPU hours are elapsed allocated device time, not measured GPU busy time. '
                               'E5 arms include shared generator training; do not sum these rows as the experiment total. '
                               'Resource snapshots describe assigned workers and do not establish the total host GPU count. '
                               'Host-wide utilization history is not included in result.json.'))
    elif report.get('action') == 'train':
        section('Training history')
        table(['Epoch', 'Training loss', 'Validation loss'],
              [[row.get(key) for key in ('epoch', 'train_loss', 'validation_loss')] for row in report.get('training', [])])
    elif report.get('action') == 'prepare':
        section('Prepared inventory')
        table(['Split', 'Trace', 'Frames', 'Source frames'],
              [[row.get(key) for key in ('split', 'sequence_id', 'frames', 'source_frames')] for row in report.get('inventory', [])],
              [60, width - 180, 60, 60])
    section('Provenance and scope')
    excluded = {'preview', 'training', 'evaluations', 'development', 'compute', 'profiles', 'inventory',
                'sensor_meta', 'session_intervals', 'selection', 'gates', 'limitations', 'task_evaluations',
                'data_efficiency', 'license_evidence'}
    key_values({key: value for key, value in report.items() if key not in excluded and not isinstance(value, (dict, list))})
    if report.get('action') == 'generate' and report.get('geometry'):
        key_values(report['geometry'])
    for line in report.get('limitations', []):
        story.append(paragraph(line))
    story.append(paragraph('This PDF summarizes the result JSON. Full per-trace arrays, controller histories, '
                           'task-budget records and machine-readable provenance remain in the JSON/artifact bundle.'))

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#637081'))
        canvas.drawString(42, 25, 'Syn | Pandar128 results')
        canvas.drawRightString(A4[0] - 42, 25, f'Page {document.page}')
        canvas.restoreState()

    with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.result-', suffix='.pdf', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        document = SimpleDocTemplate(str(temporary), pagesize=A4, rightMargin=42, leftMargin=42,
                                     topMargin=38, bottomMargin=42, title='Pandar128 LiDAR results', author='Syn')
        document.build(story, onFirstPage=footer, onLaterPages=footer)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert an existing Syn result JSON to PDF without rerunning a job.')
    parser.add_argument('result', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(write_pdf(args.result, args.output))
