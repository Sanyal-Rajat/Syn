"""Plain-language comparison summaries and shareable HTML; no model/UI dependencies."""
import base64
from html import escape
import math
from numbers import Real
from pathlib import PurePosixPath
import re

LABELS = {'real': 'Real reference', 'untrained': 'Without training', 'trained': 'With training'}
COLORS = {'real': '#94a3b8', 'untrained': '#d97706', 'trained': '#0284c7'}


def number(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def image_name(value):
    if not isinstance(value, str) or PurePosixPath(value).name != value:
        raise ValueError('Expected a camera image filename, without directories')
    if PurePosixPath(value).suffix.lower() not in ('.jpg', '.jpeg', '.png'):
        raise ValueError('Camera images must be PNG or JPEG files')
    return value


def paired_arms(report):
    comparison = report.get('comparison', {})
    if not isinstance(comparison, dict):
        return None
    arms = comparison.get('arms', {})
    if report.get('action') != 'generate' or not isinstance(arms, dict) or set(arms) != {'untrained', 'trained'}:
        return None
    epoch = comparison.get('checkpoint_epoch')
    if not number(epoch) or epoch < 1:
        return None
    if any(not isinstance(value, str) or PurePosixPath(value).name != value
            or PurePosixPath(value).suffix != '.npz' for value in arms.values()):
        return None
    if len(set(arms.values())) != 2:
        return None
    return arms


def summary(report):
    """Use measured overlap and exact 1D range W1, with shared frame support per arm."""
    result = {'paired': bool(paired_arms(report)), 'overlap': None, 'wasserstein': None,
              'overlap_frames': 0, 'wasserstein_frames': 0, 'frame_count': 0}
    frames = report.get('pixel_analysis', {}).get('frames', [])
    legacy = not frames and result['paired']
    if legacy:
        frames = report['comparison'].get('frames', [])
    result['frame_count'] = len(frames)
    arms = ('untrained', 'trained') if result['paired'] else sorted({
        pair['candidate_b'] for frame in frames for pair in frame.get('pairs', {}).values()
        if pair.get('reference_a') == 'real' and isinstance(pair.get('candidate_b'), str)
        and re.fullmatch(r'sample_\d+', pair['candidate_b'])})
    if not arms:
        return result
    shared, union = {arm: 0 for arm in arms}, {arm: 0 for arm in arms}
    distances = {arm: [] for arm in arms}
    for frame in frames:
        overlap, wasserstein = {}, {}
        for arm in shared:
            key = arm + '_vs_real'
            pair = frame.get('pairs', {}).get(key, {})
            pixels = pair.get('range_pixel', {})
            a, b = pixels.get('shared_pixels'), pixels.get('union_pixels')
            verified = legacy or (pair.get('reference_a') == 'real' and pair.get('candidate_b') == arm)
            if (verified
                    and number(a) and number(b) and 0 <= a <= b and b > 0):
                overlap[arm] = a, b
            space = frame.get('transport', {}).get('spaces', {}).get('radial_range', {})
            record = space.get('pairs', {}).get(key, {})
            value = record.get('values', {}).get('wasserstein_1')
            if (verified and space.get('kind') == 'scalar' and record.get('units', {}).get('wasserstein_1') == 'm'
                    and number(value) and value >= 0 and number(record.get('count_a'))
                    and number(record.get('count_b')) and record['count_a'] > 0 and record['count_b'] > 0):
                wasserstein[arm] = value
        if len(overlap) == len(arms):
            for arm, (a, b) in overlap.items():
                shared[arm] += a
                union[arm] += b
            result['overlap_frames'] += 1
        if len(wasserstein) == len(arms):
            for arm, value in wasserstein.items():
                distances[arm].append(value)
            result['wasserstein_frames'] += 1
    if result['overlap_frames']:
        result['overlap'] = {'real': 100., **{arm: shared[arm] / union[arm] * 100 for arm in shared}}
    if result['wasserstein_frames']:
        result['wasserstein'] = {'real': 0., **{arm: sum(values) / len(values) for arm, values in distances.items()}}
    return result


def chart_rows(values, unit):
    """Share labels, colours and displayed values between native and exported charts."""
    rows = []
    palette = ('#d97706', '#0284c7', '#059669', '#9333ea', '#db2777')
    for arm, value in values.items():
        label = LABELS.get(arm)
        if label is None:
            label = f'Synthetic scan {int(arm.removeprefix("sample_")) + 1}'
        rows.append({'variant': label, 'value': value,
                     'display': f'{value:.1f}%' if unit == '%' else f'{value:.3g} m',
                     'color': COLORS.get(arm, palette[(len(rows) - 1) % len(palette)])})
    return rows


def chart_maximum(values, unit):
    return 100. if unit == '%' else max(max(values.values()) * 1.15, 1e-6)


def chart_spec(values, title, unit, higher_better=False):
    """Native Vega-Lite chart with tooltips and explicit labels, without extra imports."""
    rows = chart_rows(values, unit)
    return {
        'data': {'values': rows}, 'height': max(180, len(rows) * 54),
        'title': {'text': title, 'anchor': 'start',
                  'subtitle': 'Higher means more overlap' if higher_better else 'Lower means a closer distance distribution'},
        'encoding': {
            'y': {'field': 'variant', 'type': 'nominal', 'sort': [row['variant'] for row in rows],
                  'axis': {'title': None, 'labelLimit': 240, 'labelFontSize': 14, 'ticks': False, 'domain': False}},
            'x': {'field': 'value', 'type': 'quantitative',
                  'scale': {'domain': [0, chart_maximum(values, unit)]},
                  'axis': {'title': 'Overlap (%)' if unit == '%' else 'Wasserstein-1 (metres)'}},
            'tooltip': [{'field': 'variant', 'title': 'Scan'}, {'field': 'display', 'title': 'Measured value'}]},
        'layer': [
            {'mark': {'type': 'bar', 'cornerRadiusEnd': 5, 'size': 32},
             'encoding': {'color': {'field': 'color', 'type': 'nominal', 'scale': None, 'legend': None}}},
            {'mark': {'type': 'text', 'align': 'left', 'dx': 8, 'fontSize': 15},
             'encoding': {'text': {'field': 'display'}}}],
        'config': {'view': {'stroke': None}}}


def chart(values, title, unit, higher_better=False):
    """Self-contained SVG for downloads; zero values retain explicit labels."""
    rows = chart_rows(values, unit)
    maximum = chart_maximum(values, unit)
    height = 138 + len(rows) * 54
    left, span = 240, 590
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 {height}" role="img" aria-label="{escape(title, quote=True)}">',
             f'<title>{escape(title)}</title><rect width="1000" height="{height}" rx="18" fill="#ffffff"/>',
             '<g font-family="Arial, sans-serif" fill="#17253a">',
             f'<text x="28" y="35" font-size="21" font-weight="bold">{escape(title)}</text>',
             f'<text x="28" y="60" font-size="16">{"Higher means more overlap" if higher_better else "Lower means a closer distance distribution"}</text>']
    for index, row in enumerate(rows):
        y = 90 + index * 54
        parts.extend([f'<text x="28" y="{y + 25}" font-size="19">{escape(row["variant"])}</text>',
            f'<rect x="{left}" y="{y}" width="{span}" height="34" rx="5" fill="#edf2f7"/>',
            f'<rect x="{left}" y="{y}" width="{row["value"] / maximum * span:.3f}" height="34" rx="5" fill="{row["color"]}"/>',
            f'<text x="850" y="{y + 25}" font-size="21" font-weight="bold">{row["display"]}</text>'])
    end = '100%' if unit == '%' else f'{maximum:.3g} m'
    parts.extend([f'<text x="{left}" y="{height - 25}" font-size="14">0{unit if unit == "%" else " m"}</text>',
                  f'<text x="{left + span}" y="{height - 25}" text-anchor="end" font-size="14">{end}</text></g></svg>'])
    return ''.join(parts)


def gallery(report):
    """List every recorded camera/scan image; label training only for a proven pair."""
    arms = paired_arms(report)
    result = []
    for row in report.get('camera_overlays', {}).get('frames', []):
        cards = [{'label': 'Recorded photo', 'file': row.get('camera')},
                 {'label': 'Real scan overlay', 'file': row.get('real', {}).get('file')}]
        synthetic = row.get('synthetic', [])
        if arms:
            used = set()
            for arm in ('untrained', 'trained'):
                filename = arms[arm]
                record = next((value for value in synthetic if value.get('sample') == filename), {})
                cards.append({'label': LABELS[arm], 'file': record.get('file')})
                used.add(filename)
            synthetic = [record for record in synthetic if record.get('sample') not in used]
        cards.extend({'label': f'Additional synthetic scan {index + 1}' if arms else f'Synthetic scan {index + 1}',
                      'file': record.get('file')} for index, record in enumerate(synthetic))
        result.append({'frame': row.get('frame', len(result)), 'cards': cards})
    return result


def overlap_text(data):
    if not data['paired']:
        return 'Each synthetic scan is compared with the same real recording.'
    if not data['overlap']:
        return 'Scan-overlap measurements are not available for both versions.'
    delta = data['overlap']['trained'] - data['overlap']['untrained']
    if abs(delta) < .05:
        return 'The two versions have similar measured scan overlap.'
    return f'The trained version has {abs(delta):.1f} percentage points {"more" if delta > 0 else "less"} overlap with the real scan.'


def share_html(report, data, frames):
    """Self-contained presentation using already computed scores and fetched image bytes."""
    blocks = ['<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>See what training changes</title><style>body{margin:0;background:#f4f7fb;color:#17253a;font:17px system-ui,sans-serif}',
        'main{max-width:1400px;margin:auto;padding:28px}h1{font-size:38px}h2{margin-top:32px}.grid{display:grid;',
        'grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}figure{margin:0;background:white;border-radius:12px;overflow:hidden}',
        'figcaption{font-weight:600;padding:12px}img{display:block;width:100%}svg{display:block;width:100%;max-width:1100px}',
        'p{line-height:1.6}.note{color:#475569}</style></head><body><main><h1>See what training changes</h1>',
        '<p>Photos are recorded; the coloured laser points are synthetic.</p>']
    blocks.append('<p>' + escape(overlap_text(data)) + '</p>')
    if not data['paired']:
        blocks.append('<p>This run has no verified untrained/trained pair; its synthetic scans keep neutral labels.</p>')
    if data['overlap']:
        blocks.extend([chart(data['overlap'], 'Scan overlap with the real recording', '%', True),
            f'<p class="note">Uses {data["overlap_frames"]} of {data["frame_count"]} frames. Overlap compares the laser-return masks; 100% means the same occupied scan pixels.</p>'])
    else:
        blocks.append('<p>Scan-overlap measurements are missing.</p>')
    if data['wasserstein']:
        blocks.extend([chart(data['wasserstein'], 'Distance-distribution difference (Wasserstein-1)', 'm'),
            f'<p class="note">Uses {data["wasserstein_frames"]} of {data["frame_count"]} frames. This is the average of per-frame Wasserstein-1 distances in metres, counting each frame equally.</p>'])
    else:
        blocks.append('<p>Wasserstein-1 measurements are missing. Refresh the analysis in the technical app.</p>')
    for index, frame in enumerate(frames, 1):
        blocks.append(f'<h2>Scene view {index}</h2><div class="grid">')
        for card in frame['cards']:
            blocks.append('<figure><figcaption>' + escape(card['label']) + '</figcaption>')
            if card.get('data'):
                kind = 'png' if str(card.get('file', '')).lower().endswith('.png') else 'jpeg'
                encoded = base64.b64encode(card['data']).decode('ascii')
                blocks.append(f'<img alt="{escape(card["label"], quote=True)}" src="data:image/{kind};base64,{encoded}">')
            else:
                blocks.append('<p>Image unavailable.</p>')
            blocks.append('</figure>')
        blocks.append('</div>')
    blocks.append('<p class="note">The charts describe scan coverage and distance distributions. They do not score photographic realism or establish that training improves every aspect of the result. The synthetic scans use the same conditioning scene.</p></main></body></html>')
    return ''.join(blocks)
