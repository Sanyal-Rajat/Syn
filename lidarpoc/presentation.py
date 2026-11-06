"""Plain-language comparison summaries and shareable HTML; no model/UI dependencies."""
import base64
from html import escape
from io import BytesIO
import math
from numbers import Real
from pathlib import PurePosixPath
import re

LABELS = {'real': 'Actual', 'untrained': 'Without training', 'trained': 'With training'}
COLORS = {'real': '#94a3b8', 'untrained': '#d97706', 'trained': '#0284c7'}
CHARTS = {
    'overlap': {'title': 'Coverage match', 'unit': '%', 'higher_better': True,
                'description': 'A higher percentage means the coverage is closer to the real scene.'},
    'wasserstein': {'title': 'Distance difference', 'unit': 'm', 'higher_better': False,
                    'description': 'Compares the pattern of measured distances. Smaller values mean a closer match.'}}


def number(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def image_name(value):
    if not isinstance(value, str) or PurePosixPath(value).name != value:
        raise ValueError('Expected a camera image filename, without directories')
    if PurePosixPath(value).suffix.lower() not in ('.jpg', '.jpeg', '.png'):
        raise ValueError('Camera images must be PNG or JPEG files')
    return value


def presentation_image(data, label):
    """Replace the existing 25-row overlay banner without changing scene pixels."""
    from PIL import Image, ImageDraw

    with Image.open(BytesIO(data)) as original:
        image = original.convert('RGB')
    image.info.clear()
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 24), fill='black')
    draw.text((8, 6), label, fill='white')
    output = BytesIO()
    image.save(output, format='PNG')
    return output.getvalue()


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
    order = [arm for arm in ('real', 'untrained', 'trained') if arm in values]
    order.extend(arm for arm in values if arm not in order)
    for arm in order:
        value = values[arm]
        label = LABELS.get(arm)
        if label is None:
            label = f'Version {int(arm.removeprefix("sample_")) + 1}'
        rows.append({'variant': label, 'value': value,
                     'display': f'{value:.1f}%' if unit == '%' else f'{value:.3g} m',
                     'color': COLORS.get(arm, palette[(len(rows) - 1) % len(palette)])})
    return rows


def chart_maximum(values, unit):
    return 100. if unit == '%' else max(max(values.values()) * 1.15, 1e-6)


def chart_spec(values, title, unit, higher_better=False):
    """Keep names left, bars central and measured values in a separate right column."""
    rows = chart_rows(values, unit)
    colors = {'field': 'color', 'type': 'nominal', 'scale': None, 'legend': None}
    return {
        'data': {'values': rows}, 'height': max(210, len(rows) * 68),
        'padding': {'left': 8, 'right': 110, 'top': 16, 'bottom': 12},
        'autosize': {'type': 'fit-x', 'contains': 'padding', 'resize': True},
        'title': {'text': title, 'anchor': 'start', 'offset': 22,
                  'subtitle': 'Higher means a closer match' if higher_better else 'Lower means a closer match'},
        'encoding': {
            'y': {'field': 'variant', 'type': 'nominal', 'sort': [row['variant'] for row in rows],
                  'scale': {'type': 'band', 'paddingInner': .45, 'paddingOuter': .3},
                  'axis': {'title': None, 'labelLimit': 200, 'labelPadding': 16,
                           'labelFontSize': 15, 'ticks': False, 'domain': False}},
            'x': {'field': 'value', 'type': 'quantitative', 'stack': None,
                  'scale': {'domain': [0, chart_maximum(values, unit)], 'nice': False},
                  'axis': {'title': 'Match (%)' if unit == '%' else 'Difference (metres)'}},
            'tooltip': [{'field': 'variant', 'title': 'Version'}, {'field': 'display', 'title': 'Result'}]},
        'layer': [
            {'mark': {'type': 'bar', 'orient': 'horizontal', 'size': 32,
                      'cornerRadius': 5, 'color': '#edf2f7'},
             'encoding': {'x': {'value': 0}, 'x2': {'value': 'width'}}},
            {'mark': {'type': 'bar', 'cornerRadiusEnd': 5, 'size': 32},
             'encoding': {'color': colors, 'x2': {'datum': 0}}},
            {'transform': [{'filter': 'datum.value === 0'}],
             'mark': {'type': 'tick', 'orient': 'vertical', 'size': 32, 'thickness': 4},
             'encoding': {'color': colors}},
            {'mark': {'type': 'text', 'align': 'left', 'baseline': 'middle', 'dx': 18, 'fontSize': 16},
             'encoding': {'x': {'value': 'width'}, 'text': {'field': 'display'}}}],
        'config': {'view': {'stroke': None}}}


def chart(values, title, unit, higher_better=False):
    """Self-contained SVG for downloads; zero values retain explicit labels."""
    rows = chart_rows(values, unit)
    maximum = chart_maximum(values, unit)
    height = 158 + len(rows) * 68
    left, span = 240, 590
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 {height}" role="img" aria-label="{escape(title, quote=True)}">',
             f'<title>{escape(title)}</title><rect width="1000" height="{height}" rx="18" fill="#ffffff"/>',
             '<g font-family="Arial, sans-serif" fill="#17253a">',
             f'<text x="28" y="35" font-size="21" font-weight="bold">{escape(title)}</text>',
             f'<text x="28" y="60" font-size="16">{"Higher" if higher_better else "Lower"} means a closer match</text>']
    for index, row in enumerate(rows):
        y = 104 + index * 68
        parts.extend([f'<text x="28" y="{y + 25}" font-size="19">{escape(row["variant"])}</text>',
            f'<rect x="{left}" y="{y}" width="{span}" height="34" rx="5" fill="#edf2f7"/>',
            f'<rect x="{left}" y="{y}" width="{row["value"] / maximum * span:.3f}" height="34" rx="5" fill="{row["color"]}"/>',
            f'<text x="850" y="{y + 25}" font-size="21" font-weight="bold">{row["display"]}</text>'])
        if row['value'] == 0:
            parts.append(f'<line x1="{left}" x2="{left}" y1="{y + 2}" y2="{y + 32}" stroke="{row["color"]}" stroke-width="4"/>')
    end = '100%' if unit == '%' else f'{maximum:.3g} m'
    parts.extend([f'<text x="{left}" y="{height - 25}" font-size="14">0{unit if unit == "%" else " m"}</text>',
                  f'<text x="{left + span}" y="{height - 25}" text-anchor="end" font-size="14">{end}</text></g></svg>'])
    return ''.join(parts)


def gallery(report):
    """List every recorded camera/scan image; label training only for a proven pair."""
    arms = paired_arms(report)
    result = []
    for row in report.get('camera_overlays', {}).get('frames', []):
        cards = [{'label': 'Camera photo', 'file': row.get('camera'), 'overlay': False},
                 {'label': LABELS['real'], 'file': row.get('real', {}).get('file'), 'overlay': True}]
        synthetic = row.get('synthetic', [])
        if arms:
            used = set()
            for arm in ('untrained', 'trained'):
                filename = arms[arm]
                record = next((value for value in synthetic if value.get('sample') == filename), {})
                cards.append({'label': LABELS[arm], 'file': record.get('file'), 'overlay': True})
                used.add(filename)
            synthetic = [record for record in synthetic if record.get('sample') not in used]
        cards.extend({'label': f'Additional version {index + 1}' if arms else f'Version {index + 1}',
                      'file': record.get('file'), 'overlay': True} for index, record in enumerate(synthetic))
        result.append({'frame': row.get('frame', len(result)), 'cards': cards})
    return result


def overlap_text(data):
    if not data['paired']:
        return 'Each generated version is compared with the same real scene.'
    if not data['overlap']:
        return 'Coverage results are not available for both versions.'
    delta = data['overlap']['trained'] - data['overlap']['untrained']
    if abs(delta) < .05:
        return 'Both versions have a similar coverage match.'
    return ('The version with training' if delta > 0 else 'The version without training') + ' has the closer coverage match.'


def chart_caption(data, key):
    text = f'{CHARTS[key]["description"]} Based on {data[key + "_frames"]} of {data["frame_count"]} views.'
    if key == 'wasserstein':
        text += ' Actual is compared with itself, so its difference is zero. A small line marks zero.'
    return text


def share_html(report, data, frames):
    """Self-contained presentation using computed scores and the displayed image copies."""
    blocks = ['<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Scene comparison</title><style>body{margin:0;background:#f4f7fb;color:#17253a;font:17px system-ui,sans-serif}',
        'main{max-width:1400px;margin:auto;padding:28px}h1{font-size:38px}h2{margin-top:32px}.grid{display:grid;',
        'grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}figure{margin:0;background:white;border-radius:12px;overflow:hidden}',
        'figcaption{font-weight:600;padding:12px}img{display:block;width:100%}svg{display:block;width:100%;max-width:1100px}',
        'p{line-height:1.6}.note{color:#475569}</style></head><body><main><h1>Scene comparison</h1>',
        '<p>The generated versions show coloured scene points on recorded photos.</p>']
    blocks.append('<p>' + escape(overlap_text(data)) + '</p>')
    if not data['paired']:
        blocks.append('<p>These results show individual versions. A comparison with both versions is not included.</p>')
    for key, info in CHARTS.items():
        if data[key]:
            blocks.extend([chart(data[key], info['title'], info['unit'], info['higher_better']),
                           '<p class="note">' + escape(chart_caption(data, key)) + '</p>'])
        else:
            blocks.append('<p>' + escape(info['title']) + ' is unavailable for this result.</p>')
    for index, frame in enumerate(frames, 1):
        blocks.append(f'<h2>View {index}</h2><div class="grid">')
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
    blocks.append('<p class="note">Each graph describes one aspect of the scene. The generated versions share the recorded photo.</p></main></body></html>')
    return ''.join(blocks)
