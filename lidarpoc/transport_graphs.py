"""Distribution inputs and OT plots shared by the UI, offline viewers and PDF."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation

from .comparison import frame_cloud
from .eval.transport_gap import LABELS, empirical_cdf, multivariate_gap, scalar_gap
from .logs import get_logger

COLORS = ['#60a5fa', '#fbbf24', '#34d399', '#fb7185', '#c084fc', '#22d3ee']


def analyze_frame(scenes, ranges, depths, rgb, pairs, frame, max_range, settings):
    clouds = {name: frame_cloud(scene, frame) for name, scene in scenes.items()}
    scalar = {'radial_range': ({name: value[np.isfinite(value)] for name, value in ranges.items()}, 'm')}
    if clouds['real'][1] is not None:
        scalar['intensity'] = ({name: data[1] if data[1] is not None else np.array([]) for name, data in clouds.items()}, '0-1')
    if depths:
        scalar['camera_depth'] = ({name: value[np.isfinite(value)] for name, value in depths.items()}, 'm')
    spaces, arrays = {}, {}
    image_features, image_attributes = {}, {}
    if rgb:
        # Same union foreground for all pairs; shared backgrounds do not dominate RGB scores.
        roi = binary_dilation(np.logical_or.reduce([np.isfinite(value) for value in depths.values()]),
                              structure=np.ones((3, 3)))
        roi[:25] = False
        v, u = np.nonzero(roi)
        uv = np.column_stack((u / max(1, roi.shape[1] - 1), v / max(1, roi.shape[0] - 1)))
        image_attributes = {name: image[roi].astype(float) / 255 for name, image in rgb.items()}
        image_features = {name: np.column_stack((uv, values)) for name, values in image_attributes.items()}
        scalar['overlay_luminance'] = ({name: value @ np.array([.2126, .7152, .0722])
                                       for name, value in image_attributes.items()}, '0-1')
    for space, (values, unit) in scalar.items():
        spaces[space] = {'kind': 'scalar', 'pairs': {key: scalar_gap(values[a], values[b], unit)
                        for key, (a, b) in pairs.items()},
                        'cdfs': {name: empirical_cdf(value) for name, value in values.items()}}
    attributes = {}
    intensity_available = all(value[1] is not None for value in clouds.values())
    for name, (xyz, intensity) in clouds.items():
        features = [np.linalg.norm(xyz, axis=1) / max_range]
        if intensity_available:
            features.append(intensity)
        attributes[name] = np.column_stack(features)
    inputs = {'cloud_xyz': ({name: value[0] for name, value in clouds.items()}, attributes, max_range, 'm',
                           {name: len(value[0]) / ranges[name].size for name, value in clouds.items()})}
    if image_features:
        inputs['overlay_xy_rgb'] = (image_features, image_attributes, 1., 'normalized', {name: 1. for name in rgb})
    for space, (values, features, scale, unit, masses) in inputs.items():
        results = {}
        for key, (a, b) in pairs.items():
            get_logger().info('OT frame %d: %s, %s; support limit=%d', frame, space, key, settings.max_points)
            results[key], plans = multivariate_gap(values[a], values[b], scale, unit, masses[a], masses[b],
                                                    features[a], features[b], settings)
            arrays.update({f'ot_{space}_{key}_{name}': value for name, value in plans.items()})
        spaces[space] = {'kind': 'multivariate', 'pairs': results}
    return {'spaces': spaces}, arrays


def figures(transport):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    result = []
    for space, data in transport['spaces'].items():
        keys = list(next(iter(data['pairs'].values()))['values'])
        columns = min(3, len(keys))
        rows = int(np.ceil(len(keys) / columns))
        units = next(iter(data['pairs'].values()))['units']
        figure = make_subplots(rows=rows, cols=columns,
            subplot_titles=tuple(f'{LABELS[key]} ({units[key]})' for key in keys))
        for index, metric in enumerate(keys):
            figure.add_trace(go.Bar(x=[key.replace('_vs_', ' vs ') for key in data['pairs']],
                y=[value['values'][metric] for value in data['pairs'].values()],
                marker_color=[COLORS[i % len(COLORS)] for i in range(len(data['pairs']))]),
                row=index // columns + 1, col=index % columns + 1)
        figure.update_layout(height=300 * rows + 60, showlegend=False)
        result.append((f'Optimal transport - {space}', figure))
        if data.get('cdfs'):
            figure = go.Figure()
            for name, values in data['cdfs'].items():
                figure.add_trace(go.Scatter(x=values['x'], y=values['cdf'], mode='lines', name=name,
                    line={'shape': 'hv'}))
            figure.update_layout(height=360, xaxis_title=f'{space} ({units["wasserstein_1"]})',
                                 yaxis_title='Empirical cumulative probability', yaxis_range=[0, 1])
            result.append((f'Distributions compared by Wasserstein - {space}', figure))
    return result


def write_pngs(directory, frame, transport):
    """Portable bar/CDF plots, without browser-based image export."""
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 13)
    except OSError:
        font = ImageFont.load_default()
    graphs = []
    for space, data in transport['spaces'].items():
        metrics = list(next(iter(data['pairs'].values()))['values'])
        units = next(iter(data['pairs'].values()))['units']
        columns = min(3, len(metrics))
        rows = int(np.ceil(len(metrics) / columns))
        canvas = Image.new('RGB', (420 * columns, 310 * rows), '#111827')
        draw = ImageDraw.Draw(canvas)
        for index, metric in enumerate(metrics):
            x, y = index % columns * 420, index // columns * 310
            draw.text((x + 12, y + 8), LABELS[metric], font=font, fill='white')
            draw.text((x + 12, y + 26), space + ' | ' + units[metric], font=font, fill='white')
            valid = [value['values'][metric] for value in data['pairs'].values() if value['values'][metric] is not None]
            maximum = max(valid, default=0.)
            for position, (pair, value) in enumerate(data['pairs'].items()):
                left = x + 45 + position * 350 / len(data['pairs'])
                width = min(50, 230 / len(data['pairs']))
                amount = value['values'][metric]
                if amount is not None:
                    top = y + 190 - amount / max(maximum, 1e-12) * 130
                    draw.rectangle((left, top, left + width, y + 190), fill=COLORS[position % len(COLORS)])
                    label = f'{amount:.4g}'
                else:
                    label = 'n/a'
                draw.text((left, y + 196), label, font=font, fill='white')
                if position < 6:
                    draw.text((x + 12, y + 222 + position * 13), pair.replace('_vs_', ' vs '),
                              font=font, fill=COLORS[position % len(COLORS)])
            draw.line((x + 40, y + 55, x + 40, y + 190, x + 395, y + 190), fill='white')
            header = f'Max {maximum:.4g}; lower is better' if valid else 'Unavailable: see solver diagnostics'
            draw.text((x + 45, y + 45), header, font=font, fill='white')
        filename = f'pixel_graphs/ot_{space}_frame_{frame:03d}.png'
        canvas.save(directory / filename)
        graphs.append({'file': filename, 'label': f'Optimal transport: {space} (solver details and approximations in JSON)'})
    scalar = {space: data for space, data in transport['spaces'].items() if data.get('cdfs')}
    if scalar:
        canvas = Image.new('RGB', (420 * min(2, len(scalar)), 300 * int(np.ceil(len(scalar) / 2))), '#111827')
        draw = ImageDraw.Draw(canvas)
        for index, (space, data) in enumerate(scalar.items()):
            x, y = index % 2 * 420, index // 2 * 300
            draw.text((x + 12, y + 8), space + ' | empirical CDF', font=font, fill='white')
            all_values = [value for record in data['cdfs'].values() for value in record['x']]
            low, high = (min(all_values), max(all_values)) if all_values else (0., 1.)
            for position, (name, record) in enumerate(data['cdfs'].items()):
                points = [(x + 40 + (a - low) / max(high - low, 1e-12) * 350, y + 190 - b * 140)
                          for a, b in zip(record['x'], record['cdf'])]
                if len(points) > 1:
                    draw.line(points, fill=COLORS[position % len(COLORS)], width=2)
                if position < 6:
                    draw.text((x + 12, y + 212 + position * 13), name, font=font, fill=COLORS[position % len(COLORS)])
            draw.line((x + 40, y + 45, x + 40, y + 190, x + 395, y + 190), fill='white')
            draw.text((x + 12, y + 32), '1', font=font, fill='white')
            draw.text((x + 40, y + 193), f'{low:.4g}', font=font, fill='white')
            draw.text((x + 330, y + 193), f'{high:.4g}', font=font, fill='white')
        filename = f'pixel_graphs/ot_cdfs_frame_{frame:03d}.png'
        canvas.save(directory / filename)
        graphs.append({'file': filename, 'label': 'Empirical distribution curves; metrics use all valid scalar values'})
    return graphs
