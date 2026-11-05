"""Pixel analysis and portable graphs for conditioned Generate jobs; no model imports."""
import argparse
import csv
from html import escape
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .camera import load_camera, project_points
from .comparison import ARMS, PAIRS, frame_cloud, load_arrays
from .eval.spatial_gap import overlay_rgb_gap, pixel_gap
from .io import write_json
from .logs import get_logger

VERSION = 1
COLORS = ['#60a5fa', '#fbbf24', '#34d399', '#fb7185', '#c084fc', '#22d3ee']
ERROR_COLORS = ['#38bdf8', '#34d399', '#facc15', '#f97316', '#ef4444']


def display_map(values):
    """Largest absolute difference per display block; metrics stay full resolution."""
    step = max(1, int(np.ceil(values.shape[1] / 320)))
    height, width = (int(np.ceil(size / step)) for size in values.shape)
    padded = np.full((height * step, width * step), np.nan)
    padded[:values.shape[0], :values.shape[1]] = values
    blocks = padded.reshape(height, step, width, step).transpose(0, 2, 1, 3).reshape(height, width, -1)
    largest = np.where(np.isfinite(blocks), abs(blocks), -np.inf).argmax(axis=-1)
    return np.take_along_axis(blocks, largest[..., None], axis=-1)[..., 0]


def map_groups(maps, keys):
    groups = [('LiDAR range-grid pixels', [('Range error', 'range_depth_error', 'm', False),
        ('Return occupancy B-A', 'range_occupancy_difference', '', True)])]
    if all(key + '_pixel_depth_error' in maps for key in keys):
        groups.append(('Camera pixels', [('Optical-depth error', 'pixel_depth_error', 'm', False),
            ('Overlay RGB error', 'pixel_rgb_error', '0-1', False),
            ('Return occupancy B-A', 'pixel_occupancy_difference', '', True)]))
    return groups


def error_histograms(maps, keys):
    result = {}
    for title, suffix in [('Range error (m)', 'range_depth_error'), ('Camera depth error (m)', 'pixel_depth_error'),
                          ('Overlay RGB error (0-1)', 'pixel_rgb_error')]:
        if not all(key + '_' + suffix in maps for key in keys):
            continue
        values = {key: maps[key + '_' + suffix][np.isfinite(maps[key + '_' + suffix])] for key in keys}
        maximum = max((float(value.max()) for value in values.values() if len(value)), default=1.)
        edges = np.linspace(0, max(maximum, 1e-6), 33)
        result[title] = {'edges': edges.tolist(),
                        'counts': {key: np.histogram(value, edges)[0].tolist() for key, value in values.items()}}
    return result


def pixel_figures(row, maps):
    """Shared interactive figures for the dedicated and combined offline viewers."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    keys = list(row['pairs'])
    labels = [key.replace('_vs_', ' vs ').replace('_', ' ') for key in keys]
    figures = []
    for title, specifications in map_groups(maps, keys):
        figure = make_subplots(rows=len(specifications), cols=len(keys),
                               subplot_titles=tuple(f'{label}: {spec[0]}' for spec in specifications for label in labels))
        for line, (_, suffix, unit, signed) in enumerate(specifications, 1):
            maximum = max((float(abs(maps[key + '_' + suffix][np.isfinite(maps[key + '_' + suffix])]).max())
                           for key in keys if np.isfinite(maps[key + '_' + suffix]).any()), default=1.)
            for column, key in enumerate(keys, 1):
                figure.add_trace(go.Heatmap(z=display_map(maps[key + '_' + suffix]),
                    zmin=-1 if signed else 0, zmax=1 if signed else max(maximum, 1e-6),
                    colorscale=[[0, '#f97316'], [.5, '#475569'], [1, '#38bdf8']] if signed else ERROR_COLORS,
                    showscale=column == len(keys), colorbar={'len': .8 / len(specifications),
                        'y': 1 - (line - .5) / len(specifications), 'title': unit or 'B-A'}, hoverongaps=False,
                    hovertemplate='Pixel-block difference: %{z:.5f} ' + unit + '<extra></extra>'), row=line, col=column)
        figure.update_yaxes(autorange='reversed')
        figure.update_layout(height=320 * len(specifications), showlegend=False)
        figures.append((title, figure))
    histograms = row['histograms']
    figure = make_subplots(rows=1, cols=len(histograms), subplot_titles=tuple(histograms))
    for column, values in enumerate(histograms.values(), 1):
        edges = np.asarray(values['edges'])
        for index, (key, counts) in enumerate(values['counts'].items()):
            counts = np.asarray(counts)
            figure.add_trace(go.Scatter(x=(edges[:-1] + edges[1:]) / 2,
                y=counts / counts.sum() if counts.sum() else [None] * len(counts),
                name=key.replace('_vs_', ' vs '), legendgroup=key, showlegend=column == 1,
                line={'color': COLORS[index % len(COLORS)]}), row=1, col=column)
    figure.update_yaxes(title_text='Fraction of compared pixels')
    figure.update_layout(height=390)
    figures.append(('Pixel-error distributions', figure))
    return figures


def write_pngs(directory, row, maps):
    """Portable PNGs using the existing Pillow dependency; no browser or Kaleido."""
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 13)
    except OSError:
        font = ImageFont.load_default()
    keys = list(row['pairs'])
    graphs = []
    for group, specifications in map_groups(maps, keys):
        columns = min(3, len(keys))
        blocks = int(np.ceil(len(keys) / columns))
        canvas = Image.new('RGB', (380 * columns, 285 * blocks * len(specifications)), '#111827')
        draw = ImageDraw.Draw(canvas)
        for line, (label, suffix, unit, signed) in enumerate(specifications):
            maximum = max((float(abs(maps[key + '_' + suffix][np.isfinite(maps[key + '_' + suffix])]).max())
                           for key in keys if np.isfinite(maps[key + '_' + suffix]).any()), default=1.)
            maximum = max(maximum, 1e-6)
            palette = np.array([[249, 115, 22], [71, 85, 105], [56, 189, 248]]) if signed else np.array(
                [[56, 189, 248], [52, 211, 153], [250, 204, 21], [249, 115, 22], [239, 68, 68]])
            for index, key in enumerate(keys):
                x, y = (index % columns) * 380, (line * blocks + index // columns) * 285
                values = display_map(maps[key + '_' + suffix])
                fraction = (np.nan_to_num(values) + 1) / 2 if signed else np.nan_to_num(values) / maximum
                pixels = np.stack([np.interp(fraction, np.linspace(0, 1, len(palette)), palette[:, channel])
                                   for channel in range(3)], axis=-1).astype('uint8')
                pixels[~np.isfinite(values)] = [17, 24, 39]
                image = Image.fromarray(pixels)
                image.thumbnail((350, 190), Image.Resampling.NEAREST)
                canvas.paste(image, (x + 15, y + 52))
                draw.text((x + 15, y + 6), key.replace('_vs_', ' vs '), font=font, fill='white')
                draw.text((x + 15, y + 27), label, font=font, fill='white')
                legend = 'Orange: A only | gray: shared | blue: B only' if signed else f'Blue/green to red: 0 to {maximum:.5g} {unit}'
                draw.text((x + 15, y + 247), legend, font=font, fill='white')
                draw.text((x + 15, y + 266), f'Valid pixels: {np.isfinite(maps[key + "_" + suffix]).sum():,}; blank = no support', font=font, fill='white')
        filename = f'pixel_graphs/{"camera" if group == "Camera pixels" else "range_grid"}_frame_{row["frame"]:03d}.png'
        canvas.save(directory / filename)
        graphs.append({'file': filename, 'label': group + ' - error and occupancy maps'})
    histograms = row['histograms']
    canvas = Image.new('RGB', (480 * len(histograms), 340), '#111827')
    draw = ImageDraw.Draw(canvas)
    for column, (title, histogram) in enumerate(histograms.items()):
        offset = column * 480
        draw.text((offset + 25, 8), title, font=font, fill='white')
        draw.line([(offset + 45, 40), (offset + 45, 225), (offset + 455, 225)], fill='white')
        probabilities = {key: np.asarray(counts) / max(1, sum(counts)) for key, counts in histogram['counts'].items()}
        maximum = max((float(values.max()) for values in probabilities.values()), default=0.)
        for index, (key, values) in enumerate(probabilities.items()):
            if sum(histogram['counts'][key]):
                points = [(offset + 45 + (i + .5) / len(values) * 410, 225 - value / max(maximum, 1e-12) * 175)
                          for i, value in enumerate(values)]
                draw.line(points, fill=COLORS[index % len(COLORS)], width=2)
            if index < 6:
                draw.text((offset + 20, 260 + index * 13), key.replace('_vs_', ' vs '), font=font, fill=COLORS[index % len(COLORS)])
        draw.text((offset + 10, 30), f'{maximum:.3g}', font=font, fill='white')
        draw.text((offset + 45, 230), '0', font=font, fill='white')
        label = f'{histogram["edges"][-1]:.5g}'
        box = draw.textbbox((0, 0), label, font=font)
        draw.text((offset + 455 - (box[2] - box[0]), 230), label, font=font, fill='white')
        if not maximum:
            draw.text((offset + 100, 110), 'No compared pixels', font=font, fill='white')
    filename = f'pixel_graphs/distributions_frame_{row["frame"]:03d}.png'
    canvas.save(directory / filename)
    return [*graphs, {'file': filename, 'label': 'Pixel-error distributions (fraction of compared pixels)'}]


def analyze_pixels(directory, report):
    """Analyze every generated sample against real; paired jobs also compare both arms."""
    directory = Path(directory).resolve()
    if not (directory / 'real_scene.npz').is_file():
        return {'version': VERSION, 'status': 'unavailable', 'reason': 'Pixel gaps require the real conditioning scene.', 'frames': []}
    files = ARMS if report.get('comparison') else {path.stem: path.name for path in sorted(directory.glob('sample_*.npz'))}
    if not files:
        raise ValueError('No generated NPZ files are available for pixel analysis')
    scenes = {name: load_arrays(directory / filename) for name, filename in {'real': 'real_scene.npz', **files}.items()}
    reference = scenes['real']
    for name, scene in scenes.items():
        if (scene['range'].ndim != 3 or scene['range'].shape != reference['range'].shape
                or scene['valid'].shape != scene['range'].shape
                or not np.allclose(scene['elevations'], reference['elevations'], atol=1e-7, rtol=0)):
            raise ValueError(f'{name}: pixel analysis requires matching [T,H,W] ranges, masks and beam grids')
        if name in ARMS and str(scene.get('training_status', '')) != name:
            raise ValueError(f'{name}: paired-generation training-status metadata is missing')
    pairs = PAIRS if report.get('comparison') else {name + '_vs_real': ('real', name) for name in files}
    overlays = report.get('camera_overlays', {})
    diagnostic = overlays.get('reason', 'Timestamp-matched camera overlays are unavailable for this job.')
    camera = None
    if overlays.get('status') == 'completed':
        try:
            camera = load_camera(directory / 'overlays/calibration.json')
        except (OSError, ValueError, KeyError) as error:
            diagnostic = str(error)
    folder = directory / 'pixel_graphs'
    folder.mkdir(exist_ok=True)
    frames = []
    for frame in range(reference['range'].shape[0]):
        ranges = {name: np.where(scene['valid'][frame].astype(bool) & np.isfinite(scene['range'][frame])
                                & (scene['range'][frame] > 0), scene['range'][frame], np.nan) for name, scene in scenes.items()}
        images, depths, rgb, reason = {}, {}, {}, diagnostic
        row = next((row for row in overlays.get('frames', []) if row['frame'] == frame), None)
        if camera is not None and row is not None:
            try:
                with Image.open(directory / 'overlays' / row['camera']) as image:
                    size = image.size
                images['camera'] = 'overlays/' + row['camera']
                for name, scene in scenes.items():
                    entry = row['real'] if name == 'real' else next(value for value in row['synthetic'] if value['sample'] == files[name])
                    filename = 'overlays/' + entry['file']
                    with Image.open(directory / filename) as image:
                        rgb[name] = np.asarray(image.convert('RGB'))
                    images[name] = filename
                    pixels, distances, _ = project_points(frame_cloud(scene, frame)[0], camera, size)
                    depth = np.full((size[1], size[0]), np.nan, dtype='float32')
                    depth[pixels[:, 1], pixels[:, 0]] = distances
                    depths[name] = depth
                reason = None
            except (OSError, ValueError, KeyError, StopIteration) as error:
                images, depths, rgb, reason = {}, {}, {}, str(error)
        metrics, maps = {}, {}
        for key, (a, b) in pairs.items():
            grid, error, occupancy = pixel_gap(ranges[a], ranges[b])
            maps[key + '_range_depth_error'], maps[key + '_range_occupancy_difference'] = error, occupancy
            values = {'reference_a': a, 'candidate_b': b, 'range_pixel': grid, 'camera_pixel': None, 'overlay_rgb': None}
            if depths:
                values['camera_pixel'], error, occupancy = pixel_gap(depths[a], depths[b])
                maps[key + '_pixel_depth_error'], maps[key + '_pixel_occupancy_difference'] = error, occupancy
                values['overlay_rgb'], maps[key + '_pixel_rgb_error'] = overlay_rgb_gap(rgb[a], rgb[b], depths[a], depths[b])
            metrics[key] = values
        map_file = f'pixel_graphs/maps_{frame:03d}.npz'
        np.savez_compressed(directory / map_file, **maps)
        result = {'frame': frame, 'pairs': metrics, 'maps': map_file, 'images': images, 'camera_reason': reason,
                  'camera_skew_ms': row.get('skew_ms') if row else None, 'histograms': error_histograms(maps, pairs)}
        result['graphs'] = write_pngs(directory, result, maps)
        frames.append(result)
        get_logger().info('Exported pixel graphs for frame %d; camera=%s', frame, 'available' if depths else reason)
    result = {'version': VERSION, 'status': 'completed' if all(row['images'] for row in frames) else 'range_grid_only',
              'frames': frames, 'html': 'pixel_graphs.html', 'csv': 'pixel_graphs/metrics.csv',
              'definitions': {'range_grid': 'LiDAR beam/azimuth pixels; radial ranges in metres, independent of camera availability.',
                  'camera': 'Timestamp-matched optical-camera pixels; optical depth in metres.',
                  'support': 'Depth error uses shared valid pixels; occupancy uses the union. Missing support is blank/null, never fabricated zero.',
                  'display': 'Display blocks retain the largest absolute difference. All metrics and saved NPZ maps use full resolution.',
                  'overlay_rgb': 'RGB error on projected foreground over the same recorded photo, excluding the label strip; not synthetic RGB fidelity.',
                  'conditioning': 'Geometry agreement is constrained by real conditioning and range clamping; these are descriptive paired-scene gaps.'}}
    with (directory / result['csv']).open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['frame', 'pair', 'space', 'metric', 'value'])
        for row in frames:
            for pair, values in row['pairs'].items():
                for space in ('range_pixel', 'camera_pixel', 'overlay_rgb'):
                    for metric, value in (values[space] or {}).items():
                        writer.writerow([row['frame'], pair, space, metric, value])
    write_pixel_html(directory, result)
    write_json(directory / 'pixel_graphs.json', result)
    return result


def write_pixel_html(directory, result):
    sections, first = [], True
    for row in result['frames']:
        content = []
        if row['camera_reason']:
            content.append('<p>Camera pixels unavailable: ' + escape(row['camera_reason']) + '. LiDAR range-grid graphs are available below.</p>')
        with np.load(directory / row['maps'], allow_pickle=False) as maps:
            for index, (title, figure) in enumerate(pixel_figures(row, maps)):
                figure.update_layout(template='plotly_dark', margin={'t': 90, 'b': 50})
                content.append('<h2>' + escape(title) + '</h2>' + figure.to_html(full_html=False,
                    include_plotlyjs=first, div_id=f'pixel_{row["frame"]}_{index}', config={'responsive': True}))
                first = False
        sections.append(f'<section data-frame="{row["frame"]}" {"hidden" if row["frame"] else ""}>{"".join(content)}</section>')
    options = ''.join(f'<option value="{row["frame"]}">Frame {row["frame"]}</option>' for row in result['frames'])
    definitions = ''.join('<li>' + escape(value) + '</li>' for value in result['definitions'].values())
    document = ('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Pixel domain gaps</title><style>body{background:#111827;color:#e2e8f0;font:15px system-ui;margin:24px}'
        'select{padding:8px}h2{margin-top:32px}</style></head><body><h1>Pixel domain gaps</h1>'
        '<p>Signed occupancy: A is the reference, B is the candidate. Orange: A only; blue: B only. '
        'Heatmap scales use the observed maximum per metric. Blank pixels have no valid support.</p>'
        f'<ul>{definitions}</ul><label>Temporal frame <select onchange="showFrame(this.value)">{options}</select></label>'
        + ''.join(sections) + '<script>function showFrame(value){document.querySelectorAll("section[data-frame]").forEach(section=>{'
        'section.hidden=section.dataset.frame!==value;});requestAnimationFrame(()=>document.querySelectorAll(".plotly-graph-div")'
        '.forEach(graph=>{if(graph.offsetParent!==null)Plotly.Plots.resize(graph);}));}</script></body></html>')
    (directory / result['html']).write_text(document, encoding='utf-8')


def rebuild(directory):
    """Backfill existing generated artifacts without training, sampling or remote data access."""
    directory = Path(directory).resolve()
    source = directory / 'result.json'
    report = json.loads(source.read_text())
    if report.get('action') != 'generate':
        raise ValueError('Pixel graphs require an existing Generate job')
    report['pixel_analysis'] = analyze_pixels(directory, report)
    if report.get('comparison'):
        from .comparison import analyze
        report['comparison'] = analyze(directory, report, report['comparison']['voxel_size_m'])
    write_json(source, report)
    from .report import write_pdf
    write_pdf(source)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    arguments = parser.parse_args()
    report = rebuild(arguments.job)
    print(json.dumps({'status': report['pixel_analysis']['status'], 'viewer': str(arguments.job / 'pixel_graphs.html')}, indent=2))
