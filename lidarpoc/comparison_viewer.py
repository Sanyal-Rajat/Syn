"""Offline, side-by-side comparison viewer; reads metrics computed on the server."""
import base64
from html import escape
from pathlib import Path

import numpy as np

from .comparison import ARMS, PAIRS, frame_cloud, load_arrays

LABELS = {'untrained': 'Without training', 'trained': 'With training', 'real': 'Recorded LiDAR'}


def image_html(directory, filename, label):
    if not filename:
        return f'<figure><figcaption>{escape(label)}</figcaption><p>Camera image unavailable.</p></figure>'
    path = (directory / filename).resolve()
    if not path.is_relative_to(directory) or path.suffix.lower() not in ('.jpg', '.png', '.jpeg'):
        raise ValueError('Comparison image must be a PNG/JPG inside the job directory')
    kind = 'png' if path.suffix.lower() == '.png' else 'jpeg'
    data = base64.b64encode(path.read_bytes()).decode('ascii')
    return f'<figure><figcaption>{escape(label)}</figcaption><img src="data:image/{kind};base64,{data}" alt="{escape(label)}"></figure>'


def display_map(values):
    """Largest absolute difference per display block; metrics stay full resolution."""
    step = max(1, int(np.ceil(values.shape[1] / 320)))
    height, width = (int(np.ceil(size / step)) for size in values.shape)
    padded = np.full((height * step, width * step), np.nan)
    padded[:values.shape[0], :values.shape[1]] = values
    blocks = padded.reshape(height, step, width, step).transpose(0, 2, 1, 3).reshape(height, width, -1)
    largest = np.where(np.isfinite(blocks), abs(blocks), -np.inf).argmax(axis=-1)
    return np.take_along_axis(blocks, largest[..., None], axis=-1)[..., 0]


def write_viewer(directory, report):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    directory = Path(directory).resolve()
    scenes = {name: load_arrays(directory / filename) for name, filename in {'real': 'real_scene.npz', **ARMS}.items()}
    sections, first = [], True

    def chart(figure, identity, height):
        nonlocal first
        figure.update_layout(template='plotly_dark', height=height, margin={'t': 55, 'b': 50},
                             paper_bgcolor='#111827', plot_bgcolor='#111827')
        result = figure.to_html(full_html=False, include_plotlyjs=first, div_id=identity, config={'responsive': True})
        first = False
        return result

    for row in report['frames']:
        frame = row['frame']
        images = row['images']
        content = ['<div class="columns">', *(image_html(directory, images.get(name), LABELS[name] + ' · LiDAR on recorded RGB')
                    for name in ARMS), '</div>']
        if row.get('camera_skew_ms') is not None:
            content.append(f'<p>Camera/LiDAR time difference: {row["camera_skew_ms"]:.3f} ms.</p>')
        if images:
            content.extend(['<details><summary>Recorded camera and real-LiDAR reference</summary><div class="columns">',
                image_html(directory, images.get('camera'), 'Recorded camera'),
                image_html(directory, images.get('real'), 'Recorded LiDAR on the same photo'), '</div></details>'])
        clouds = {name: frame_cloud(scene, frame)[0] for name, scene in scenes.items()}
        joined = np.concatenate(list(clouds.values()))
        nearby = joined[(abs(joined[:, :2]) <= 30).all(axis=1)]
        z = nearby[:, 2] if len(nearby) else np.array([-2., 4.])
        axis = {'visible': False}
        scene = {'xaxis': {**axis, 'range': [-30, 30]}, 'yaxis': {**axis, 'range': [-30, 30]},
                 'zaxis': {**axis, 'range': [float(z.min()) - .5, float(z.max()) + .5]}, 'aspectmode': 'data',
                 'camera': {'eye': {'x': 1.3, 'y': -1.7, 'z': 1.0}}}
        figure = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]],
                               subplot_titles=tuple(LABELS[name] for name in ARMS))
        for column, name in enumerate(ARMS, 1):
            points = clouds[name]
            shown = points[np.linspace(0, len(points) - 1, min(len(points), 40000), dtype=int)] if len(points) else points
            figure.add_trace(go.Scatter3d(x=shown[:, 0], y=shown[:, 1], z=shown[:, 2], mode='markers',
                marker={'size': 1.6, 'color': shown[:, 2], 'colorscale': 'Turbo', 'cmin': -2, 'cmax': 6},
                name=LABELS[name], hovertemplate='X=%{x:.2f}m Y=%{y:.2f}m Z=%{z:.2f}m<extra></extra>'), row=1, col=column)
        figure.update_layout(scene=scene, scene2=scene, showlegend=False)
        full = {f'{name}.{dimension}axis.range': [float(joined[:, index].min()) - .5, float(joined[:, index].max()) + .5]
                for name in ('scene', 'scene2') for index, dimension in enumerate('xyz')} if len(joined) else {}
        near = {f'{name}.{dimension}axis.range': scene[dimension + 'axis']['range']
                for name in ('scene', 'scene2') for dimension in 'xyz'}
        figure.update_layout(updatemenus=[{'buttons': [
            {'label': 'Nearby scene', 'method': 'relayout', 'args': [near]},
            {'label': 'Full scene', 'method': 'relayout', 'args': [full]}], 'x': 0, 'y': 1.15}])
        content.extend(['<h2>Point clouds · shared scale and synchronized camera</h2>', chart(figure, f'clouds_{frame}', 520)])

        specifications = [('Camera depth MAE (m)', 'pixel', 'depth_mae_m'),
                          ('Pixel occupancy IoU', 'pixel', 'occupancy_iou'),
                          ('Overlay foreground RGB MAE (0–1)', 'overlay_rgb', 'foreground_mae_0_1'),
                          ('Chamfer mean distance (m)', 'voxel', 'chamfer_mean_m'),
                          ('Voxel occupancy IoU', 'voxel', 'occupancy_iou'),
                          ('Voxel density JS distance (0–1)', 'voxel', 'js_distance')]
        figure = make_subplots(rows=2, cols=3, subplot_titles=tuple(item[0] for item in specifications))
        pair_labels = [key.replace('_vs_', ' / ').replace('_', ' ') for key in PAIRS]
        for index, (_, group, key) in enumerate(specifications):
            values = [(row['pairs'][pair][group] or {}).get(key) for pair in PAIRS]
            figure.add_trace(go.Bar(x=pair_labels, y=values, marker_color=['#e5a950', '#4ec9b0', '#67a9ee']),
                             row=index // 3 + 1, col=index % 3 + 1)
        figure.update_layout(showlegend=False)
        content.extend(['<h2>Pixel / voxel gaps</h2><p>IoU: higher is better. Distances and errors: lower is better. Missing support is shown as n/a, never zero.</p>',
                        chart(figure, f'scores_{frame}', 630)])
        table = ['<table><thead><tr><th>Comparison</th><th>Pixels A / B / shared</th><th>Points A / B</th><th>Voxels A / B / shared</th></tr></thead><tbody>']
        for key, values in row['pairs'].items():
            pixel, voxel = values['pixel'], values['voxel']
            count = 'n/a' if pixel is None else f"{pixel['occupied_pixels_a']} / {pixel['occupied_pixels_b']} / {pixel['shared_pixels']}"
            table.append(f'<tr><td>{escape(key)}</td><td>{count}</td><td>{voxel["points_a"]} / {voxel["points_b"]}</td>'
                         f'<td>{voxel["occupied_voxels_a"]} / {voxel["occupied_voxels_b"]} / {voxel["shared_voxels"]}</td></tr>')
        content.append(''.join(table) + '</tbody></table>')

        with np.load(directory / row['maps'], allow_pickle=False) as maps:
            if images:
                figure = make_subplots(rows=3, cols=3, subplot_titles=tuple(pair_labels) * 3)
                maximum = max((float(np.nanmax(maps[key + '_pixel_depth_error']))
                               for key in PAIRS if np.isfinite(maps[key + '_pixel_depth_error']).any()), default=1)
                for column, key in enumerate(PAIRS, 1):
                    figure.add_trace(go.Heatmap(z=display_map(maps[key + '_pixel_depth_error']), zmin=0,
                        zmax=max(maximum, .001), colorscale='Magma', showscale=column == 3,
                        hovertemplate='Display block maximum error: %{z:.4f}m<extra></extra>'), row=1, col=column)
                    figure.add_trace(go.Heatmap(z=display_map(maps[key + '_pixel_rgb_error']), zmin=0, zmax=1,
                        colorscale='Magma', showscale=False,
                        hovertemplate='Overlay RGB MAE (0–1): %{z:.4f}<extra></extra>'), row=2, col=column)
                    figure.add_trace(go.Heatmap(z=display_map(maps[key + '_pixel_occupancy_difference']), zmin=-1, zmax=1,
                        colorscale=[[0, '#f97316'], [.5, '#334155'], [1, '#38bdf8']], showscale=False,
                        hovertemplate='Occupancy B−A: %{z}<extra></extra>'), row=3, col=column)
                figure.update_yaxes(autorange='reversed', title_text='Camera pixel blocks')
                content.extend(['<h2>Camera pixel maps: depth error / RGB overlay error / occupancy</h2>'
                    '<p>Rows: optical-depth MAE in metres; RGB MAE (0–1) within overlay foreground; '
                    'occupancy B minus A (orange: A only, blue: B only). Missing support is blank. '
                    'Each display block retains its largest absolute difference; all metrics use full resolution.</p>',
                                chart(figure, f'pixels_{frame}', 940)])
            figure = make_subplots(rows=1, cols=3, specs=[[{'type': 'scene'}] * 3], subplot_titles=tuple(pair_labels))
            scale = max((float(abs(maps[key + '_voxel_density_difference']).max())
                         for key in PAIRS if len(maps[key + '_voxel_density_difference'])), default=1)
            for column, key in enumerate(PAIRS, 1):
                centers, difference = maps[key + '_voxel_centers'], maps[key + '_voxel_density_difference']
                indices = np.flatnonzero(difference)
                if len(indices) > 20000:
                    indices = indices[np.linspace(0, len(indices) - 1, 20000, dtype=int)]
                centers = centers[indices]
                figure.add_trace(go.Scatter3d(x=centers[:, 0], y=centers[:, 1], z=centers[:, 2], mode='markers',
                    marker={'size': 2, 'color': difference[indices], 'colorscale': 'RdBu', 'cmin': -max(scale, 1e-12),
                            'cmax': max(scale, 1e-12), 'showscale': column == 3}, name=key,
                    hovertemplate='Voxel center: %{x:.2f}, %{y:.2f}, %{z:.2f}m<extra></extra>'), row=1, col=column)
            figure.update_layout(scene=scene, scene2=scene, scene3=scene, showlegend=False)
            content.extend(['<h2>Per-voxel density differences (B minus A)</h2>', chart(figure, f'voxels_{frame}', 430)])

        feature_names = ['range_m', 'height_m', 'intensity']
        figure = make_subplots(rows=1, cols=3, subplot_titles=tuple(feature_names))
        for column, feature in enumerate(feature_names, 1):
            for name in scenes:
                values = row['covariates'][name].get(feature)
                if values is None:
                    continue
                edges, counts = np.asarray(values['edges']), np.asarray(values['counts'])
                figure.add_trace(go.Scatter(x=(edges[:-1] + edges[1:]) / 2,
                    y=counts / max(1, counts.sum()), mode='lines', name=LABELS[name],
                    legendgroup=name, showlegend=column == 1), row=1, col=column)
        content.extend(['<h2>Covariate distributions</h2><p>Histograms use the same bins in all arms. These are descriptive scene discrepancies, not a population-level shift test.</p>',
                        chart(figure, f'features_{frame}', 390)])
        script = (f'<script>(function(){{let busy=false;const graph=document.getElementById("clouds_{frame}");'
            'graph.on("plotly_relayout",function(event){if(busy)return;for(const source of ["scene","scene2"]){'
            'const camera=event[source+".camera"];if(camera){busy=true;const target=source==="scene"?"scene2":"scene";'
            'Plotly.relayout(graph,{[target+".camera"]:camera}).finally(()=>{busy=false});break;}}});})();</script>')
        sections.append(f'<section class="frame" data-frame="{frame}" {"hidden" if frame else ""}>{"".join(content)}{script}</section>')

    options = ''.join(f'<option value="{row["frame"]}">Frame {row["frame"]}</option>' for row in report['frames'])
    limitations = ''.join(f'<li>{escape(line)}</li>' for line in report['limitations'])
    if report.get('camera_diagnostic'):
        limitations += '<li>Camera unavailable: ' + escape(report['camera_diagnostic']) + '</li>'
    if report.get('calibration_convention'):
        limitations += '<li>' + escape(report['calibration_convention']) + '</li>'
    definitions = ''.join(f'<dt>{escape(key)}</dt><dd>{escape(value)}</dd>' for key, value in report['definitions'].items())
    document = ('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Untrained / trained LiDAR comparison</title><style>'
        'body{margin:0;background:#111827;color:#e2e8f0;font:15px system-ui,sans-serif}main{max-width:1600px;margin:auto;padding:20px}'
        '.columns{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:16px}'
        'figure{margin:0}img{width:100%;height:auto}figcaption{padding:10px;font-weight:bold}h2{margin-top:32px}'
        'table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #475569;text-align:left}'
        'details{padding:12px;background:#1e293b}select{padding:8px}dd{margin-bottom:12px}</style></head><body><main>'
        '<h1>Without training / with training</h1><p>One fresh random-initialized model and one trained checkpoint; '
        'same scene and sampling seed. Images use recorded RGB backgrounds. Pixel/voxel metrics were computed on the compute machine.</p>'
        f'<p>Voxel size: {report["voxel_size_m"]:g} m · Shared sampling seed: {report["sampling_seed"]} · '
        f'Initialization seed: {report["initialization_seed"]} · Trained checkpoint epoch: {report["checkpoint_epoch"]}</p>'
        f'<label>Temporal frame <select onchange="showFrame(this.value)">{options}</select></label>'
        f'{"".join(sections)}<details><summary>Definitions and limits</summary><dl>{definitions}</dl><ul>{limitations}</ul></details>'
        '<script>function showFrame(value){document.querySelectorAll(".frame").forEach(section=>{'
        'section.hidden=section.dataset.frame!==value;});requestAnimationFrame(()=>{document.querySelectorAll(".plotly-graph-div")'
        '.forEach(graph=>{if(graph.offsetParent!==null)Plotly.Plots.resize(graph);});});}</script></main></body></html>')
    path = directory / 'comparison.html'
    path.write_text(document, encoding='utf-8')
    return path
