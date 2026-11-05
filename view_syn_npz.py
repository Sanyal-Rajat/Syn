"""Inspect a numeric Syn NPZ and export an offline, interactive LiDAR viewer."""
import argparse
from html import escape
import json
from pathlib import Path

import numpy as np


def load_frame(path, frame=0, manifest=None):
    with np.load(path, allow_pickle=False) as data:
        print('Arrays:', ', '.join(data.files))
        if not {'range', 'valid'} <= set(data.files):
            raise ValueError('Expected a generated/prepared Syn NPZ with range and valid arrays; raw publisher recordings use another schema.')
        ranges, valid = data['range'], data['valid']
        print('Range shape:', ranges.shape, 'dtype:', ranges.dtype)
        if ranges.shape != valid.shape or ranges.ndim not in (2, 3):
            raise ValueError('range/valid must have matching [H,W] or [T,H,W] shapes')
        count = ranges.shape[0] if ranges.ndim == 3 else 1
        if not 0 <= frame < count:
            raise ValueError(f'Frame must be between 0 and {count - 1}')
        image = ranges[frame] if ranges.ndim == 3 else ranges
        mask = valid[frame] if valid.ndim == 3 else valid
        mask = mask.astype(bool) & np.isfinite(image) & (image > 0)
        intensity = None
        if 'intensity' in data.files:
            values = data['intensity']
            if values.shape != ranges.shape:
                raise ValueError('Intensity shape does not match range')
            present = data['has_intensity'] if 'has_intensity' in data.files else np.asarray(True)
            present = bool(present) if present.ndim == 0 else bool(present[frame])
            if present:
                intensity = values[frame] if ranges.ndim == 3 else values
        if 'elevations' in data.files:
            elevations = data['elevations']
        elif manifest:
            elevations = np.asarray(json.loads(Path(manifest).read_text())['sensor_meta']['elevations'])
        else:
            raise ValueError('Beam elevations missing. For prepared frame caches, supply --manifest /path/to/manifest.json')
        if elevations.shape != (image.shape[0],) or not np.isfinite(elevations).all():
            raise ValueError('Expected one finite beam elevation in radians per range-image row')
    ring, column = np.nonzero(mask)
    yaw = (column + .5) / image.shape[1] * 2 * np.pi - np.pi
    distance, pitch = image[ring, column], elevations[ring]
    points = np.column_stack((distance * np.cos(pitch) * np.cos(yaw),
                              distance * np.cos(pitch) * np.sin(yaw), distance * np.sin(pitch)))
    print(f'Frame {frame}/{count - 1}: {mask.sum():,} valid returns ({mask.mean():.1%})')
    return image, mask, intensity, elevations, points


def visualize(path, frame=0, output=None, manifest=None, max_points=30000):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if max_points < 1:
        raise ValueError('max-points must be positive')
    path = Path(path)
    output = Path(output) if output else path.with_suffix('.html')
    if output.resolve() in {path.resolve(), Path(manifest).resolve() if manifest else path.resolve()}:
        raise ValueError('HTML output must not overwrite an input file')
    ranges, valid, intensity, elevations, points = load_frame(path, frame, manifest)
    fig = make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'type': 'scene', 'colspan': 2}, None]],
                        row_heights=[.35, .65], subplot_titles=('Range (m)', 'Intensity (0–1)', 'Sensor-frame point cloud'))
    azimuth = np.rad2deg((np.arange(ranges.shape[1]) + .5) / ranges.shape[1] * 2 * np.pi - np.pi)
    angles = np.rad2deg(elevations)
    fig.add_trace(go.Heatmap(z=np.where(valid, ranges, np.nan), x=azimuth, y=angles,
                            colorscale='Turbo', colorbar={'x': .46, 'y': .85, 'len': .3}), row=1, col=1)
    if intensity is not None:
        fig.add_trace(go.Heatmap(z=np.where(valid, intensity, np.nan), x=azimuth, y=angles, zmin=0, zmax=1,
                                colorscale='Gray', colorbar={'x': 1.02, 'y': .85, 'len': .3}), row=1, col=2)
    if len(points) > max_points:
        points = points[np.random.default_rng(0).choice(len(points), max_points, replace=False)]
    fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='markers',
                             marker={'size': 1.5, 'color': points[:, 2], 'colorscale': 'Turbo'},
                             hovertemplate='x=%{x:.2f} m<br>y=%{y:.2f} m<br>z=%{z:.2f} m<extra></extra>'), row=2, col=1)
    fig.update_xaxes(title_text='Azimuth (degrees)', row=1)
    fig.update_yaxes(title_text='Elevation (degrees)', row=1)
    fig.update_layout(title=f'{escape(path.name)} · frame {frame} · {valid.sum():,} returns', height=950,
                      scene={'xaxis_title': 'X (m)', 'yaxis_title': 'Y (m)', 'zaxis_title': 'Z (m)', 'aspectmode': 'data'},
                      showlegend=False, template='plotly_white')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output), include_plotlyjs=True, auto_open=False)
    print('Open in your browser:', output.resolve())
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('npz', type=Path)
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--manifest', type=Path, help='Beam metadata for prepared caches without embedded elevations')
    parser.add_argument('--max-points', type=int, default=30000, help='Maximum displayed 3D points; range images retain all returns')
    args = parser.parse_args()
    try:
        visualize(args.npz, args.frame, args.output, args.manifest, args.max_points)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
