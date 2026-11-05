"""Descriptive pixel/voxel discrepancies on aligned LiDAR data; no model imports."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import jensenshannon


def distribution_gap(a, b):
    """Jensen-Shannon distance (base 2) and total variation of count distributions."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.sum() == 0 or b.sum() == 0:
        return {'js_distance': None, 'total_variation': None}
    a, b = a / a.sum(), b / b.sum()
    distance = float(jensenshannon(a, b, base=2))
    return {'js_distance': distance if np.isfinite(distance) else None,
            'total_variation': float(abs(a - b).sum() / 2)}


def pixel_gap(a, b):
    """Compare registered sparse camera-depth images without filling missing pixels."""
    if a.shape != b.shape:
        raise ValueError('Pixel comparisons need identical calibrated image grids')
    valid_a, valid_b = np.isfinite(a), np.isfinite(b)
    shared, union = valid_a & valid_b, valid_a | valid_b
    error = a[shared] - b[shared]
    metrics = {'occupied_pixels_a': int(valid_a.sum()), 'occupied_pixels_b': int(valid_b.sum()),
        'shared_pixels': int(shared.sum()), 'union_pixels': int(union.sum()),
        'occupancy_iou': float(shared.sum() / union.sum()) if union.any() else None,
        'occupancy_disagreement': float((valid_a ^ valid_b).sum() / union.sum()) if union.any() else None,
        'depth_mae_m': float(abs(error).mean()) if len(error) else None,
        'depth_rmse_m': float(np.sqrt(np.mean(error**2))) if len(error) else None,
        'depth_p95_error_m': float(np.quantile(abs(error), .95)) if len(error) else None}
    depth_error = np.full(a.shape, np.nan, dtype='float32')
    depth_error[shared] = abs(error)
    # -1 means only A, +1 only B, 0 shared/empty; NaN means empty in both.
    occupancy = np.where(union, valid_b.astype(float) - valid_a.astype(float), np.nan).astype('float32')
    return metrics, depth_error, occupancy


def voxel_gap(a, b, size):
    """Use one sparse sensor-frame voxel grid with fixed origin (0,0,0)."""
    if not np.isfinite(size) or size <= 0:
        raise ValueError('Voxel size must be positive and finite')
    a, b = np.asarray(a), np.asarray(b)
    cells_a = np.floor(a / size).astype('int64')
    cells_b = np.floor(b / size).astype('int64')
    cells, inverse = np.unique(np.concatenate((cells_a, cells_b)), axis=0, return_inverse=True)
    count_a = np.bincount(inverse[:len(a)], minlength=len(cells))
    count_b = np.bincount(inverse[len(a):], minlength=len(cells))
    shared = (count_a > 0) & (count_b > 0)
    metrics = {'voxel_size_m': size, 'points_a': len(a), 'points_b': len(b),
        'occupied_voxels_a': int((count_a > 0).sum()), 'occupied_voxels_b': int((count_b > 0).sum()),
        'union_voxels': len(cells), 'shared_voxels': int(shared.sum()),
        'occupancy_iou': float(shared.mean()) if len(cells) else None,
        'occupancy_disagreement': float(1 - shared.mean()) if len(cells) else None,
        **distribution_gap(count_a, count_b), 'chamfer_mean_m': None, 'nearest_p95_m': None}
    if len(a) and len(b):
        forward = cKDTree(b).query(a, workers=-1)[0]
        reverse = cKDTree(a).query(b, workers=-1)[0]
        metrics['chamfer_mean_m'] = float((forward.mean() + reverse.mean()) / 2)
        metrics['nearest_p95_m'] = float(np.quantile(np.concatenate((forward, reverse)), .95))
    density_a = count_a / max(1, len(a))
    density_b = count_b / max(1, len(b))
    return metrics, (cells + .5) * size, density_b - density_a


def covariates(points, intensity, max_range):
    """Shared physical bins include all valid ranges/heights/intensities."""
    values = {'range_m': (np.linalg.norm(points, axis=1), np.linspace(0, max_range, 65)),
              'height_m': (points[:, 2], np.linspace(-max_range, max_range, 129))}
    if intensity is not None:
        values['intensity'] = (intensity, np.linspace(0, 1, 33))
    result = {}
    for name, (data, edges) in values.items():
        # End bins retain out-of-range values rather than silently losing mass.
        counts = np.histogram(np.clip(data, edges[0], edges[-1]), edges)[0]
        result[name] = {'edges': edges.tolist(), 'counts': counts.tolist()}
    return result


def overlay_rgb_gap(a, b, depth_a, depth_b):
    """Return RGB metrics and a per-pixel MAE map within the projected foreground."""
    from scipy.ndimage import binary_dilation

    if a.shape != b.shape or a.shape[:2] != depth_a.shape:
        raise ValueError('Overlay RGB and depth dimensions differ')
    roi = binary_dilation(np.isfinite(depth_a) | np.isfinite(depth_b), structure=np.ones((3, 3)))
    roi[:25] = False  # draw_overlay's black strip includes row 24.
    difference = (a.astype(float) - b.astype(float)) / 255
    metrics = {'foreground_pixels': int(roi.sum()),
        'full_image_mae_0_1': float(abs(difference[25:]).mean()) if a.shape[0] > 25 else None,
        'foreground_mae_0_1': float(abs(difference[roi]).mean()) if roi.any() else None,
        'foreground_rmse_0_1': float(np.sqrt(np.mean(difference[roi]**2))) if roi.any() else None}
    histogram_a, histogram_b = [], []
    for channel in range(3):
        histogram_a.extend(np.histogram(a[..., channel][roi], np.linspace(0, 256, 33))[0])
        histogram_b.extend(np.histogram(b[..., channel][roi], np.linspace(0, 256, 33))[0])
    metrics.update(distribution_gap(histogram_a, histogram_b))
    error = np.full(roi.shape, np.nan, dtype='float32')
    error[roi] = abs(difference[roi]).mean(axis=1)
    return metrics, error
