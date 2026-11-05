"""Camera geometry and image drawing, independent of training and job execution."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def mounting_transform(mount):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler('ZYX', [float(mount[f'{axis}_deg'])
        for axis in ('yaw', 'pitch', 'roll')], degrees=True).as_matrix()
    matrix[:3, 3] = [float(mount[f'pos_{axis}_m']) for axis in 'xyz']
    return matrix


def load_camera(path, model='rational'):
    """Load publisher mounts or an explicit optical camera_from_sensor calibration."""
    document = json.loads(Path(path).read_text())
    if 'camera_from_sensor' in document:
        camera = {key: np.asarray(document[key], dtype=float)
                  for key in ('camera_from_sensor', 'camera_matrix', 'image_size')}
        camera['distortion'] = np.asarray(document.get('distortion', []), dtype=float)
        camera['model'] = document.get('model', model)
        camera['convention'] = 'explicit optical transform; image_size is width,height'
    else:
        sensor = mounting_transform(document['Pandar 128']['Mounting position'])
        entry = document['ADAS Front Camera Wide']
        body = mounting_transform(entry['Mounting position'])
        optical_from_body = np.eye(4)
        optical_from_body[:3, :3] = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
        parameters = entry['Sensor specific parameters']
        intrinsics, distortion = parameters['intrinsics'], parameters['distortion']
        camera = {'camera_from_sensor': optical_from_body @ np.linalg.inv(body) @ sensor,
                  'camera_matrix': np.array([[float(intrinsics['f_x']), 0, float(intrinsics['c_x'])],
                                            [0, float(intrinsics['f_y']), float(intrinsics['c_y'])], [0, 0, 1]]),
                  'image_size': np.array([int(parameters['video_size'][key]) for key in ('width', 'height')]),
                  'distortion': np.array([float(distortion.get(key, 0))
                      for key in ('K1', 'K2', 'P1', 'P2', 'K3', 'K4', 'K5', 'K6')]), 'model': model,
                  'convention': 'Publisher mounting interpretation: vehicle/body X forward, Y left, Z up; '
                      'ZYX yaw/pitch/roll; optical X right, Y down, Z forward. '
                      'Rational model interprets K4 as the first denominator coefficient. '
                      'These conventions are assumptions; check real-LiDAR alignment or supply an explicit calibration.'}
    transform, intrinsics = camera['camera_from_sensor'], camera['camera_matrix']
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-5)):
        raise ValueError('camera_from_sensor must be a rigid optical-frame transform')
    if (intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all()
            or min(intrinsics[0, 0], intrinsics[1, 1]) <= 0 or not np.allclose(intrinsics[2], [0, 0, 1])
            or camera['image_size'].shape != (2,) or not np.isfinite(camera['image_size']).all()
            or (camera['image_size'] <= 0).any()):
        raise ValueError('Invalid camera intrinsics or image_size')
    if (camera['model'] not in ('rational', 'pinhole') or camera['distortion'].ndim != 1
            or len(camera['distortion']) not in (0, 4, 5, 8) or not np.isfinite(camera['distortion']).all()):
        raise ValueError('Use rational/pinhole with 0, 4, 5 or 8 OpenCV-ordered distortion coefficients')
    return camera


def project_points(points, camera, image_size):
    """Project sensor-frame XYZ; retain the nearest return per image pixel."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('Expected sensor-frame XYZ [N,3]')
    size = np.asarray(image_size, dtype=float)
    scale = size / camera['image_size']
    if not np.isclose(scale[0], scale[1], rtol=.01):
        raise ValueError('Image aspect ratio differs from calibration; supply calibrated intrinsics for cropped images')
    transform = camera['camera_from_sensor']
    optical = points @ transform[:3, :3].T + transform[:3, 3]
    indices = np.flatnonzero(np.isfinite(optical).all(axis=1) & (optical[:, 2] > .05))
    optical = optical[indices]
    x, y = (optical[:, :2] / optical[:, 2, None]).T
    if camera['model'] == 'rational':
        k1, k2, p1, p2, k3, k4, k5, k6 = np.pad(camera['distortion'], (0, 8 - len(camera['distortion'])))
        r2 = x * x + y * y
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            radial = (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3) / (1 + k4 * r2 + k5 * r2**2 + k6 * r2**3)
            x, y = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    normalized = np.column_stack((x, y, np.ones(len(x))))
    pixels = (normalized @ camera['camera_matrix'].T)[:, :2] * scale
    inside = np.isfinite(pixels).all(axis=1) & (pixels >= 0).all(axis=1) & (pixels < size).all(axis=1)
    pixels, depth, indices = pixels[inside].astype(int), optical[inside, 2], indices[inside]
    order = np.argsort(depth, kind='stable')
    _, nearest = np.unique(pixels[order, 1] * int(size[0]) + pixels[order, 0], return_index=True)
    selected = order[nearest]
    return pixels[selected], depth[selected], indices[selected]


def draw_overlay(image, points, camera, label):
    """Draw depth-coloured returns on recorded RGB, without changing either input."""
    from PIL import Image, ImageDraw

    image = image.convert('RGB').copy()
    pixels, depth, _ = project_points(points, camera, image.size)
    # Blue at 0 m, green at 40 m, red at 80 m and beyond, shared across real/synthetic.
    fraction = np.clip(depth / 80, 0, 1)
    colors = (255 * np.column_stack((np.clip(2 * fraction - 1, 0, 1),
                                    1 - abs(2 * fraction - 1), np.clip(1 - 2 * fraction, 0, 1)))).astype('uint8')
    layer = Image.new('RGBA', image.size)
    draw = ImageDraw.Draw(layer)
    # Draw distant markers first so nearer returns also win where marker discs overlap.
    for index in np.argsort(depth)[::-1]:
        x, y = pixels[index]
        draw.ellipse((int(x) - 1, int(y) - 1, int(x) + 1, int(y) + 1), fill=(*map(int, colors[index]), 220))
    image = Image.alpha_composite(image.convert('RGBA'), layer).convert('RGB')
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 24), fill='black')
    draw.text((8, 6), f'{label} on recorded RGB | {len(pixels):,} pixels | depth: blue 0m, green 40m, red 80m+', fill='white')
    return image, len(pixels)
