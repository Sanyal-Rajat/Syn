"""Read publisher camera records and match odometry timestamps without index pairing."""
from contextlib import contextmanager
import io
from pathlib import Path
import shutil
import struct
import tempfile
import zipfile

import numpy as np

from .download import ensure_camera_files


@contextmanager
def read_records(path):
    """Memory-map numeric np.savez records; bound RAM for publisher object variants."""
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        names = [name for name in ('data.npy', 'video_list.npy', 'odom_list.npy') if name in archive.namelist()]
        if len(names) != 1:
            raise ValueError(f'{path.name}: expected one data/video_list/odom_list array')
        entry = archive.getinfo(names[0])
        with archive.open(entry) as stream:
            version = np.lib.format.read_magic(stream)
            reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, fortran, dtype = reader(stream)
        if dtype.hasobject:
            import psutil
            if entry.file_size > psutil.virtual_memory().available * .4:
                raise ValueError(f'{path.name}: object camera archive exceeds safe available RAM; convert to numeric timestamped records first')
            with np.load(path, allow_pickle=True) as source:
                yield source[names[0].removesuffix('.npy')]
        elif entry.compress_type == zipfile.ZIP_STORED:
            # np.savez (the publisher writer) stores each NPY member without compression.
            with path.open('rb') as stream:
                stream.seek(entry.header_offset)
                header = stream.read(30)
                filename_size, extra_size = struct.unpack('<HH', header[26:30])
                stream.seek(filename_size + extra_size, 1)
                np.lib.format.read_magic(stream)
                reader(stream)
                offset = stream.tell()
            array = np.memmap(path, mode='r', dtype=dtype, shape=shape, offset=offset, order='F' if fortran else 'C')
            try:
                yield array
            finally:
                array._mmap.close()
        else:
            with tempfile.TemporaryDirectory(prefix='syn-camera-') as folder:
                target = Path(folder) / 'records.npy'
                with archive.open(entry) as source, target.open('wb') as destination:
                    shutil.copyfileobj(source, destination, length=8 << 20)
                array = np.load(target, mmap_mode='r', allow_pickle=False)
                try:
                    yield array
                finally:
                    array._mmap.close()


def field(records, name):
    if name in (records.dtype.names or ()):
        return records[name]
    if records.dtype == object and records.ndim == 1 and all(isinstance(record, dict) and name in record for record in records):
        return np.asarray([record[name] for record in records])
    raise ValueError(f'Publisher camera/odometry records need the {name!r} field; frame-index matching is disabled')


def seconds(values, unit):
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.datetime64):
        result = values.astype('datetime64[ns]').astype('int64') * 1e-9
        if np.isnat(values).any():
            raise ValueError('Invalid camera/odometry timestamp')
    else:
        result = values.astype('float64') * {'s': 1, 'ms': 1e-3, 'us': 1e-6, 'ns': 1e-9}[unit]
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError('Camera/odometry timestamps must be finite nonempty vectors')
    return result


def matching_camera_frames(scan_path, indices, dataset, progress, unit='s', max_skew_ms=50, width=1280):
    from PIL import Image

    video_path, odom_path = ensure_camera_files(scan_path, dataset, progress)
    with read_records(odom_path) as odometry:
        times = seconds(field(odometry, 't'), unit)
    indices = np.asarray(indices, dtype=int)
    if (indices < 0).any() or (indices >= len(times)).any():
        raise ValueError('Conditioning frame indices exceed the source odometry records')
    frames = []
    with read_records(video_path) as video:
        video_times = seconds(field(video, 't'), unit)
        images = field(video, 'data')
        if len(images) != len(video_times):
            raise ValueError('Camera image and timestamp counts differ')
        for index in indices:
            nearest = int(np.argmin(abs(video_times - times[index])))
            skew = float(abs(video_times[nearest] - times[index]) * 1000)
            if skew > max_skew_ms:
                raise ValueError(f'Camera timestamp mismatch at LiDAR frame {index}: {skew:.3f} ms exceeds {max_skew_ms:g} ms')
            image = images[nearest]
            if isinstance(image, (bytes, bytearray)) or (isinstance(image, np.ndarray) and image.ndim == 1 and image.dtype == np.uint8):
                with Image.open(io.BytesIO(bytes(image))) as encoded:
                    image = encoded.convert('RGB')
            else:
                image = np.asarray(image)
                if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] not in (3, 4):
                    raise ValueError('Expected an RGB uint8 camera image or encoded image bytes')
                image = Image.fromarray(image).convert('RGB')
            image.thumbnail((width, width), Image.Resampling.LANCZOS)
            frames.append((image.copy(), {'source_scan': str(Path(scan_path).resolve()),
                'source_video': str(video_path), 'source_odometry': str(odom_path),
                'lidar_frame': int(index), 'camera_frame': nearest, 'lidar_timestamp_s': float(times[index]),
                'camera_timestamp_s': float(video_times[nearest]), 'skew_ms': skew}))
    return frames
