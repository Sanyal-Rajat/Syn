"""Download pinned Pandar128 scans/odometry; extract camera traces on demand."""
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

from ..io import write_json

DATASET_ROOT = Path(__file__).resolve().parents[2] / 'datasets' / 'pandar128'
ARCHIVES = tuple(f'train_part{index}.zip' for index in range(1, 5)) + ('validation.zip', 'test.zip')


def ensure_camera_files(scan_path, dataset, progress):
    """Extract only the conditioning trace's video/odometry from pinned cached ZIPs."""
    scan = Path(scan_path).expanduser().resolve()
    trace = scan.name.removesuffix('_scan_list.npz')
    targets = [scan.with_name(trace + suffix) for suffix in ('_video_list.npz', '_odom_list.npz')]
    missing = [path for path in targets if not path.is_file()]
    if not missing:
        return targets
    root = next((parent for parent in scan.parents if (parent / '.download.json').is_file()
                 or (parent / '.downloads').is_dir()), None)
    if root is None:
        raise ValueError('Camera/odometry files are missing beside the source scan and no cached dataset archives were found')
    receipt_path = root / '.download.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
    if any(receipt.get(key, dataset[key]) != dataset[key] for key in ('repo_id', 'revision')):
        raise ValueError('Camera archive revision differs from the pinned dataset')
    names = [name for name, rows in receipt.get('archives', {}).items()
             if any((root / row['path']).resolve() == scan for row in rows)]
    archives = [root / '.downloads' / name for name in names] if names else sorted((root / '.downloads').glob('*.zip'))
    for archive_path in archives:
        if not archive_path.is_file() and archive_path.name in ARCHIVES:
            from huggingface_hub import hf_hub_download
            progress(f'downloading camera source {archive_path.name}')
            archive_path = Path(hf_hub_download(repo_id=dataset['repo_id'], revision=dataset['revision'],
                repo_type='dataset', filename=archive_path.name, local_dir=root / '.downloads'))
        with zipfile.ZipFile(archive_path) as archive:
            for target in missing[:]:
                matches = [entry for entry in archive.infolist()
                           if not entry.is_dir() and PurePosixPath(entry.filename).name == target.name]
                if len(matches) > 1:
                    raise ValueError(f'Ambiguous camera trace in {archive_path.name}: {target.name}')
                if not matches:
                    continue
                progress(f'extracting camera source {target.name}')
                temporary = target.with_suffix('.npz.partial')
                try:
                    with archive.open(matches[0]) as source, temporary.open('wb') as destination:
                        shutil.copyfileobj(source, destination, length=8 << 20)
                    if temporary.stat().st_size != matches[0].file_size:
                        raise ValueError(f'Incomplete camera extraction: {target.name}')
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
                missing.remove(target)
        if not missing:
            return targets
    raise ValueError('Camera source files absent from cached archives: ' + ', '.join(path.name for path in missing))


def ensure_dataset(root, dataset, progress):
    root = Path(root or DATASET_ROOT).expanduser().resolve()
    receipt_path = root / '.download.json'
    identity = {key: dataset[key] for key in ('repo_id', 'revision')}
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if any(receipt.get(key) != value for key, value in identity.items()):
            raise ValueError('Dataset download revision differs from the pinned source; use another directory')
    else:
        # Existing manually extracted data remains usable without a network request.
        if all(any((root / split).glob('**/*_scan_list.npz')) for split in ('train', 'validation', 'test')):
            return str(root)
        root.mkdir(parents=True, exist_ok=True)
        receipt = {**identity, 'archives': {}}
        write_json(receipt_path, receipt)

    from huggingface_hub import hf_hub_download

    for index, name in enumerate(ARCHIVES):
        records = receipt['archives'].get(name, [])
        if records and receipt.get('contents_version', {}).get(name) == 2 and all((root / row['path']).is_file() and (root / row['path']).stat().st_size == row['size']
                           for row in records):
            progress(f'reusing extracted {name}', index + 1, len(ARCHIVES))
            continue
        progress(f'downloading {name}', index, len(ARCHIVES))
        print(f'Downloading {name} into {root / ".downloads"}', flush=True)
        archive_path = hf_hub_download(**identity, repo_type='dataset', filename=name,
                                       local_dir=root / '.downloads')
        split = 'train' if name.startswith('train_part') else name.removesuffix('.zip')
        records, targets = [], set()
        with zipfile.ZipFile(archive_path) as archive:
            scans = [entry for entry in archive.infolist()
                     if not entry.is_dir() and entry.filename.endswith('_scan_list.npz')]
            if not scans:
                raise ValueError(f'{name}: no LiDAR *_scan_list.npz files found')
            scans += [entry for entry in archive.infolist() if not entry.is_dir() and entry.filename.endswith('_odom_list.npz')]
            for scan_index, entry in enumerate(scans):
                filename = PurePosixPath(entry.filename).name
                suffix = '_odom_list.npz' if filename.endswith('_odom_list.npz') else '_scan_list.npz'
                trace = filename.removesuffix(suffix)
                relative = Path(split) / trace / filename
                if trace in ('', '.', '..') or relative in targets or any(
                        row['path'] == relative.as_posix() for other, rows in receipt['archives'].items()
                        if other != name for row in rows):
                    raise ValueError(f'{name}: duplicate or invalid LiDAR trace {filename}')
                targets.add(relative)
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix('.npz.partial')
                progress(f'extracting {name}: {filename}', scan_index, len(scans))
                with archive.open(entry) as source, temporary.open('wb') as destination:
                    shutil.copyfileobj(source, destination, length=8 << 20)
                if temporary.stat().st_size != entry.file_size:
                    raise ValueError(f'{name}: incomplete scan extraction: {filename}')
                temporary.replace(target)
                records.append({'path': relative.as_posix(), 'size': entry.file_size})
        receipt['archives'][name] = records
        receipt.setdefault('contents_version', {})[name] = 2
        write_json(receipt_path, receipt)
        progress(f'extracted {name}', index + 1, len(ARCHIVES))
    return str(root)
