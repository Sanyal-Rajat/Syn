"""Download the pinned Pandar128 archives and cache only extracted LiDAR scans."""
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

from ..io import write_json

DATASET_ROOT = Path(__file__).resolve().parents[2] / 'datasets' / 'pandar128'
ARCHIVES = tuple(f'train_part{index}.zip' for index in range(1, 5)) + ('validation.zip', 'test.zip')


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
        if records and all((root / row['path']).is_file() and (root / row['path']).stat().st_size == row['size']
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
            for scan_index, entry in enumerate(scans):
                filename = PurePosixPath(entry.filename).name
                trace = filename.removesuffix('_scan_list.npz')
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
        write_json(receipt_path, receipt)
        progress(f'extracted {name}', index + 1, len(ARCHIVES))
    return str(root)
