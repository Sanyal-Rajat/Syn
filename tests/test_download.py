import json
from pathlib import Path
import zipfile

import huggingface_hub
import pytest

from lidarpoc.config import RunConfig
from lidarpoc.data import download
from lidarpoc.data.split_manifest import load_manifest
from lidarpoc.provenance import license_gate
from lidarpoc.runner import run


@pytest.fixture
def publisher(tmp_path, monkeypatch):
    archives, calls = {}, []
    for name in download.ARCHIVES:
        split = 'train' if name.startswith('train_part') else name.removesuffix('.zip')
        path = tmp_path / name
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr(f'{split}/{name}/{name}_scan_list.npz', b'scan data')
            archive.writestr(f'{split}/{name}/{name}_video_list.npz', b'camera data')
        archives[name] = path

    def fetch(**kwargs):
        calls.append(kwargs)
        return str(archives[kwargs['filename']])

    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', fetch)
    return archives, calls


def test_pinned_download_extracts_only_scans_and_reuses_completed_data(tmp_path, monkeypatch, publisher):
    archives, calls = publisher
    root = tmp_path / 'project' / 'datasets' / 'pandar128'
    monkeypatch.setattr(download, 'DATASET_ROOT', root)
    asset = license_gate()['dataset']
    stages = []
    assert download.ensure_dataset('', asset, lambda *args: stages.append(args)) == str(root)
    assert len(list(root.glob('**/*_scan_list.npz'))) == 6
    assert not list(root.glob('**/*_video_list.npz'))
    assert len(calls) == 6
    assert all(call['repo_id'] == asset['repo_id'] and call['revision'] == asset['revision']
               and call['repo_type'] == 'dataset' and call['local_dir'] == root / '.downloads' for call in calls)
    receipt = json.loads((root / '.download.json').read_text())
    assert set(receipt['archives']) == set(archives)
    assert any('downloading' in stage[0] for stage in stages)
    download.ensure_dataset('', asset, lambda *_: None)
    assert len(calls) == 6
    missing = receipt['archives']['train_part2.zip'][0]['path']
    (root / missing).unlink()
    download.ensure_dataset('', asset, lambda *_: None)
    assert len(calls) == 7 and calls[-1]['filename'] == 'train_part2.zip'
    assert (root / missing).read_bytes() == b'scan data'


def test_interrupted_extraction_is_not_mistaken_for_a_complete_dataset(tmp_path, monkeypatch, publisher):
    archives, calls = publisher
    with zipfile.ZipFile(archives['test.zip'], 'a') as archive:
        archive.writestr('test/second/second_scan_list.npz', b'another scan')
    copy = download.shutil.copyfileobj

    def interrupted(source, destination, **kwargs):
        if source.name.endswith('second_scan_list.npz'):
            destination.write(b'partial')
            raise OSError('interrupted')
        copy(source, destination, **kwargs)

    root = tmp_path / 'data'
    asset = license_gate()['dataset']
    monkeypatch.setattr(download.shutil, 'copyfileobj', interrupted)
    with pytest.raises(OSError, match='interrupted'):
        download.ensure_dataset(root, asset, lambda *_: None)
    assert all(list((root / split).glob('**/*_scan_list.npz')) for split in ('train', 'validation', 'test'))
    assert 'test.zip' not in json.loads((root / '.download.json').read_text())['archives']
    monkeypatch.setattr(download.shutil, 'copyfileobj', copy)
    download.ensure_dataset(root, asset, lambda *_: None)
    assert len(calls) == 7 and calls[-1]['filename'] == 'test.zip'
    assert (root / 'test/second/second_scan_list.npz').read_bytes() == b'another scan'
    assert not list(root.glob('**/*.partial'))


def test_existing_manual_dataset_needs_no_download(prepared, publisher):
    _, calls = publisher
    root = Path(prepared).parents[1] / 'raw'
    assert download.ensure_dataset(root, license_gate()['dataset'], lambda *_: None) == str(root)
    assert not calls


@pytest.mark.parametrize('invalid', ['duplicate', 'no_scans', 'revision'])
def test_rejects_incomplete_or_conflicting_downloads(tmp_path, publisher, invalid):
    archives, _ = publisher
    root = tmp_path / 'data'
    asset = license_gate()['dataset']
    if invalid == 'revision':
        root.mkdir()
        (root / '.download.json').write_text(json.dumps({'repo_id': asset['repo_id'], 'revision': 'other'}))
    else:
        with zipfile.ZipFile(archives['train_part2.zip'], 'w') as archive:
            filename = 'train/train_part1.zip/train_part1.zip_scan_list.npz' if invalid == 'duplicate' else 'video.npz'
            archive.writestr(filename, b'conflicting data')
    with pytest.raises(ValueError, match={'duplicate': 'duplicate', 'no_scans': 'no LiDAR', 'revision': 'revision'}[invalid]):
        download.ensure_dataset(root, asset, lambda *_: None)


def test_prepare_downloads_into_project_folder_and_records_server_path(prepared, tmp_path, monkeypatch, publisher):
    archives, calls = publisher
    raw = Path(prepared).parents[1] / 'raw'
    root = tmp_path / 'project/datasets/pandar128'
    monkeypatch.setattr(download, 'DATASET_ROOT', root)
    for name, path in archives.items():
        split = 'train' if name.startswith('train_part') else name.removesuffix('.zip')
        with zipfile.ZipFile(path, 'w') as archive:
            for index, source in enumerate(sorted((raw / split).glob('**/*_scan_list.npz'))):
                trace = f'{name}_{index}'
                archive.write(source, f'{split}/{trace}/{trace}_scan_list.npz')
    config = RunConfig(action='prepare', input_frame='sensor', elevations_file=str(tmp_path / 'beams.txt'),
                       height=4, width=16, max_frames=4)
    report = run(config, tmp_path / 'run')
    assert len(calls) == 6
    assert report['dataset_root'] == str(root)
    assert json.loads((tmp_path / 'run/request.json').read_text())['dataset_root'] == str(root)
    manifest = load_manifest(report['manifest'], verify_files=True)
    assert len(manifest['splits']['train']) == 8
    assert len(manifest['splits']['validation']) == len(manifest['splits']['test']) == 2
