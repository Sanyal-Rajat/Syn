"""Pin source/data evidence and record runtime dependencies in every own checkpoint."""
from importlib import metadata
import json
from pathlib import Path

from .data.split_manifest import digest

LICENSES = Path(__file__).resolve().parents[1] / 'LICENSES'


def license_gate(manifest=None):
    evidence = json.loads((LICENSES / 'manifest.json').read_text())
    for name in ('model', 'dataset'):
        asset = evidence[name]
        text = (LICENSES / asset['license_file']).read_text()
        if digest(LICENSES / asset['license_file']) != asset['license_sha256'] or 'MIT License' not in text:
            raise ValueError(f'{name} license evidence is missing or changed')
    if evidence['pretrained_weights'] or evidence['extra_assets']:
        raise ValueError('Pretrained weights and external assets are excluded by the selected specification')
    if manifest is not None:
        if manifest['dataset'] != evidence['dataset']['repo_id'] or manifest['revision'] != evidence['dataset']['revision']:
            raise ValueError('Dataset identity/revision differs from pinned publisher evidence')
    return evidence


def runtime_lock():
    return sorted([{'name': distribution.metadata['Name'], 'version': distribution.version,
                    'license': distribution.metadata.get('License-Expression') or distribution.metadata.get('License') or 'UNRESOLVED'}
                   for distribution in metadata.distributions()], key=lambda row: row['name'].lower())


def provenance(manifest_path, config):
    from .models import lidardm_adapter, lidardm_unet
    return {'initialization': 'random', 'training_manifest_sha256': digest(manifest_path),
            'model_source_commit': license_gate()['model']['commit'],
            'adapter_sha256': digest(lidardm_adapter.__file__), 'unet_sha256': digest(lidardm_unet.__file__),
            'license_manifest': license_gate(), 'runtime_dependency_lock': runtime_lock(), 'config': config}


def validate_checkpoint(checkpoint, manifest_path):
    provenance = checkpoint.get('provenance', {})
    evidence = license_gate()
    if provenance.get('initialization') != 'random' or provenance.get('model_source_commit') != evidence['model']['commit']:
        raise ValueError('Checkpoint lacks the application\'s clean random-initialization provenance')
    if provenance.get('training_manifest_sha256') != digest(manifest_path):
        raise ValueError('Checkpoint training manifest does not match the selected frozen dataset manifest')
    if provenance.get('license_manifest') != evidence or not provenance.get('runtime_dependency_lock'):
        raise ValueError('Checkpoint lacks matching license evidence or runtime dependency lock')
