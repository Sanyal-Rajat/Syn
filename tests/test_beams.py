from pathlib import Path

import numpy as np
import pytest

from lidarpoc.config import RunConfig
from lidarpoc.data.scan_to_range import infer_elevations, project
from lidarpoc.data.split_manifest import load_manifest, prepare
from lidarpoc.provenance import license_gate


def points_at(angles):
    return 20 * np.column_stack((np.cos(angles), np.zeros_like(angles), np.sin(angles)))


def test_uneven_128_beam_returns_keep_sparse_outer_elevations_and_are_repeatable():
    rng = np.random.default_rng(42)
    truth = np.deg2rad(np.r_[np.linspace(-26, -5, 32), np.linspace(-4, 4, 64), np.linspace(5, 14, 32)])
    population = np.r_[np.full(32, 200), np.full(64, 6000), np.full(32, 200)]
    angles = np.repeat(truth, population) + rng.normal(0, np.deg2rad(.001), population.sum())
    xyz = points_at(angles)
    beams, diagnostics = infer_elevations([xyz], return_diagnostics=True)
    np.testing.assert_array_equal(beams, infer_elevations([xyz]))
    assert len(beams) == 128 and np.diff(beams).min() > 1e-5
    assert np.rad2deg(abs(beams - truth)).max() < .01
    assert diagnostics['p99_residual_deg'] < .01
    assert diagnostics['sampled_training_returns'] == 200000
    assert not diagnostics['factory_ring_ids']
    heldout = points_at(np.tile(truth, 10) + rng.normal(0, np.deg2rad(.001), 1280))
    assert project(heldout, None, beams)['rejected_fraction'] == 0


def test_beam_fit_keeps_strict_tolerance_and_actionable_failures():
    # Four rows cannot cover a continuous 80-degree field within 0.15 degrees.
    with pytest.raises(ValueError, match=r'p99 angular error.*limit.*coordinate frame'):
        infer_elevations([points_at(np.deg2rad(np.linspace(-40, 40, 10000)))], height=4)
    with pytest.raises(ValueError, match='Not enough training returns'):
        infer_elevations([], height=4)
    with pytest.raises(ValueError, match='distinct training elevations'):
        infer_elevations([np.tile([10., 0, 0], (1000, 1))], height=4)
    with pytest.raises(ValueError, match='finite XYZ'):
        infer_elevations([np.array([[1., 0, np.nan]])], height=4)


def test_automatic_preparation_exports_empirical_fit_diagnostics(prepared, tmp_path):
    raw = Path(prepared).parents[1] / 'raw'
    config = RunConfig(action='prepare', dataset_root=str(raw), input_frame='sensor',
                       height=4, width=16, max_frames=4, infer_beams=True)
    manifest = load_manifest(prepare(config, tmp_path / 'inferred', license_gate()), verify_files=True)
    sensor = manifest['sensor_meta']
    np.testing.assert_allclose(np.rad2deg(sensor['elevations']), [-12, -4, 4, 12], atol=1e-4)
    assert sensor['beam_source'] == 'empirical_training_elevation_bins_not_factory_ring_ids'
    assert sensor['beam_fit']['method'] == 'minimax_angular_coverage'
    assert sensor['beam_fit']['p99_residual_deg'] < sensor['beam_fit']['tolerance_deg']
