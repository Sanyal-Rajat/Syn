import numpy as np
import pytest

from lidarpoc.config import RunConfig
from lidarpoc.data.split_manifest import prepare
from lidarpoc.provenance import license_gate


@pytest.fixture
def prepared(tmp_path):
    elevations = np.deg2rad([-12., -4., 4., 12.])
    beam_file = tmp_path / 'beams.txt'
    np.savetxt(beam_file, np.rad2deg(elevations))
    dtype = [('Classification', 'u1'), ('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4')]
    for split_index, split in enumerate(('train', 'validation', 'test')):
        for trace_index in range(2):
            trace = f'fixture_{split}_{trace_index}'
            folder = tmp_path / 'raw' / split / trace
            folder.mkdir(parents=True)
            frames = np.empty(4, dtype=object)
            for frame_index in range(4):
                xyz = []
                for ring, elevation in enumerate(elevations):
                    for column in range(8 + (frame_index % 2)):
                        angle = (column + .5) / 16 * 2 * np.pi - np.pi
                        distance = 10 + 3 * split_index + 2 * trace_index + ring + frame_index * .4
                        xyz.append((0, distance * np.cos(elevation) * np.cos(angle),
                                    distance * np.cos(elevation) * np.sin(angle), distance * np.sin(elevation),
                                    40 + 20 * ring + frame_index))
                frames[frame_index] = np.array(xyz, dtype=dtype)
            np.savez_compressed(folder / f'{trace}_scan_list.npz', data=frames)
    config = RunConfig(action='prepare', dataset_root=str(tmp_path / 'raw'), input_frame='sensor',
                       elevations_file=str(beam_file), height=4, width=16, window=2, max_frames=4)
    manifest = prepare(config, tmp_path / 'prepared', license_gate())
    return manifest
