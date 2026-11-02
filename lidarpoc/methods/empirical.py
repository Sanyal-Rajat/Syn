"""Independent per-ring empirical sensor baseline fit on training windows only."""
import numpy as np


def sample(training, count, seed=0):
    rng = np.random.default_rng(seed)
    shape = training[0]['range'].shape
    pooled = {key: np.concatenate([row[key].reshape(-1, *shape[-2:]) for row in training])
              for key in ('range', 'valid', 'intensity')}
    result = []
    for _ in range(count):
        arrays = {key: np.zeros(shape, dtype=bool if key == 'valid' else 'float32') for key in pooled}
        for ring in range(shape[-2]):
            available = pooled['valid'][:, ring]
            indices = np.flatnonzero(available)
            if not len(indices):
                continue
            valid = rng.uniform(size=(shape[0], shape[-1])) < available.mean()
            choice = rng.choice(indices, size=valid.shape)
            arrays['valid'][:, ring] = valid
            for key in ('range', 'intensity'):
                arrays[key][:, ring] = np.where(valid, pooled[key][:, ring].reshape(-1)[choice], 0)
        arrays['has_intensity'] = training[0]['has_intensity']
        result.append(arrays)
    return result
