"""Small labeled point-segmentation probe with fixed real/synthetic label budgets."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def labeled_points(samples, sensor):
    features, labels = [], []
    height, width = len(sensor['elevations']), sensor['width']
    elevation = np.broadcast_to(np.asarray(sensor['elevations'])[None, :, None], (1, height, width))
    azimuth = np.broadcast_to(np.asarray(sensor['azimuths'])[None, None], (1, height, width))
    for sample in samples:
        if 'labels' not in sample:
            continue
        target = np.asarray(sample['labels'])
        valid = np.asarray(sample['valid']).astype(bool) & (target >= 0)
        shape = np.asarray(sample['range']).shape
        channels = [np.asarray(sample['range']) / sensor['max_range_m'], np.asarray(sample['intensity']),
                    np.broadcast_to(elevation, shape), np.broadcast_to(np.cos(azimuth), shape),
                    np.broadcast_to(np.sin(azimuth), shape)]
        features.append(np.stack(channels, -1)[valid])
        labels.append(target[valid])
    if not labels or not sum(len(value) for value in labels):
        raise ValueError('Downstream evaluation requires confirmed semantic labels on visible returns')
    return np.concatenate(features).astype('float32'), np.concatenate(labels).astype('int64')


def metric(model, samples, sensor, classes):
    rows = []
    for sample in samples:
        x, y = labeled_points([sample], sensor)
        if (y >= classes).any():
            raise ValueError('Held-out labels contain classes absent from the frozen training label map')
        with torch.no_grad():
            prediction = model(torch.from_numpy(x)).argmax(-1).numpy()
        confusion = np.bincount(classes * y + prediction, minlength=classes**2).reshape(classes, classes)
        rows.append((sample['sequence_id'], confusion))
    grouped = {}
    for identity, confusion in rows:
        grouped[identity] = grouped.get(identity, np.zeros_like(confusion)) + confusion
    output = []
    for identity, confusion in grouped.items():
        union = confusion.sum(0) + confusion.sum(1) - np.diag(confusion)
        iou = np.divide(np.diag(confusion), union, out=np.full(classes, np.nan), where=union > 0)
        output.append({'sequence_id': identity, 'mean_iou': float(np.nanmean(iou)),
                       'accuracy': float(np.trace(confusion) / confusion.sum()), 'confusion': confusion.tolist()})
    return output


def fit(real, synthetic, development, sensor, real_budget, synthetic_budget, steps, seed):
    rx, ry = labeled_points(real, sensor)
    sx, sy = labeled_points(synthetic, sensor)
    classes = int(max(ry.max(), sy.max())) + 1
    if classes < 2:
        raise ValueError('A task-transfer probe needs at least two confirmed training classes')
    rng = np.random.default_rng(seed)
    real_index = rng.permutation(len(ry))[:real_budget]
    synthetic_index = rng.permutation(len(sy))[:synthetic_budget]
    if len(real_index) != real_budget or len(synthetic_index) != synthetic_budget:
        raise ValueError('Requested identical label budget exceeds available labeled points')
    x = torch.from_numpy(np.concatenate([rx[real_index], sx[synthetic_index]]))
    y = torch.from_numpy(np.concatenate([ry[real_index], sy[synthetic_index]]))
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    for _ in range(steps):
        index = torch.randperm(len(y))[:min(256, len(y))]
        loss = F.cross_entropy(model(x[index]), y[index])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return model, {'classes': classes, 'real_labels': real_budget, 'synthetic_labels': synthetic_budget,
                   'steps': steps, 'development': metric(model, development, sensor, classes)}
