import math
import torch


def masked_mse(prediction, target, mask):
    weights = mask.to(prediction.dtype)
    return ((prediction - target).square() * weights).sum() / weights.sum().clamp_min(1)


def encode(batch, max_range):
    valid = batch['valid'].bool()
    ranges = 2 * torch.log1p(batch['range'].clamp_min(0)) / math.log1p(max_range) - 1
    has_intensity = batch['has_intensity'].bool()[..., None, None]
    values = torch.stack((torch.where(valid, ranges, 0), valid.float() * 2 - 1,
                          torch.where(valid & has_intensity, batch['intensity'] * 2 - 1, 0)), dim=2)
    mask = torch.stack((valid, torch.ones_like(valid), valid & has_intensity), dim=2)
    return values.flatten(1, 2), mask.flatten(1, 2)
