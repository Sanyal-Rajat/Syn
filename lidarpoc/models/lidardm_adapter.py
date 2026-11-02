"""LidarDM conditional-UNet frontend adapted to masked, temporal range images.

This is a pixel/ray-space adaptation, not the upstream SDF/VAE world-generation
pipeline. It creates no pretrained VAE, asset bank, map encoder or external weights.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from diffusers import DDPMScheduler, DDIMScheduler

from .lidardm_unet import UNet2DModelSimpleConditional
from .masking import encode, masked_mse


class LidarDMAdapter(nn.Module):
    def __init__(self, config, sensor_meta):
        super().__init__()
        self.config, self.sensor_meta = config, sensor_meta
        window, channels = config['window'], config['channels']
        self.unet = UNet2DModelSimpleConditional(pretrained=None,
            sample_size=(len(sensor_meta['elevations']), sensor_meta['width']),
            in_channels=5 * window + 2, out_channels=3 * window,
            block_out_channels=(channels, channels * 2, channels * 4),
            down_block_types=('DownBlock2D', 'DownBlock2D', 'DownBlock2D'),
            up_block_types=('UpBlock2D', 'UpBlock2D', 'UpBlock2D'),
            layers_per_block=1, norm_num_groups=4, add_attention=False)
        self.scheduler = DDPMScheduler(num_train_timesteps=config['diffusion_steps'], beta_schedule='squaredcos_cap_v2',
                                       prediction_type='epsilon', clip_sample=True)
        elevation = torch.tensor(sensor_meta['elevations'], dtype=torch.float32)
        azimuth = torch.tensor(sensor_meta['azimuths'], dtype=torch.float32)
        geometry = torch.stack((elevation[:, None].expand(-1, len(azimuth)) / (math.pi / 2),
                                azimuth[None].expand(len(elevation), -1) / math.pi))
        self.register_buffer('sensor_angles', geometry[None])

    def conditioning(self, batch_size, geometry=None, known=None):
        shape = (batch_size, self.config['window'], *self.sensor_angles.shape[-2:])
        if geometry is None:
            geometry = torch.zeros(shape, device=self.sensor_angles.device)
            known = torch.zeros(shape, dtype=torch.bool, device=self.sensor_angles.device)
        normalized = 2 * geometry.clamp_min(0).log1p() / math.log1p(self.sensor_meta['max_range_m']) - 1
        anchors = torch.stack((torch.where(known, normalized, 0), known.float()), dim=2).flatten(1, 2)
        return torch.cat((anchors, self.sensor_angles.expand(batch_size, -1, -1, -1)), dim=1)

    def forward(self, batch, return_prediction=False):
        x0, mask = encode(batch, self.sensor_meta['max_range_m'])
        timesteps = torch.randint(0, self.config['diffusion_steps'], (len(x0),), device=x0.device)
        noise = torch.randn_like(x0)
        noisy = self.scheduler.add_noise(x0, noise, timesteps)
        known = batch['valid'].bool() & (torch.rand_like(batch['range']) < .5)
        # Null conditioning examples allow unconditional sampling without changing models.
        known &= (torch.rand((len(x0), 1, 1, 1), device=x0.device) > .2)
        cond = self.conditioning(len(x0), batch['range'], known)
        prediction = self.unet(noisy, cond, timesteps).sample
        diffusion = masked_mse(prediction, noise, mask)
        alpha = self.scheduler.alphas_cumprod.to(x0.device)[timesteps, None, None, None]
        clean = ((noisy - (1 - alpha).sqrt() * prediction) / alpha.sqrt()).clamp(-1, 1)
        clean = clean.reshape(len(x0), self.config['window'], 3, *clean.shape[-2:])
        target = x0.reshape_as(clean)
        valid = batch['valid'].bool()
        geometry = (F.relu((clean[:, :, 0] - target[:, :, 0]).abs() - .01) * known).sum() / known.sum().clamp_min(1)
        drop = F.binary_cross_entropy_with_logits(clean[:, :, 1] * 4, valid.float())
        if self.config['window'] > 1:
            shared = valid[:, 1:] & valid[:, :-1]
            temporal = masked_mse(clean[:, 1:, 0] - clean[:, :-1, 0],
                                  target[:, 1:, 0] - target[:, :-1, 0], shared)
        else:
            temporal = clean.sum() * 0
        loss = diffusion + self.config['geometry_weight'] * geometry + self.config['raydrop_weight'] * drop
        loss += self.config['temporal_weight'] * temporal
        result = {'loss': loss, 'diffusion': diffusion, 'geometry': geometry, 'raydrop': drop, 'temporal': temporal}
        if return_prediction:
            result['prediction'] = {
                'range': ((clean[:, :, 0] + 1) / 2 * math.log1p(self.sensor_meta['max_range_m'])).expm1(),
                'valid': (clean[:, :, 1] * 4).sigmoid(), 'intensity': (clean[:, :, 2] + 1) / 2}
        return result

    def train_step(self, real_batch):
        return self(real_batch)

    @torch.no_grad()
    def sample(self, batch_size, conditioning=None, seed=0, steps=None, guidance=1., force_cfg=False):
        device = self.sensor_angles.device
        generator = torch.Generator(device=device).manual_seed(seed)
        shape = (batch_size, 3 * self.config['window'], *self.sensor_angles.shape[-2:])
        latent = torch.randn(shape, device=device, generator=generator)
        geometry = None if conditioning is None else conditioning['range'].to(device)
        known = None if conditioning is None else conditioning['valid'].to(device).bool()
        if geometry is not None and tuple(geometry.shape) != (batch_size, self.config['window'], *shape[-2:]):
            raise ValueError('Conditioning range/valid must have shape [B,T,H,W] matching the sensor')
        if geometry is not None and (known.shape != geometry.shape or not torch.isfinite(geometry).all()
                                     or (geometry[known] < .1).any() or (geometry[known] > self.sensor_meta['max_range_m']).any()):
            raise ValueError('Conditioning needs matching masks and finite, valid ranges within sensor bounds')
        cond = self.conditioning(batch_size, geometry, known)
        null_cond = self.conditioning(batch_size) if guidance != 1. or force_cfg else None
        scheduler = DDIMScheduler.from_config(self.scheduler.config)
        scheduler.set_timesteps(steps or self.config['sample_steps'], device=device)
        for timestep in scheduler.timesteps:
            prediction = self.unet(latent, cond, timestep).sample
            if null_cond is not None:
                unconditioned = self.unet(latent, null_cond, timestep).sample
                prediction = unconditioned + guidance * (prediction - unconditioned)
            latent = scheduler.step(prediction, timestep, latent, generator=generator).prev_sample
        value = latent.clamp(-1, 1).reshape(batch_size, self.config['window'], 3, *shape[-2:])
        ranges = ((value[:, :, 0] + 1) / 2 * math.log1p(self.sensor_meta['max_range_m'])).expm1()
        valid = (value[:, :, 1] > 0) & (ranges >= .1)
        if conditioning is not None:
            tolerance = self.config['geometry_tolerance_m']
            ranges = torch.where(known, ranges.maximum(geometry - tolerance).minimum(geometry + tolerance), ranges)
            if conditioning.get('complete_scene', True):
                valid &= known
        return {'range': torch.where(valid, ranges, 0).float(), 'valid': valid,
                'intensity': torch.where(valid, (value[:, :, 2] + 1) / 2, 0).float()}
