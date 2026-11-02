"""Range-image adaptation of LidarDM/core/models/unet.py at 9d16b795b640.

Copyright (c) 2025 Vlas Zyrianov. MIT notice: LICENSES/MODEL-MIT.txt.
The upstream conditional concatenation is retained; pretrained loading is excluded.
"""
import torch
from diffusers import UNet2DModel


class UNet2DModelSimpleConditional(UNet2DModel):
    def __init__(self, pretrained=None, **kwargs):
        if pretrained is not None:
            raise ValueError('Upstream pretrained weights are excluded; use random initialization')
        super().__init__(**kwargs)

    def forward(self, x, cond, t):
        return super().forward(torch.cat((x, cond), dim=1), t)
