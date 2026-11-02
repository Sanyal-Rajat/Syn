from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class RunConfig(BaseModel):
    action: Literal['prepare', 'train', 'generate'] = 'train'
    dataset_root: str = ''
    manifest: str = ''
    checkpoint: str = ''
    calibration_file: str = str(Path(__file__).resolve().parents[1] / 'LICENSES/Pandar128-Configuration.json')
    input_frame: Literal['sensor', 'vehicle'] = 'vehicle'
    elevations_file: str = ''
    infer_beams: bool = True
    height: int = Field(128, ge=4, le=128)
    width: int = Field(512, ge=8, le=2048)
    max_range_m: float = Field(120., gt=1, le=200)
    intensity_max: float = Field(255., gt=0)
    beam_tolerance_deg: float = Field(.15, gt=0, le=1)
    max_frames: int = Field(128, ge=2, le=100000)
    window: int = Field(2, ge=1, le=5)
    channels: int = Field(32, ge=8, le=128)
    batch_size: int = Field(2, ge=1, le=64)
    epochs: int = Field(5, ge=1, le=10000)
    lr: float = Field(1e-4, gt=0, le=.01)
    diffusion_steps: int = Field(1000, ge=10, le=2000)
    sample_steps: int = Field(50, ge=2, le=200)
    samples_per_trace: int = Field(4, ge=1, le=100)
    geometry_weight: float = Field(.1, ge=0, le=10)
    raydrop_weight: float = Field(.1, ge=0, le=10)
    temporal_weight: float = Field(.1, ge=0, le=10)
    geometry_tolerance_m: float = Field(.1, gt=0, le=1)
    gradient_accumulation: int = Field(1, ge=1, le=64)
    workers: int = Field(0, ge=0, le=16)
    gpus: int = Field(1, ge=0, le=16)
    seed: int = Field(0, ge=0, le=2**31 - 1)
    rl_enabled: bool = True
    methods: list[Literal['ppo', 'random_search', 'bayesian', 'cma_es']] = Field(
        default_factory=lambda: ['ppo', 'random_search', 'bayesian'], min_length=1, max_length=4)
    budget: int = Field(30, ge=6, le=5000)
    conditioning: str = ''
    profile_file: str = ''
    profile_name: str = 'ppo'

    @model_validator(mode='after')
    def validate_inputs(self):
        if self.height % 4 or self.width % 4 or self.channels % 4:
            raise ValueError('Height, width and model channels must be divisible by four')
        if self.sample_steps > self.diffusion_steps:
            raise ValueError('Sampling steps cannot exceed diffusion training steps')
        if self.action == 'prepare' and not self.infer_beams and not self.elevations_file:
            raise ValueError('Provide measured beam elevations or enable training-only beam estimation')
        if self.action != 'prepare' and not self.manifest:
            raise ValueError('Provide the prepared split manifest path on the server')
        if self.action == 'generate' and not self.checkpoint:
            raise ValueError('Generation requires a checkpoint produced by this application')
        if self.action == 'train' and self.checkpoint:
            raise ValueError('First-version training starts from random initialization; pretrained loading is excluded')
        if len(set(self.methods)) != len(self.methods):
            raise ValueError('Calibration methods must be unique')
        return self
