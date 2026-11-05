from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .ot.config import ExperimentConfig, OTConfig
from .methods.registry import CALIBRATION_METHODS, CalibrationMethod
from .rl.config import RLConfig


class RunConfig(BaseModel):
    action: Literal['prepare', 'train', 'generate', 'experiment'] = 'train'
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
    batch_size: int = Field(512, ge=1, le=4096)
    auto_optimize: bool = True
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
    workers: int = Field(-1, ge=-1, le=16)
    gpus: int = Field(-1, ge=-1)
    seed: int = Field(0, ge=0, le=2**31 - 1)
    rl_enabled: bool = True
    methods: list[CalibrationMethod] = Field(
        default_factory=lambda: ['ppo', 'random_search', 'bayesian'], min_length=1, max_length=len(CALIBRATION_METHODS))
    rl: RLConfig = Field(default_factory=RLConfig)
    budget: int = Field(30, ge=6, le=5000)
    conditioning: str = ''
    auto_conditioning: bool = True
    camera_overlay: bool = True
    camera_calibration: str = ''
    camera_model: Literal['rational', 'pinhole'] = 'rational'
    camera_timestamp_unit: Literal['s', 'ms', 'us', 'ns'] = 's'
    camera_max_skew_ms: float = Field(50., gt=0, le=1000)
    profile_file: str = ''
    profile_name: str = 'ppo'
    controller_file: str = ''
    ot: OTConfig = Field(default_factory=OTConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    warm_start: str = ''
    odometry_mode: Literal['disabled', 'world_from_vehicle', 'previous_from_current'] = 'disabled'

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
        if self.controller_file and (self.action != 'generate' or not (self.conditioning or self.auto_conditioning)):
            raise ValueError('A learned generation controller requires Generate with a conditioning NPZ or automatic real-scene extraction')
        if self.action == 'train' and self.checkpoint:
            raise ValueError('First-version training starts from random initialization; pretrained loading is excluded')
        if len(set(self.methods)) != len(self.methods):
            raise ValueError('Calibration methods must be unique')
        return self
