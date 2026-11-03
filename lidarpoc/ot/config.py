from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ..methods.registry import ControllerMethod


class OTConfig(BaseModel):
    conditional_weight: float = Field(0., ge=0, le=100)
    mass_weight: float = Field(0., ge=0, le=100)
    physics_weight: float = Field(0., ge=0, le=100)
    regularization_weight: float = Field(0., ge=0, le=100)
    epsilon: float = Field(.1, gt=0, le=10)
    tau_source: float = Field(1., gt=0, le=100)
    tau_target: float = Field(1., gt=0, le=100)
    iterations: int = Field(60, ge=5, le=500)
    range_edges_m: list[float] = Field(default_factory=lambda: [0., 30., 60., 120.], min_length=2, max_length=16)
    feature_scales: list[float] = Field(default_factory=lambda: [120., 1., 1.5707963267948966, 1., 1.], min_length=5, max_length=5)
    feature_weights: list[float] = Field(default_factory=lambda: [1., 1., .25, .5, .5], min_length=5, max_length=5)
    ring_groups: int = Field(4, ge=1, le=16)
    azimuth_sectors: int = Field(4, ge=1, le=16)
    max_points: int = Field(64, ge=4, le=512)
    min_samples: int = Field(4, ge=2, le=64)
    reference: str = ''

    @model_validator(mode='after')
    def physical_units(self):
        import math
        if any(not math.isfinite(value) or value <= 0 for value in self.feature_scales):
            raise ValueError('OT feature scales must be finite, positive physical calibration units')
        if any(not math.isfinite(value) or value < 0 for value in self.feature_weights) or not any(self.feature_weights):
            raise ValueError('OT feature weights must be finite, nonnegative, and not all zero')
        if (any(not math.isfinite(value) or value < 0 for value in self.range_edges_m)
                or any(a >= b for a, b in zip(self.range_edges_m, self.range_edges_m[1:]))):
            raise ValueError('OT range strata must have finite, increasing nonnegative edges')
        if self.min_samples > self.max_points:
            raise ValueError('OT minimum bin support exceeds its point budget')
        return self


class ExperimentConfig(BaseModel):
    conditional_weight: float = Field(1., gt=0, le=100)
    mass_weight: float = Field(1., gt=0, le=100)
    physics_weight: float = Field(1., gt=0, le=100)
    regularization_weight: float = Field(.001, ge=0, le=100)
    ablations: list[Literal['E0', 'E1', 'E2', 'E3', 'E4', 'E5']] = Field(
        default_factory=lambda: ['E0', 'E1', 'E2', 'E3', 'E4', 'E5'], min_length=1)
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2], min_length=1, max_length=20)
    controllers: list[ControllerMethod] = Field(
        default_factory=lambda: ['fixed', 'random_search', 'bayesian', 'cma_es', 'sac'], min_length=1)
    scene_files: list[str] = Field(default_factory=list)
    development_scene_files: list[str] = Field(default_factory=list)
    test_scene_files: list[str] = Field(default_factory=list)
    synthetic_scenes: int = Field(4, ge=1, le=100)
    controller_max_samples: int = Field(2, ge=1, le=16)
    controller_seconds: float = Field(300., gt=0, le=86400)
    physics_penalty: float = Field(1., ge=0)
    compute_penalty: float = Field(.001, ge=0)
    sac_warmup: int = Field(4, ge=1, le=1000)
    sac_batch_size: int = Field(16, ge=2, le=256)
    downstream: bool = False
    label_source_confirmed: bool = False
    label_budgets: list[int] = Field(default_factory=lambda: [512, 2048], min_length=1, max_length=10)
    downstream_steps: int = Field(100, ge=1, le=10000)
    task_min_improvement: float = Field(0., ge=0, le=1)
    rl_min_improvement: float = Field(.01, ge=0, le=1)

    @model_validator(mode='after')
    def experiment_design(self):
        for values in (self.ablations, self.seeds, self.controllers, self.label_budgets):
            if len(values) != len(set(values)):
                raise ValueError('Experiment arms, seeds, controllers and label budgets must be unique')
        if any(value < 0 or value > 2**31 - 1 for value in self.seeds) or any(value < 1 for value in self.label_budgets):
            raise ValueError('Experiment seeds and label budgets are out of range')
        if self.downstream and not self.label_source_confirmed:
            raise ValueError('Confirm the semantic label source before enabling downstream task evaluation')
        if 'E5' in self.ablations and 'E4' not in self.ablations:
            raise ValueError('E5 freezes an E4 model; include E4 in the experiment')
        if any((self.scene_files, self.development_scene_files, self.test_scene_files)) and not all(
                (self.scene_files, self.development_scene_files, self.test_scene_files)):
            raise ValueError('External simulator scenes require separate training, development and test file lists')
        return self
