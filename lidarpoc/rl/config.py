"""Settings shared by the newly adapted RL methods."""
from pydantic import BaseModel, Field, model_validator


class RLConfig(BaseModel):
    warmup: int = Field(4, ge=0, le=1000)
    batch_size: int = Field(16, ge=2, le=256)
    replay_size: int = Field(10000, ge=2, le=100000)
    rollout_steps: int = Field(5, ge=1, le=256)
    n_steps: int = Field(3, ge=1, le=64)
    value_min: float = Field(-20., ge=-10000, le=10000, allow_inf_nan=False)
    value_max: float = Field(20., ge=-10000, le=10000, allow_inf_nan=False)

    @model_validator(mode='after')
    def support(self):
        if self.value_min >= self.value_max:
            raise ValueError('Categorical return support requires value_min < value_max')
        return self
