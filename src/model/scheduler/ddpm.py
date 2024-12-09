

import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional, Union, List


# Refer to https://github.com/huggingface/diffusers/blob/89e4d6219805975bd7d253a267e1951badc9f1c0/src/diffusers/schedulers/scheduling_ddpm.py#L129
@dataclass
class DDPMSchedulerCfg:
    num_train_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"
    trained_betas: Optional[Union[np.ndarray, List[float]]] = None
    variance_type: str = "fixed_small"
    clip_sample: bool = True
    prediction_type: str = "epsilon"
    thresholding: bool = False
    dynamic_thresholding_ratio: float = 0.995
    clip_sample_range: float = 1.0
    sample_max_value: float = 1.0
    timestep_spacing: str = "leading"
    steps_offset: int = 0
    rescale_betas_zero_snr: bool = False