import numpy as np
from dataclasses import dataclass, asdict
from typing import Literal, Union
from diffusers import DDIMScheduler, DDPMScheduler

from .ddim import DDIMSchedulerCfg
from .ddpm import DDPMSchedulerCfg

_SchedulerCfg = DDIMSchedulerCfg | DDPMSchedulerCfg

@dataclass
class SchedulerCfg:
    name: Literal["ddim", "ddpm"]
    num_train_timesteps: int
    num_inference_steps: int
    pretrained_from: str | None
    kwargs: _SchedulerCfg

SCHEDULER = {
    "ddim": DDIMScheduler,
    "ddpm": DDPMScheduler,
}

def _parse_cfg(cfg: _SchedulerCfg):

    # Original type and types returned by hydra may conflict
    if type(cfg.trained_betas) == list:
        cfg.trained_betas = np.array(cfg.trained_betas)
    
    return cfg

def get_scheduler(
    cfg: SchedulerCfg
) -> Union[DDPMScheduler, DDIMScheduler]:
     
    if cfg.pretrained_from is None:
        return SCHEDULER[cfg.name](**asdict(_parse_cfg(cfg.kwargs)))

    else:
        return SCHEDULER[cfg.name].from_pretrained(cfg.pretrained_from, subfolder="scheduler")
