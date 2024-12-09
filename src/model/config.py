from torch import Tensor

from jaxtyping import Float
from pathlib import Path
from dataclasses import dataclass, fields
from .autoencoder import AutoencoderCfg
from .denoiser import DenoiserCfg
from .scheduler import SchedulerCfg
from typing import Any, Dict, Iterator, Literal, Optional, Protocol, runtime_checkable


@dataclass
class RayEncodingsCfg:
    num_origin_octaves: int=10
    num_direction_octaves: int=8


@dataclass
class ModelCfg:
    denoiser: DenoiserCfg
    scheduler: SchedulerCfg
    autoencoder: AutoencoderCfg
    ray_encodings: RayEncodingsCfg
    use_cfg: bool=False
    cfg_scale: float=3.0
    cfg_train: bool=True
    use_ray_encoding: bool=True
    srt_ray_encoding: bool=False
    use_ddim_scheduler: bool=False
    use_plucker: bool=False
    ema: bool=False
    use_ema_sampling: bool=False
    enable_xformers_memory_efficient_attention: bool=False

@dataclass
class LRSchedulerCfg:
    name: str
    frequency: int = 1
    interval: Literal["epoch", "step"] = "step"
    kwargs: Dict[str, Any] | None = None


@dataclass
class FreezeCfg:
    denoiser: bool = False
    autoencoder: bool = True
    
@dataclass
class OptimizerCfg:
    name: str
    lr: float
    scale_lr: bool
    kwargs: Dict[str, Any] | None = None
    scheduler: LRSchedulerCfg | None = None
  

@dataclass
class TestCfg:
    output_dir: Path
    limit_frames: int | None
    sampling_mode: Literal["anchored", "autoregressive"] | None
    num_anchors_views: int=4

@dataclass
class TrainCfg:
    step_offset: int
    cfg_train: bool=True