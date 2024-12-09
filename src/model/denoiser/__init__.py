
import torch

from .denoiser import Denoiser
from .mvunet import MultiViewUNetCfg, MultiViewUNet

DENOISER = {
    "mv_unet": MultiViewUNet
}

DenoiserCfg = MultiViewUNetCfg

def get_denoiser(
    denoiser_cfg: DenoiserCfg,
    in_channels: int, 
    out_channels: int,
) -> Denoiser:
    return DENOISER[denoiser_cfg.name](denoiser_cfg, in_channels, out_channels)


