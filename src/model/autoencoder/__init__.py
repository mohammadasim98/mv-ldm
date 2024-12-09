from dataclasses import dataclass
from typing import Union, Literal, List, Tuple
from diffusers import AutoencoderKL

from .autoencoder_kl import AutoencoderKLCfg

_AutoencoderCfg = AutoencoderKLCfg

@dataclass
class AutoencoderCfg:
    name: Literal["kl"]
    pretrained_from: str | None
    kwargs: _AutoencoderCfg

AUTOENCODERS = {
    "kl": AutoencoderKL
}

def _parse_cfg(cfg: _AutoencoderCfg):

    # Original type and types returned by hydra may conflict
    if type(cfg.down_block_types) == list:
        cfg.down_block_types = tuple(cfg.down_block_types)
    if type(cfg.up_block_types) == list:
        cfg.up_block_types = tuple(cfg.up_block_types)
    if type(cfg.block_out_channels) == list:
        cfg.block_out_channels = tuple(cfg.block_out_channels)
    if latents_mean is not None and type(cfg.latents_mean) == list:
        cfg.latents_mean = tuple(cfg.latents_mean)
    if latents_std is not None and type(cfg.latents_std) == list:
        cfg.latents_std = tuple(cfg.latents_std)
    
    return cfg

def get_autoencoder(
    cfg: AutoencoderCfg
) -> Union[AutoencoderKL]:
    
    if cfg.pretrained_from is None:
        return AUTOENCODERS[cfg.name](**asdict(_parse_cfg(cfg.kwargs)))

    else:
        return AUTOENCODERS[cfg.name].from_pretrained(cfg.pretrained_from, subfolder="vae")
