import torch
import os 
from torch import Tensor, nn
from dataclasses import dataclass
from jaxtyping import Float, Int64
from typing import Literal, Optional, Tuple
from einops import rearrange, repeat
from diffusers import UNetSpatioTemporalConditionModel

from .attention import get_attn_blocks
from .denoiser import Denoiser


from .attention import MultiViewAttentionCfg
DISABLE_TORCH_COMPILE = int(os.getenv('DISABLE_TORCH_COMPILE', True))
if DISABLE_TORCH_COMPILE == 0:
    DISABLE_TORCH_COMPILE = False
else:
    DISABLE_TORCH_COMPILE = True
print("DISABLE_TORCH_COMPILE: ", DISABLE_TORCH_COMPILE)
@dataclass
class UNet2DModelCfg:
    name: Literal["unet"]
    down_block_types: list | Tuple
    mid_block_type: str
    up_block_types: list | Tuple
    only_cross_attention: bool
    block_out_channels: list | Tuple

@dataclass
class SVDUNetCfg:
    name: Literal["svd_unet"]
    autoencoder: UNet2DModelCfg

    pretrained_from: str | None = None

class SVDUNet(Denoiser[SVDUNetCfg]):
    def __init__(
        self, 
        cfg: SVDUNetCfg,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__(cfg)
        self.pretrained_from = cfg.pretrained_from


        self.unet = UNetSpatioTemporalConditionModel.from_pretrained(self.pretrained_from, subfolder="unet")

        self.unet.conv_in = nn.Conv2d(
            in_channels, cfg.autoencoder.block_out_channels[0], kernel_size=3, padding=1
        )


    def forward(
        self,
        latents:  Float[Tensor, "batch view _ height width"],
        timestep: Int64[Tensor, "batch ..."],
        cond_state: Optional[Tensor]=None
    ) -> Float[Tensor, "batch view _ height width"]:
        
        b, num_views, c, h, w = latents.shape
        hidden_states = latents
        
        
        # # 1. process timesteps
        # if len(timestep.shape) < 2:
        #     timestep = repeat(timestep, "b ... -> (b v) ...", v=num_views)
        # else:
        #     timestep = rearrange(timestep, "b v ... -> (b v) ...")
        
        # latents = rearrange(latents, "b v c h w -> b c v h w")
        added_time_ids = torch.zeros((b, 3)).to(latents.device)
        encoder_hidden_states = torch.zeros(b, 1, 1024).to(latents.device)
        # print(latents.shape)
        sample = self.unet(latents, timestep[:, 0], encoder_hidden_states=encoder_hidden_states, added_time_ids=added_time_ids)[0]
        # sample = rearrange(sample, "b c v h w -> b v c h w", v=num_views)
        
        return sample
