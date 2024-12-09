import torch
import os 
from torch import Tensor, nn
from dataclasses import dataclass
from jaxtyping import Float, Int64
from typing import Literal, Optional, Tuple
from einops import rearrange, repeat
from diffusers import UNet2DConditionModel

from .attention import get_attn_blocks
from .denoiser import Denoiser


from .attention import MultiViewAttentionCfg

DISABLE_TORCH_COMPILE = int(os.getenv('DISABLE_TORCH_COMPILE', True))
if DISABLE_TORCH_COMPILE == 0:
    DISABLE_TORCH_COMPILE = False
else:
    DISABLE_TORCH_COMPILE = True

@dataclass
class UNet2DModelCfg:
    name: Literal["unet"]
    down_block_types: list | Tuple
    mid_block_type: str
    up_block_types: list | Tuple
    only_cross_attention: bool
    block_out_channels: list | Tuple

@dataclass
class MultiViewUNetCfg:
    name: Literal["mv_unet"]
    autoencoder: UNet2DModelCfg
    multi_view_attention: MultiViewAttentionCfg
    use_ray_encoding: bool=True
    encoder_conditioning: bool=True  
    mid_conditioning: bool=True  
    decoder_conditioning: bool=True  
    pretrained_from: str | None = None

class MultiViewUNet(Denoiser[MultiViewUNetCfg]):
    def __init__(
        self, 
        cfg: MultiViewUNetCfg,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__(cfg)
        self.use_ray_encoding = cfg.use_ray_encoding
        self.pretrained_from = cfg.pretrained_from

        if self.pretrained_from is None:
            self.unet = UNet2DConditionModel(
                in_channels=in_channels, 
                out_channels=out_channels,
                down_block_types=cfg.autoencoder.down_block_types,
                mid_block_type=cfg.autoencoder.mid_block_type,
                up_block_types=cfg.autoencoder.up_block_types,
                only_cross_attention=cfg.autoencoder.only_cross_attention,
                block_out_channels=cfg.autoencoder.block_out_channels,
                cross_attention_dim=cfg.autoencoder.block_out_channels
            )
        else:
            print("Loading from Pretrained: ", self.pretrained_from)
            self.unet = UNet2DConditionModel.from_pretrained(self.pretrained_from, subfolder="unet")
            self.unet.conv_in = nn.Conv2d(
                in_channels, cfg.autoencoder.block_out_channels[0], kernel_size=3, padding=1
            )
            self.unet.conv_out = nn.Conv2d(
                cfg.autoencoder.block_out_channels[0], out_channels, kernel_size=3, padding=1
            )

        if self.cfg.encoder_conditioning:
            self.cross_attn_blocks_encoder = get_attn_blocks(
                cfg=cfg.multi_view_attention,
                unet_blocks=self.unet.down_blocks
            )
        if self.cfg.mid_conditioning:
            self.cross_attn_blocks_mid = get_attn_blocks(
                cfg=cfg.multi_view_attention,
                unet_blocks=[self.unet.mid_block]
            )
        if self.cfg.decoder_conditioning:
            self.cross_attn_blocks_decoder = get_attn_blocks(
                cfg=cfg.multi_view_attention,
                unet_blocks=self.unet.up_blocks
            ) 
            
    def forward(
        self,
        latents:  Float[Tensor, "batch view _ height width"],
        timestep: Int64[Tensor, "batch ..."],
        cond_state: Optional[Tensor]=None
    ) -> Float[Tensor, "batch view _ height width"]:
        
        b, num_views, c, h, w = latents.shape
        hidden_states = latents
        
        
        # 1. process timesteps
        if len(timestep.shape) < 2:
            timestep = repeat(timestep, "b ... -> (b v) ...", v=num_views)
        else:
            timestep = rearrange(timestep, "b v ... -> (b v) ...")
        
        t_emb = self.unet.time_proj(timestep)
        emb = self.unet.time_embedding(t_emb)
        
        
            
        hidden_states = rearrange(hidden_states, "b v ... -> (b v) ...")
        hidden_states = self.unet.conv_in(hidden_states)

        # 2. unet
        # a. downsample
        down_block_res_samples = (hidden_states,)

        for l, downsample_block in enumerate(self.unet.down_blocks):
            for i, resnet in enumerate(downsample_block.resnets):
                hidden_states = resnet(hidden_states, emb)
                if hasattr(downsample_block, 'has_cross_attention') and downsample_block.has_cross_attention:
                    cond = cond_state
                    if cond_state is None:
                        cond = repeat(torch.zeros_like(hidden_states).cuda(), "... c h w -> ... (h w) c")
                    
                    if self.pretrained_from is not None:
                        cond = torch.zeros(b*num_views, 1, 1024).cuda()

                    
                    hidden_states = downsample_block.attentions[i](
                        hidden_states, 
                        encoder_hidden_states=cond
                    ).sample
                down_block_res_samples += (hidden_states,)
            *_, h, w = hidden_states.shape
            if (h <= 32 and w <= 32):
                hidden_states = rearrange(hidden_states, "(b v) ... -> b v ...", v=num_views)

                if self.cfg.encoder_conditioning and (h <= 32 and w <= 32):
                    hidden_states = self.cross_attn_blocks_encoder[l](hidden_states)
                    
                hidden_states = rearrange(hidden_states, "b v ... -> (b v) ...")

            if downsample_block.downsamplers is not None:
                for downsample in downsample_block.downsamplers:
                    hidden_states = downsample(hidden_states)
                down_block_res_samples += (hidden_states,)
        # b. mid
        hidden_states = self.unet.mid_block.resnets[0](hidden_states, emb)

        for attn, resnet in zip(self.unet.mid_block.attentions, self.unet.mid_block.resnets[1:]):
            cond = cond_state
            if cond_state is None:
                cond = repeat(torch.zeros_like(hidden_states).cuda(), "... c h w -> ... (h w) c")
            if self.pretrained_from is not None:
                cond = torch.zeros(b*num_views, 1, 1024).cuda()
            hidden_states = attn(hidden_states, encoder_hidden_states=cond).sample
            hidden_states = resnet(hidden_states, emb)
        

        hidden_states = rearrange(hidden_states, "(b v) ... -> b v ...", v=num_views)

        if self.cfg.mid_conditioning:
            hidden_states = self.cross_attn_blocks_mid[0](hidden_states)
        hidden_states = rearrange(hidden_states, "b v ... -> (b v) ...")
            
        # c. upsample
        for l, upsample_block in enumerate(self.unet.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]

            for i, resnet in enumerate(upsample_block.resnets):
                res_hidden_states = res_samples[-1]
                res_samples = res_samples[:-1]
                hidden_states = torch.cat((hidden_states, res_hidden_states), dim=1)
                hidden_states = resnet(hidden_states, emb)
                if hasattr(upsample_block, 'has_cross_attention') and upsample_block.has_cross_attention and self.pretrained_from is None:
                    cond = cond_state
                    if cond_state is None:
                        cond = torch.zeros_like(hidden_states).cuda()
                        cond = repeat(cond, "... c h w -> ... (h w) c")
                    if self.pretrained_from is not None:
                        cond = torch.zeros(b*num_views, 1, 1024).cuda()
                    hidden_states = upsample_block.attentions[i](
                        hidden_states, 
                        encoder_hidden_states=cond
                    ).sample
            *_, h, w = hidden_states.shape
            if (h <= 32 and w <= 32):
                hidden_states = rearrange(hidden_states, "(b v) ... -> b v ...", v=num_views)

                if self.cfg.decoder_conditioning:
                    
                    hidden_states = self.cross_attn_blocks_decoder[l](hidden_states)
                hidden_states = rearrange(hidden_states, "b v ... -> (b v) ...")
            
            if upsample_block.upsamplers is not None:
                for upsample in upsample_block.upsamplers:
                    hidden_states = upsample(hidden_states)

        # 3.post-process
        sample = self.unet.conv_norm_out(hidden_states)
        sample = self.unet.conv_act(sample)
        sample = self.unet.conv_out(sample)
        sample = rearrange(sample, '(b v) ... -> b v ...', v=num_views)
        
        return sample
