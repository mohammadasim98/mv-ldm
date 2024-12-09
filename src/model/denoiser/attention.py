import torch.nn as nn

from .standard.transformer import StandardTransformer, CrossAttentionCfg
from .mvdream.attention import SpatialTransformer3D, SpatialTransformer3DCfg

MultiViewAttentionCfg = CrossAttentionCfg | SpatialTransformer3DCfg

def get_attn_blocks(
    cfg: MultiViewAttentionCfg,
    unet_blocks
    ):
    if cfg.name=="standard":
        return nn.ModuleList([
            StandardTransformer(
                cfg=cfg,
                d_in=block.resnets[-1].out_channels
            )
            for block in unet_blocks
        ])
    elif cfg.name=="spatial_transformer_3d":
        return nn.ModuleList([
            SpatialTransformer3D(
                cfg=cfg,
                d_in=block.resnets[-1].out_channels
            )
            for block in unet_blocks
        ])
