from dataclasses import dataclass
from functools import partial
from typing import Literal, Optional, Tuple

from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn
import torch
from ...transformer.transformer import Transformer
from ..backbone.unimatch.position import PositionEmbeddingSine
from ..backbone.unimatch.utils import split_feature, merge_splits

@dataclass
class CrossAttentionCfg:
    name: Literal["standard"]
    num_heads: int
    num_layers: int = 1
    d_dot: int | None = None                # default d_in // num_heads
    d_mlp: int | None = None                # if None: default d_in * d_mlp_multiplier
    d_mlp_multiplier: int | None = None
    downscale: int = 1
    pos_enc: bool=False
def feature_add_position_list(features, attn_splits, feature_channels):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)
    
    features_list = [features[:, i] for i in range(features.shape[1])]
    if attn_splits > 1:  # add position in splited window
        features_splits = [
            split_feature(x, num_splits=attn_splits, channel_last=False) for x in features_list
        ]

        position = pos_enc(features_splits[0])
        features_splits = [x + position for x in features_splits]

        out_features_list = [
            merge_splits(x, num_splits=attn_splits) for x in features_splits
        ]

    else:
        position = pos_enc(features_list[0])

        out_features_list = [x + position for x in features_list]

    return torch.stack(out_features_list, dim=1)
class StandardTransformer(nn.Module):
    cfg: CrossAttentionCfg
    depth_encoding: nn.Sequential
    transformer: Transformer
    downscaler: Optional[nn.Conv2d]
    upscaler: Optional[nn.ConvTranspose2d]
    upscale_refinement: Optional[nn.Sequential]

    def __init__(
        self,
        cfg: CrossAttentionCfg,
        d_in: int,
        d_kv: int | None=None
    ) -> None:
        super().__init__()
        assert (cfg.d_mlp is None) != (cfg.d_mlp_multiplier is None), \
            "Expected exactly one of d_mlp and d_mlp_multiplier"
        self.cfg = cfg
        self.cond_proj = True if d_kv is not None else False
        self.pos_enc = cfg.pos_enc
        self.num_heads = cfg.num_heads
        self.d_in = d_in
        self.d_kv = d_kv if d_kv is not None else d_in 

        self.transformer = Transformer(
            d_in,
            cfg.num_layers,
            cfg.num_heads,
            cfg.d_dot or d_in // cfg.num_heads,
            cfg.d_mlp or d_in * cfg.d_mlp_multiplier,
            selfatt=True,
            kv_dim=d_in
        )
        if d_kv is not None:
            self.proj = nn.Linear(d_kv, d_in)

        if cfg.downscale > 1:
            self.downscaler = nn.Conv2d(d_in, d_in, cfg.downscale, cfg.downscale)
            self.upscaler = nn.ConvTranspose2d(d_in, d_in, cfg.downscale, cfg.downscale)
            self.upscale_refinement = nn.Sequential(
                nn.Conv2d(d_in, d_in * 2, 7, 1, 3),
                nn.GELU(),
                nn.Conv2d(d_in * 2, d_in, 7, 1, 3),
            )
        else:
            self.downscaler = None
            self.upscaler = None
            self.upscale_refinement = None
    
    def forward(
        self,
        features: Float[Tensor, "batch view _ height width"],
        
    ) -> Float[Tensor, "batch view channel height width"]:
        b, v, _, h, w = features.shape
        if self.pos_enc:
            features = feature_add_position_list(features, 1, self.d_in)
            if cond_features is not None:
                cond_features = feature_add_position_list(cond_features, 1, self.d_kv)
        
        # If needed, apply downscaling.
        if self.downscaler is not None:
            features = rearrange(features, "b v c h w -> (b v) c h w")
            features = self.downscaler(features)
            features = rearrange(features, "(b v) c h w -> b v c h w", b=b, v=v)

        
        

        # Run the transformer.
        qkv = features
        
        qkv = rearrange(qkv, "b v c h w -> b (v h w) c")
        features = self.transformer.forward(
            qkv,
            None
        )
        features = rearrange(
            features,
            "b (v h w) c -> b v c h w",
            v=v,
            h=h // self.cfg.downscale,
            w=w // self.cfg.downscale,
        )

        # If needed, apply upscaling.
        if self.upscaler is not None:
            features = rearrange(features, "b v c h w -> (b v) c h w")
            features = self.upscaler(features)
            features = self.upscale_refinement(features) + features
            features = rearrange(features, "(b v) c h w -> b v c h w", b=b, v=v)
        return features
