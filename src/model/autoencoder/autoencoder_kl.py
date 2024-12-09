from typing import Literal, Optional, Tuple, List
from dataclasses import dataclass

# Refer to https://github.com/huggingface/diffusers/blob/6131a93b969f87d171148bd367fd9990d5a49b6b/src/diffusers/models/autoencoders/autoencoder_kl.py#L38
@dataclass
class AutoencoderKLCfg:

    in_channels: int = 3
    out_channels: int = 3
    down_block_types: Tuple[str] | List[str] = ("DownEncoderBlock2D", )
    up_block_types: Tuple[str] | List[str] = ("UpDecoderBlock2D", )
    block_out_channels: Tuple[int] | List[int] = (64, )
    layers_per_block: int = 1
    act_fn: str = "silu"
    latent_channels: int = 4
    norm_num_groups: int = 32
    sample_size: int = 32
    scaling_factor: float = 0.18215
    shift_factor: Optional[float] = None
    latents_mean: Optional[Tuple[float] | List[float]] = None
    latents_std: Optional[Tuple[float] | List[float]] = None
    force_upcast: float | bool = True
    use_quant_conv: bool = True
    use_post_quant_conv: bool = True
    mid_block_add_attention: bool = True