from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar
import torch
from jaxtyping import Float, Int64
from torch import nn, Tensor



T = TypeVar("T")


class Denoiser(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(
        self, 
        cfg: T
    ) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        latents: Float[Tensor, "batch view _ height width"],
        timestep: Int64[Tensor, "batch"],
        cond_state: Optional[Tensor]=None
    ) -> Float[Tensor, "batch view _ height width"]:
        pass
