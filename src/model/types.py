from dataclasses import dataclass, fields

from jaxtyping import Float
from torch import Tensor




@dataclass
class Prediction:
    image: Float[Tensor, "batch view channels height width"] | None = None
    depth: Float[Tensor, "batch view height width"] | None = None


    # NOTE assumes all fields to be on the same device
    @property
    def device(self):
        for field in fields(self):
            val = getattr(self, field.name)
            if val is not None:
                return val.device


@dataclass
class GroundTruth:
    image: Float[Tensor, "batch view channels height width"] | None = None
    near: Float[Tensor, "batch view"] | None = None
    far: Float[Tensor, "batch view"] | None = None


