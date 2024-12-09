from dataclasses import dataclass
from typing import Literal

from jaxtyping import Float
from torch import Tensor

from ..model.types import Prediction, GroundTruth
from .loss import LossCfg, Loss


@dataclass
class LossIsotropyCfg(LossCfg):
    name: Literal["isotropy"] = "isotropy"


class LossIsotropy(Loss):
    def unweighted_loss(
        self,
        prediction: Prediction,
        gt: GroundTruth | None = None
    ) -> Float[Tensor, ""]:
        max_scale = prediction.gaussian_scales.max(dim=-1).values
        min_scale = prediction.gaussian_scales.min(dim=-1).values
        # NOTE max_scale is > 0 as defined in gaussian adapter
        anisotropy = (max_scale - min_scale) / max_scale
        return anisotropy.mean()
