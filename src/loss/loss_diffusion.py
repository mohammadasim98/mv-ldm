from dataclasses import dataclass
from typing import Literal

from jaxtyping import Float
import torch
from torch import Tensor

from .loss import Loss, LossCfg, LossOutput, LossValue


@dataclass
class LossDiffusionCfg(LossCfg):
    name: Literal["mse"] = "mse"
    weights: list | None = None

class LossDiffusion(Loss):

        
    def unweighted_loss(
        self,
        prediction: GaussianMaps,
        gt: GaussianMaps
    ) -> Float[Tensor, ""]:
        weights = self.cfg.weights
        if weights is None:
            weights = [1, 1, 1, 1, 1, 1, 1]

        delta = 0
        delta += weights[0] * ((prediction.offset_xy - gt.offset_xy) ** 2).mean()
        delta += weights[1] * ((prediction.relative_disparity - gt.relative_disparity) ** 2).mean()
        delta += weights[2] * ((prediction.scales - gt.scales) ** 2).mean()
        delta += weights[3] * ((prediction.rotations - gt.rotations) ** 2).mean()
        delta += weights[4] * ((prediction.color_harmonics - gt.color_harmonics) ** 2).mean()
        delta += weights[5] * ((prediction.feature_harmonics - gt.feature_harmonics) ** 2).mean()
        delta += weights[6] * ((prediction.opacities - gt.opacities) ** 2).mean()
        
        return delta / sum(weights)
        
    def forward(
        self,
        prediction: GaussianMaps,
        gt: GaussianMaps | None = None,
        global_step: int = 0
    ) -> LossOutput:
        # Before the specified step, don't apply the loss.
        if global_step < self.cfg.apply_after_step:
            unweighted = torch.tensor(0, dtype=torch.float32, device=prediction.device)
        else:
            unweighted = self.unweighted_loss(prediction, gt)
        weighted = self.cfg.weight * unweighted
        return LossValue(unweighted, weighted)
