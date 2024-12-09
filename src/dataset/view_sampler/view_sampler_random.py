from dataclasses import dataclass
from typing import Literal

import torch

from .view_sampler import ViewIndex, ViewSampler


@dataclass
class ViewSamplerRandomCfg:
    name: Literal["random"]
    num_context_views: int
    num_target_views: int = 0


class ViewSamplerRandom(ViewSampler[ViewSamplerRandomCfg]):
    def sample(
        self,
        scene: str,
        num_views: int,
        device: torch.device = torch.device("cpu"),
    ) -> list[ViewIndex]:
        index_context = torch.randperm(num_views, generator=self.generator, device=device)[:self.cfg.num_context_views]
        index_target = torch.randperm(num_views, generator=self.generator, device=device)[:self.cfg.num_target_views] \
            if self.cfg.num_target_views > 0 else None
        return [ViewIndex(index_context, index_target)]

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views
