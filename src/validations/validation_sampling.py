from dataclasses import dataclass
from math import ceil
import os
from typing import Literal

import torch
from torch import Tensor
from torch.utils.data import IterableDataset
from torchvision.utils import make_grid, save_image

from .validation import Validation, ValidationCfg
from ..model.diffusion_wrapper import DiffusionWrapper


@dataclass
class ValidationSamplingCfg(ValidationCfg):
    name: Literal["sampling"]
    num_samples: int = 1


class ValidationSampling(Validation[ValidationSamplingCfg]):
    def __init__(
        self,
        cfg: ValidationSamplingCfg,
        dataset: IterableDataset | None = None,
    ) -> None:
        super(ValidationSampling, self).__init__(cfg, dataset)
        self.num_zfill = len(str(cfg.num_samples))

    @torch.no_grad()
    def validate(self, m: DiffusionWrapper, batch) -> None:
        samples = m.sample(batch["data"] if self.super_resolution else batch["name"], self.resolution_factors)
        if self.save:
            out_path = os.path.join(m.output_path, "validation", str(m.global_step), self.tag)
            os.makedirs(out_path, exist_ok=True)
            for i, name in enumerate(batch["name"]):
                if isinstance(name, Tensor):
                    name = str(name.item())
                save_image(
                    [batch["data"][i], samples[i]] if self.super_resolution else samples[i], 
                    os.path.join(out_path, f"{name}.jpg"),
                    normalize=True,
                    value_range=(-1, 1)
                )
        if self.super_resolution:
            grid = make_grid(
                torch.stack(batch["data"], samples, dim=0).flatten(0, 1),
                nrow=len(samples),
                normalize=True,
                value_range=(-1, 1)
            )
        else:
            grid = make_grid(samples, nrow=ceil(len(samples) ** 0.5), normalize=True, value_range=(-1, 1))
        m.logger.experiment.add_image("/".join(["val", self.tag]), grid, m.global_step)

    def __iter__(self):
        for i in range(self.cfg.num_samples):
            yield dict(name=str(i).zfill(self.num_zfill))

    def __len__(self) -> int:
        return self.cfg.num_samples
