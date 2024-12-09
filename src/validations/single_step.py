from math import ceil
import os
from typing import Dict, Union

import torch
from torch import Tensor
from torchvision.utils import make_grid, save_image

from .validation import Validation
from core.model import Model
from datasets import BaseDataset


class SingleStep(Validation):
    def __init__(
        self,
        tag: str,
        dataset: BaseDataset,
        n_sub: int,
        t: int,
        super_resolution: bool = False,
        save: bool = True
    ) -> None:
        super(SingleStep, self).__init__(tag, dataset, super_resolution)
        self.dataset = dataset
        self.n_sub = min(n_sub, len(dataset))
        self.t = t
        self.save = save

    @torch.no_grad()
    def validate(self, m: Model, batch: Dict[str, Union[str, Tensor]]) -> None:
        reconstructed = m.single_step(batch["data"], self.t)
        if self.save:
            out_path = os.path.join(m.output_path, "validation", str(m.global_step), self.tag)
            os.makedirs(out_path, exist_ok=True)
            for i, name in enumerate(batch["name"]):
                if isinstance(name, Tensor):
                    name = str(name.item())
                save_image(
                    reconstructed[i], 
                    os.path.join(out_path, f"{name}.jpg"),
                    normalize=True,
                    value_range=(-1, 1)
                )
        grid = make_grid(
            reconstructed, 
            nrow=ceil(len(reconstructed) ** 0.5),
            normalize=True,
            value_range=(-1, 1)
        )
        m.logger.experiment.add_image("/".join(["val", self.tag]), grid, m.global_step)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, Tensor]]:
        return dict(name=str(idx), data=self.dataset[idx])

    def __len__(self):
        return self.n_sub