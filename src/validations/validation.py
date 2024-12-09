from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from torch.utils.data import IterableDataset

from ..model.diffusion_wrapper import DiffusionWrapper


@dataclass
class ValidationCfg:
    tag: str


T = TypeVar("T", bound=ValidationCfg)


class Validation(ABC, Generic[T], IterableDataset):
    cfg: T

    def __init__(
        self,
        cfg: T,
        dataset: IterableDataset | None = None,
    ) -> None:
        super(Validation, self).__init__()
        self.cfg = cfg
        self.dataset = dataset

    @abstractmethod
    def validate(self, m: DiffusionWrapper, data: Any) -> None:
        raise NotImplementedError()

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError()

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()