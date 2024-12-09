from torch.utils.data import Dataset
from torch import Generator

from ..misc.step_tracker import StepTracker
from .dataset_re10k import DatasetRE10k, DatasetRE10kCfg
from .dataset_re10kv2 import DatasetRE10kV2, DatasetRE10kV2Cfg
from .types import Stage
from .view_sampler import get_view_sampler

DATASETS: dict[str, Dataset] = {
    "re10k": DatasetRE10k,
    "re10k_non_iter": DatasetRE10kV2,
}


DatasetCfg = DatasetRE10kCfg | DatasetRE10kV2Cfg


def get_dataset(
    cfg: DatasetCfg,
    stage: Stage,
    step_tracker: StepTracker | None,
    generator: Generator | None = None,
    force_shuffle: bool = False
) -> Dataset:

    view_sampler = get_view_sampler(
        cfg.view_sampler,
        stage,
        cfg.overfit_to_scene is not None,
        cfg.cameras_are_circular,
        step_tracker,
        generator
    )
    return DATASETS[cfg.name](cfg, stage, view_sampler, force_shuffle)
