from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Type, TypeVar

from dacite import Config, from_dict
from omegaconf import DictConfig, OmegaConf

from .dataset.data_module import DataLoaderCfg, DatasetCfg
from .model.config import FreezeCfg, OptimizerCfg, TestCfg, TrainCfg, ModelCfg

@dataclass
class CheckpointingCfg:
    load: Optional[str]  # Not a path, since it could be something like wandb://...
    every_n_train_steps: int
    save_top_k: int
    resume: bool = False
    save: bool = True


@dataclass
class TrainerCfg:
    max_steps: int
    val_check_interval: int | float | None
    gradient_clip_val: int | float | None
    task_steps: int | None
    precision: Literal[16, 32, 64] | Literal["16-true", "16-mixed", "bf16-true", "bf16-mixed", "32-true", "64-true"] | Literal["bf16", "16", "32", "64"] | None = None
    validate: bool = True
    accumulate_grad_batches: int=1
    limit_test_batches: int=32
    strategy: str = 'ddp_find_unused_parameters_true'

@dataclass
class RootCfg:
    wandb: dict
    mode: Literal["train", "val", "test"]
    dataset: DatasetCfg
    model: ModelCfg
    data_loader: DataLoaderCfg
    optimizer: OptimizerCfg
    checkpointing: CheckpointingCfg
    trainer: TrainerCfg
    test: TestCfg
    train: TrainCfg
    freeze: FreezeCfg
    seed: int | None
    scene_id: int | str | None = None


TYPE_HOOKS = {
    Path: Path,
}


T = TypeVar("T")


def load_typed_config(
    cfg: DictConfig,
    data_class: Type[T],
    extra_type_hooks: dict = {},
) -> T:
    return from_dict(
        data_class,
        OmegaConf.to_container(cfg),
        config=Config(type_hooks={**TYPE_HOOKS, **extra_type_hooks}),
    )

def load_typed_root_config(cfg: DictConfig) -> RootCfg:
    return load_typed_config(
        cfg,
        RootCfg,
        {},
    )
