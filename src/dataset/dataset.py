from dataclasses import dataclass

from .view_sampler import ViewSamplerCfg


@dataclass
class DatasetCfgCommon:
    image_shape: list[int]
    background_color: list[float]
    cameras_are_circular: bool
    overfit_to_scene: str | int | list | None
    view_sampler: ViewSamplerCfg
    scene: str | None
    augment: bool
    random_transform_extrinsics: bool