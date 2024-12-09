from dataclasses import dataclass
from pathlib import Path
from typing import Literal

@dataclass
class MethodCfg:
    name: str
    key: str
    path: Path


@dataclass
class ModalityCfg:
    name: str
    key: str


@dataclass
class SceneCfg:
    scene: str
    context_index: list[int]
    target_index: int | list[int]

@dataclass
class MVCCfg:
    subsample_or_initxy1: int=8
    corres_weighting: bool=False
    conf_weighting: bool=True
    confidence_key: str="conf"
    reduction: str | None="mean"
    return_matches: bool=False

@dataclass
class MVSSIMCfg:
    subsample_or_initxy1: int=8
    corres_weighting: bool=False
    conf_weighting: bool=True
    confidence_key: str="conf"
    reduction: str | None="mean"
    return_matches: bool=False
    compute_intrinsics: bool=False

@dataclass
class EvaluationCfg:
    methods: list[MethodCfg] | MethodCfg 
    side_by_side_path: Path | None
    animate_side_by_side: bool
    highlighted: list[SceneCfg]
    modalities: list[ModalityCfg] | None = None

@dataclass
class MVCEvaluationCfg:
    methods: list[MethodCfg] | MethodCfg
    weights: str
    mvc_cfg: MVCCfg
    output_path: Path | None
    types: Literal["fixed", "pairwise"]="fixed"
    img_size: int=224


class MVSSIMEvaluationCfg:

    methods: list[MethodCfg] | MethodCfg | None = None
    weights: str
    mvssim_cfg: MVSSIMCfg
    gt_directory: str | Path
    output_path: Path | None
    types: Literal["fixed", "pairwise"]="fixed"
    img_size: int=224
    gap: int=1
    nchuncks: int=20
class GenwarpEvaluationCfg:

    img_size: int=224
    output_path: Path = Path("/BS/grl-co3d/work/mv_diff/genwarp")
    ind: int | None = None
