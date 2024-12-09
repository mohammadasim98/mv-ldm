import json
from dataclasses import dataclass
from pathlib import Path

import hydra
import torch
from jaxtyping import install_import_hook
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from cleanfid import fid

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_config
    from src.dataset.data_module import DataLoaderCfg, DataModule, DatasetCfg
    from src.evaluation.evaluation_cfg import EvaluationCfg
    from src.evaluation.metric_computer import MetricComputer
    from src.global_cfg import set_cfg


@dataclass
class RootCfg:
    evaluation: EvaluationCfg
    dataset: DatasetCfg
    data_loader: DataLoaderCfg
    seed: int
    output_metrics_path: Path
    per_scene_metrics_path: Path
    output_fid_path: Path

@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="compute_metrics",
)
def evaluate(cfg_dict: DictConfig):
    cfg = load_typed_config(cfg_dict, RootCfg)
    set_cfg(cfg_dict)
    torch.manual_seed(cfg.seed)
    fids = {}
    for method in cfg.evaluation.methods:
        
        fids[f"fidclean_{method.key}"] = fid.compute_fid(str(method.path), "gt_images")
        fids[f"kidclean_{method.key}"] = fid.compute_kid(str(method.path), "gt_images")
        
    
    with cfg.output_fid_path.open("w") as f:
        json.dump(fids, f)



if __name__ == "__main__":
    evaluate()
