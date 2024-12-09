import json
from dataclasses import dataclass
from pathlib import Path

import hydra
import torch
from jaxtyping import install_import_hook
from omegaconf import DictConfig
from pytorch_lightning import Trainer

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_config, load_typed_root_config
    from src.dataset.data_module import DataLoaderCfg, DataModule, DatasetCfg
    from src.global_cfg import set_cfg
    from src.model.diffusion_wrapper import DiffusionWrapper
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.config import RootCfg

@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="main",
)
def evaluate(cfg_dict: DictConfig):
    cfg = load_typed_config(cfg_dict, RootCfg)
    set_cfg(cfg_dict)
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
    print(cfg)

    with open(cfg.dataset.view_sampler.index_path) as f:
        indices = json.load(f)

    if type(cfg.scene_id) == str and cfg.scene_id in indices.keys():
        cfg.dataset.overfit_to_scene = [cfg.scene_id]
    elif type(int(cfg.scene_id)) == int:
        cfg.dataset.overfit_to_scene = [list(indices.keys())[int(cfg.scene_id)]]
    else:
        raise(f"Either scene index {cfg.scene_id} is not defined or does not exists in {dataset.view_sampler.index_path}.")
    
    output_dir = cfg.test.output_dir

    # Set up logging with wandb.
    logger = LocalLogger()

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker(cfg.train.step_offset)
    kwargs = dict(
        model_cfg=cfg.model,
        freeze_cfg = cfg.freeze,
        optimizer_cfg=cfg.optimizer,
        test_cfg=cfg.test,
        train_cfg=cfg.train,
        step_tracker=step_tracker,
        output_dir=output_dir
    )

    data_module = DataModule(cfg.dataset, cfg.data_loader, step_tracker)

    print("CUDA is Initialized: ", torch.cuda.is_initialized())

    model_wrapper = DiffusionWrapper.load_from_checkpoint(cfg.checkpointing.load, **kwargs)

    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        precision=cfg.trainer.precision,
        limit_val_batches=1 if cfg.trainer.validate else 0,
        val_check_interval=cfg.trainer.val_check_interval if cfg.trainer.validate else None,
        check_val_every_n_epoch=None,
        enable_progress_bar=True,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        limit_test_batches=1,
        limit_predict_batches=1
    )
    model_wrapper.freeze()
    model_wrapper.eval()


    trainer.test(model_wrapper, datamodule=data_module)



if __name__ == "__main__":
    evaluate()
