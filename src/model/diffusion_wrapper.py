"""
"""

# Standard Imports
import os
import PIL
import math
import hydra
from tqdm import tqdm
from pathlib import Path
from heapq import nsmallest
from jaxtyping import Float
from dataclasses import dataclass
from einops import repeat, rearrange
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from typing import Any, Dict, Iterator, Literal, Optional

# Torch and NumPy Imports
import torch
import numpy as np
import torch.nn.functional as F
from torch import Tensor, optim, nn
from torch.nn import Module, Parameter
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

# Pytorch Lightning and Diffusers Import
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from diffusers.utils.import_utils import is_xformers_available

# Custom Imports
from .config import ModelCfg, OptimizerCfg, TestCfg, TrainCfg, FreezeCfg, LRSchedulerCfg

from .denoiser import get_denoiser
from .encodings.positional_encoding import PositionalEncoding
from .srt.layers import RayEncoder
from .scheduler import get_scheduler
from .autoencoder import get_autoencoder

from ..misc.camera_utils import absolute_to_relative_camera
from ..misc.step_tracker import StepTracker
from ..misc.image_io import prep_image, save_image, get_hist_image
from ..visualization.validation_in_3d import render_cameras
from ..visualization.annotation import add_label
from ..visualization.layout import add_border, hcat, vcat
from ..dataset.types import BatchedExample
from ..geometry.projection import get_world_rays, sample_image_grid

def freeze(m: Module) -> None:
    for param in m.parameters():
        param.requires_grad = False
    m.eval()


def unfreeze(m: Module) -> None:
    for param in m.parameters():
        param.requires_grad = True
    m.train()

class DiffusionWrapper(LightningModule):
    logger: Optional[WandbLogger]
    model_cfg: ModelCfg
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    freeze_cfg: FreezeCfg
    step_tracker: StepTracker | None
    output_dir: Path | None = None

    def __init__(
        self,
        model_cfg: ModelCfg,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        freeze_cfg: FreezeCfg,
        step_tracker: StepTracker | None,
        output_dir: Path | None = None
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.model_cfg = model_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.freeze_cfg = freeze_cfg
        self.step_tracker = step_tracker
        self.output_dir = output_dir

        print("Using Plucker Coordinates: ", model_cfg.use_plucker)
        print("Using SRT Ray Encoding: ", self.model_cfg.srt_ray_encoding)
        print("Using Standard Ray Encoding: ", model_cfg.use_ray_encoding)
        print("Using SD VAE from: ", model_cfg.denoiser.pretrained_from)
        print("Using EMA: ", self.model_cfg.ema)
        print("Using Memory Efficient Attention: ", self.model_cfg.enable_xformers_memory_efficient_attention)
        

        in_channels = model_cfg.autoencoder.kwargs.latent_channels
        out_channels = model_cfg.autoencoder.kwargs.latent_channels

        if self.model_cfg.srt_ray_encoding:
            self.ray_encoder = RayEncoder(
                pos_octaves=model_cfg.ray_encodings.num_origin_octaves,
                ray_octaves=model_cfg.ray_encodings.num_direction_octaves
            )
            in_channels += 2 * (
                model_cfg.ray_encodings.num_origin_octaves * 3 + 
                model_cfg.ray_encodings.num_direction_octaves * 3
            )
        
        else:
            self.ori_encoder = nn.Identity()
            self.dir_encoder = nn.Identity()
            if model_cfg.use_ray_encoding:
                if model_cfg.ray_encodings.num_origin_octaves > 0:
                    self.ori_encoder = PositionalEncoding(model_cfg.ray_encodings.num_origin_octaves)
                    ori_dim = self.ori_encoder.d_out(3)
                    in_channels += ori_dim
                if model_cfg.ray_encodings.num_direction_octaves > 0:
                    self.dir_encoder = PositionalEncoding(model_cfg.ray_encodings.num_direction_octaves)
                    dir_dim = self.dir_encoder.d_out(3)
                    in_channels += dir_dim 

            else:
                in_channels += 3 + 3
        
        in_channels += 1
        
        self.denoiser = get_denoiser(model_cfg.denoiser, in_channels, out_channels)
        self.autoencoder = get_autoencoder(model_cfg.autoencoder)
        self.scheduler = get_scheduler(model_cfg.scheduler)

        if self.freeze_cfg.denoiser:
            freeze(self.denoiser)
        if self.freeze_cfg.autoencoder:
            freeze(self.autoencoder)

        if self.model_cfg.ema:
            self.ema = AveragedModel(
                self.denoiser, 
                multi_avg_fn=get_ema_multi_avg_fn(0.995)
            )
   
        if self.model_cfg.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers
                self.denoiser.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly")

    def on_before_zero_grad(self, *args, **kwargs):
        if self.model_cfg.ema:
            self.ema.update_parameters(self.denoiser)   

    def setup(self, stage: str) -> None:
        # Scale base learning rates to effective batch size
        if stage == "fit":
            # assumes one fixed batch_size for all train dataloaders!
            effective_batch_size = self.trainer.accumulate_grad_batches \
                * self.trainer.num_devices \
                * self.trainer.num_nodes \
                * self.trainer.datamodule.data_loader_cfg.train.batch_size

            self.lr = effective_batch_size * self.optimizer_cfg.lr \
                if self.optimizer_cfg.scale_lr else self.optimizer_cfg.lr
        return super().setup(stage)

    def generate_image_rays(
        self,
        images: Float[Tensor, "batch view channel height width"],
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
    ) -> tuple[
        Float[Tensor, "batch view ray 2"],  # xy
        Float[Tensor, "batch view ray 3"],  # origins
        Float[Tensor, "batch view ray 3"],  # directions
    ]:
        """Generate the rays along which Gaussians are defined. For now, these rays are
        simply arranged in a grid.
        """
        b, v, _, h, w = images.shape
        device, dtype = extrinsics.device, extrinsics.dtype
        xy, _ = sample_image_grid((h, w), device=device, dtype=dtype)
        origins, directions = get_world_rays(
            rearrange(xy, "h w xy -> (h w) xy"),
            rearrange(extrinsics, "b v i j -> b v () i j"),
            rearrange(intrinsics, "b v i j -> b v () i j"),
        )
        return repeat(xy, "h w xy -> b v (h w) xy", b=b, v=v), origins, directions

    def set_timesteps(self, num: Optional[int]=None):
        """
            Args:
                num (Optional[int]): Override the number of inference steps. Default to None.
        """
        num_inference_timesteps = self.model_cfg.scheduler.num_inference_steps if num is None else num
        self.scheduler.set_timesteps(num_inference_timesteps)    
        
    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        print("Setting Max Timesteps for validation to: ", self.model_cfg.scheduler.num_train_timesteps)
        self.set_timesteps()
    def on_test_batch_start(self, batch, batch_idx, dataloader_idx=0):
        print("Setting Max Timesteps for testing to: ", self.model_cfg.scheduler.num_train_timesteps)
        self.set_timesteps()
    def on_predict_batch_start(self, batch, batch_idx, dataloader_idx=0):
        print("Setting Max Timesteps for prediction to: ", self.model_cfg.scheduler.num_train_timesteps)
        self.set_timesteps()
    def on_validation_batch_end(self, batch, batch_idx, dataloader_idx=0):
        print("Setting Max Timesteps for training to: ", self.model_cfg.scheduler.num_train_timesteps)
        self.scheduler.set_timesteps(self.model_cfg.scheduler.num_train_timesteps)
    
    def sample_indices(self, batch, index: int | Tensor, random: int=True):
        b, v_t, c, h, w = batch["target"]["image"].shape
        b, v_c, c, h, w = batch["context"]["image"].shape
        device = batch["context"]["image"].device
        
        
        if index > 1:
            context_image = batch["context"]["image"][:, :index, ...]
            context_extrinsics = batch["context"]["extrinsics"][:, :index, ...]
            context_intrinsics = batch["context"]["intrinsics"][:, :index, ...]
            context_near = batch["context"]["near"][:, :index, ...]
            context_far = batch["context"]["far"][:, :index, ...]
            context_index = batch["context"]["index"][:, :index]
            rel_index = torch.randint(0, index, size=(1,), dtype=torch.long).to(context_far.device).item()
        else:
            if random:
                index = torch.randint(0, v_c, size=(1,), dtype=torch.long).to(device).item()
            else: 
                index = 0
                
            index_mask = torch.zeros((v_c,)).cuda()
            index_mask[index] = 1.0
            
            index_mask = index_mask.bool()
            context_image = batch["context"]["image"][:, index_mask, ...]
            context_extrinsics = batch["context"]["extrinsics"][:, index_mask, ...]
            context_intrinsics = batch["context"]["intrinsics"][:, index_mask, ...]
            context_near = batch["context"]["near"][:, index_mask, ...]
            context_far = batch["context"]["far"][:, index_mask, ...]
            context_index = batch["context"]["index"][:, index_mask]
            
            target_image = torch.concat([batch["target"]["image"],  batch["context"]["image"][:, ~index_mask, ...]], dim=1)
            target_extrinsics = torch.concat([batch["target"]["extrinsics"],  batch["context"]["extrinsics"][:, ~index_mask, ...]], dim=1)
            target_intrinsics = torch.concat([batch["target"]["intrinsics"],  batch["context"]["intrinsics"][:, ~index_mask, ...]], dim=1)
            target_near = torch.concat([batch["target"]["near"],  batch["context"]["near"][:, ~index_mask, ...]], dim=1)
            target_far = torch.concat([batch["target"]["far"],  batch["context"]["far"][:, ~index_mask, ...]], dim=1)
            target_index = torch.concat([batch["target"]["index"],  batch["context"]["index"][:, ~index_mask, ...]], dim=1)

            
            rel_index = index
            batch = {
                "context": batch["context"],
                "target": {
                    "image": target_image,
                    "extrinsics": target_extrinsics,
                    "intrinsics": target_intrinsics,
                    "near": target_near,
                    "far": target_far,
                    "index": target_index,
                },
                "scene": batch["scene"]
            }
        return {
                "context": {
                    "image": context_image,
                    "extrinsics": context_extrinsics,
                    "intrinsics": context_intrinsics,
                    "near": context_near,
                    "far": context_far,
                    "index": context_index,
                },
                "target": batch["target"],
                "scene": batch["scene"]
            }, rel_index
    
    def first_stage_encode(self, inputs):
        b, v, c, h, w = inputs.shape
        inputs = rearrange(inputs, "b v c h w -> (b v) c h w ")
        inputs = inputs * 2.0 - 1.0 
        with torch.no_grad():  
            latents = self.autoencoder.encode(inputs).latent_dist.sample() * 0.18215

        latents = rearrange(latents, "(b v) c h w -> b v c h w", b=b)
        
        return latents
    
    def last_stage_decode(self, latents):
        b, v, c, h, w = latents.shape
        latents = rearrange(latents, "b v c h w -> (b v) c h w ")

        latents = (1 / 0.18215) * latents     
        with torch.no_grad(): 
            image = self.autoencoder.decode(latents).sample  
        image = rearrange(image, "(b v) c h w -> b v c h w", b=b)

        return (image / 2 + 0.5).clamp(0, 1) 

        
    def ray_encode(self, batch, context_latents, target_latents):
        b, *_, hl, wl = context_latents.shape

        _, origins_context, directions_context = self.generate_image_rays(context_latents, batch["context"]["extrinsics"], batch["context"]["intrinsics"])
        _, origins_target, directions_target = self.generate_image_rays(target_latents, batch["target"]["extrinsics"], batch["target"]["intrinsics"])

        origins = torch.concat([origins_context, origins_target], dim=1)
        directions = torch.concat([directions_context, directions_target], dim=1)
        if self.model_cfg.use_plucker:
            origins = torch.cross(origins, directions, dim=-1)
        if self.model_cfg.srt_ray_encoding:
            origins = rearrange(origins, "b v (h w) c -> (b v) (h w) c", h=hl, w=wl)
            directions = rearrange(directions, "b v (h w) c -> (b v) (h w) c", h=hl, w=wl)
            ray_encodings = self.ray_encoder(origins, directions)
            ray_encodings = rearrange(ray_encodings, "(b v) (h w) c -> b v c h w", b=b, h=hl, w=wl)
        else:
            origins_enc = self.ori_encoder(origins) 
            directions_enc = self.dir_encoder(directions) 
            ray_encodings = torch.concat([origins_enc, directions_enc], dim=-1)   
            ray_encodings = rearrange(ray_encodings, "b v (h w) c -> b v c h w", h=hl, w=wl)
        
        return ray_encodings
    
    def training_step(self, batch, batch_idx):
        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)
            self.log(f"step_tracker/step", self.step_tracker.get_step())

        
        b, v_c, c, h, w = batch["context"]["image"].shape
        b, v_t, c, h, w = batch["target"]["image"].shape
        device = batch["context"]["image"].device

        # Prepare context and target views 
        index = torch.randint(1, v_c+1, size=(1,), dtype=torch.long).to(device).item()
        batch, rel_index = self.sample_indices(batch, index, True)
        
        b, v_t, c, h, w = batch["target"]["image"].shape
        b, v_c, c, h, w = batch["context"]["image"].shape
        
        
        # Randomly choose between absolute and relative poses
        concat_extr = torch.concat([batch["context"]["extrinsics"], batch["target"]["extrinsics"]], dim=1)
        relative_pose = np.random.choice([False, True], 1, p=[0.50, 0.50])
        if relative_pose == 0:
            # Transform from absolute to relative poses
            rel_extrinsics = absolute_to_relative_camera(concat_extr, index=rel_index).float()
        else:
            rel_extrinsics = concat_extr
        
        batch["context"]["extrinsics"] = rel_extrinsics[:, :v_c, ...]
        batch["target"]["extrinsics"] = rel_extrinsics[:, v_c:, ...]

        # Encode context and target views with VAE
        inputs = torch.concat([batch["context"]["image"], batch["target"]["image"]], dim=1)
        latents = self.first_stage_encode(inputs)
        context_latents = latents[:, :v_c, ...]
        target_latents = latents[:, v_c:, ...]
        
        # Add noise to target views
        target_noise = torch.randn_like(target_latents).to(target_latents.device)  
        timestep_target = torch.randint(
            0, 
            self.model_cfg.scheduler.num_train_timesteps, 
            size=(b,), 
            device=self.device,
            dtype=torch.long
        ) 
        noisy_latents = self.scheduler.add_noise(target_latents, target_noise, timestep_target)
        
        # Prepare context and target inputs to diffusion model
        ray_encodings = self.ray_encode(batch, context_latents, target_latents)
        target_mask = torch.ones((*target_latents.shape[:2], 1, *target_latents.shape[3:])).to(target_latents.device)
        context_mask = torch.zeros((*context_latents.shape[:2], 1, *context_latents.shape[3:])).to(context_latents.device)
        target_inputs = torch.concat([noisy_latents, target_mask], dim=2)
        timestep_context = torch.zeros((b, ), dtype=torch.long).to(context_latents.device)        
        
        unconditional = True
        if self.train_cfg.cfg_train:
            # Randomly choose to train conditionally or unconditionally
            unconditional = np.random.choice([False, True], 1, p=[0.90, 0.10])
        
        if unconditional:
            # For unconditionl, drop the context views
            ray_encodings = ray_encodings[:, v_c:, ...]
            inputs = target_inputs
            timesteps = repeat(timestep_target, "b -> b vt", vt=v_t)
        else:
            context_inputs = torch.concat([context_latents, context_mask], dim=2)
            inputs = torch.concat([context_inputs, target_inputs], dim=1)
            timesteps = torch.concat([
                    repeat(timestep_context, "b -> b vc", vc=v_c),
                    repeat(timestep_target, "b -> b vt", vt=v_t) 
                ], 
                dim=1
            )
        inputs = torch.concat([inputs, ray_encodings], dim=2)

        # Denoise
        pred = self.denoiser.forward(latents=inputs, timestep=timesteps)
        if unconditional:
            pred_out = pred
        else:
            pred_out = pred[:, v_c:, ...]

        # Diffusion Loss
        loss = F.mse_loss(pred_out.float(), target_latents.float(), reduction="mean")

        self.log("loss/diffusion", loss)   
        return loss
    
    def step(self, model, x_t, ts, context_inputs, ray_encodings, target_mask):
        
        b, v_c, *_ = context_inputs.shape 
        b, v_t, *_ = x_t.shape 
        x_t_scaled = self.scheduler.scale_model_input(x_t, ts)

        timestep_context = torch.tensor([0], device=self.device, dtype=torch.long)
        timestep_target = ts.to(torch.long).to(x_t.device).unsqueeze(0)
        timestep_target = repeat(timestep_target, "() -> b", b=b)
        timestep_context = repeat(timestep_context, "() -> b", b=b)
        timesteps = torch.concat([
                repeat(timestep_context, "b -> b vc", vc=v_c),
                repeat(timestep_target, "b -> b vt", vt=v_t) 
            ], 
            dim=1
        )
        target_inputs = torch.concat([x_t_scaled, target_mask], dim=2)
        
        inputs = torch.concat([context_inputs, target_inputs], dim=1)
        inputs = torch.concat([inputs, ray_encodings], dim=2)

        # Conditional Forward Pass
        pred_conditional = model.forward(inputs, timesteps)

        if self.model_cfg.use_cfg:
            inputs = torch.concat([target_inputs, ray_encodings[:, v_c:, ...]], dim=2)
            timesteps = repeat(timestep_target, "b -> b vt", vt=v_t)
            # Unconditional Forward Pass
            pred_unconditional = model.forward(inputs, timesteps)

            # CFG Compose
            pred_out = pred_unconditional + self.model_cfg.cfg_scale * (pred_conditional[:, v_c:, ...] - pred_unconditional) 
        else:
                
            pred_out = pred_conditional[:, v_c:, ...]


        # print(self.scheduler.config.prediction_type)
        sch_out = self.scheduler.step(pred_out, ts, x_t)

        return sch_out.prev_sample
    
    def sample(self, batch):
        
        b, v_c, c, h, w = batch["context"]["image"].shape
        b, v_t, c, h, w = batch["target"]["image"].shape

        if self.model_cfg.use_ema_sampling:
            print("Loading EMA weights")
            # self.ema.copy_to(self.denoiser)
            model = self.ema
        else:
            model = self.denoiser

        inputs = batch["context"]["image"]

        latents = self.first_stage_encode(inputs)

        context_latents = latents
 
        x_t = torch.randn((context_latents.shape[0], v_t, *context_latents.shape[2:])).to(context_latents.device)  
        x_t *= self.scheduler.init_noise_sigma 
            
        target_mask = torch.ones((*x_t.shape[:2], 1, *x_t.shape[3:])).to(x_t.device)
        context_mask = torch.zeros((*context_latents.shape[:2], 1, *context_latents.shape[3:])).to(context_latents.device)

        context_inputs = torch.concat([context_latents, context_mask], dim=2)

        
        *_, hl, wl = context_latents.shape

        ray_encodings = self.ray_encode(batch, context_latents, x_t)

        for i, ts in enumerate(tqdm(self.scheduler.timesteps)):

            x_t = self.step(model, x_t, ts, context_inputs, ray_encodings, target_mask)

        return self.last_stage_decode(x_t), batch
            
    @rank_zero_only
    def validation_step(self, batch, batch_idx):

        device = batch["context"]["image"].device
        

        b, v_c, c, h, w = batch["context"]["image"].shape

        cameras = hcat(*render_cameras(batch, 256))
        self.logger.log_image(
            "cameras", [prep_image(add_border(cameras))], step=self.global_step
        )            
            
        index = 1
        batch, rel_index = self.sample_indices(batch, index, True)

        b, v_c, c, h, w = batch["context"]["image"].shape
        rel_extrinsics = absolute_to_relative_camera(torch.concat([batch["context"]["extrinsics"], batch["target"]["extrinsics"]], dim=1), index=rel_index).float()
        batch["context"]["extrinsics"] = rel_extrinsics[:, :v_c, ...]
        batch["target"]["extrinsics"] = rel_extrinsics[:, v_c:, ...]
        
        sampled_views, batch = self.sample(batch)
        sampled_views = sampled_views.clip(0.0, 1.0)
        context_views = batch["context"]["image"]
        target_views = batch["target"]["image"]
        b, v_t, *_ = sampled_views.shape
        b, v_c, *_ = context_views.shape
        latents = self.first_stage_encode(target_views)
        target_views = self.last_stage_decode(latents)
        for j in range(b):
            scene = batch["scene"][j]

            context_vis = add_label(vcat(*[context_views[j, i, ...] for i in range(v_c)]), "Context Views")
            target_vis = add_label(vcat(*[target_views[j, i, ...] for i in range(v_t)]), "Original Targets")
            sample_vis = add_label(vcat(*[sampled_views[j, i, ...] for i in range(v_t)]), "Sampled Targets")
            vis = hcat(context_vis, target_vis, sample_vis)
            self.logger.log_image(
                f"Sampled Comparison {j} ",
                [prep_image(vis)],
                step=self.step_tracker.get_step(),
                caption=[f"Sampled Comparison ({scene})"],
            )
        sampled_hist = get_hist_image(sampled_views, title=f"Target Distribution")
        target_hist = get_hist_image(target_views, title=f"Sampled Distribution")
        
        hist_vis = hcat(target_hist, sampled_hist)

        self.logger.log_image(
            f"Distributions",
            [prep_image(hist_vis)],
            step=self.step_tracker.get_step(),
            caption=["Histogram"],
        )
    def test_batch(self, batch, batch_idx):

        b, v_c = batch["context"]["extrinsics"].shape[:2]
    
        output_dir = self.output_dir / "test"
        names = []
        if os.path.exists(output_dir):  
            names = os.listdir(output_dir)
        # # print(names)
        exclude_index = []
        include_index = []
        for i, scene in enumerate(batch['scene']):    
            if scene in names:
                exclude_index.append(i)
                print("Skipping: ", scene)
            else:
                os.makedirs(output_dir / scene, exist_ok=True)
                include_index.append(i)
                
        if len(include_index) <= 0:
            return

        scenes = []
        for index in include_index:
            names.append(batch['scene'][index])
            scenes.append(batch['scene'][index])
        batch['scene'] = scenes
        mask = torch.ones((b,)).bool()
        mask[exclude_index] = False
        
        new_batch = {
            "context": {
                "image": batch["context"]["image"][mask, ...],
                "extrinsics": batch["context"]["extrinsics"][mask, ...],
                "intrinsics": batch["context"]["intrinsics"][mask, ...],
                "near": batch["context"]["near"][mask, ...],
                "far": batch["context"]["far"][mask, ...],
                "index": batch["context"]["index"][mask, ...],
            },
            "target": {
                "image": batch["target"]["image"][mask, ...],
                "extrinsics": batch["target"]["extrinsics"][mask, ...],
                "intrinsics": batch["target"]["intrinsics"][mask, ...],
                "near": batch["target"]["near"][mask, ...],
                "far": batch["target"]["far"][mask, ...],
                "index": batch["target"]["index"][mask, ...],
            },
            "scene": batch["scene"]

        }
        batch = new_batch

        b, v_c = batch["context"]["extrinsics"].shape[:2]
        b, v_t = batch["target"]["extrinsics"].shape[:2]

        device = batch["context"]["image"].device
        index = 1
        context_image = batch["context"]["image"][:, :index, ...]
        context_extrinsics = batch["context"]["extrinsics"][:, :index, ...]
        context_intrinsics = batch["context"]["intrinsics"][:, :index, ...]
        context_near = batch["context"]["near"][:, :index, ...]
        context_far = batch["context"]["far"][:, :index, ...]
        context_index = batch["context"]["index"][:, :index]
        
        batch = {
            "context": {
                "image": context_image,
                "extrinsics": context_extrinsics,
                "intrinsics": context_intrinsics,
                "near": context_near,
                "far": context_far,
                "index": context_index,
            },
            "target": batch["target"],
            "scene": batch["scene"]
        }
        b, v_c, c, h, w = batch["context"]["image"].shape
        rel_index = 0
        rel_extrinsics = absolute_to_relative_camera(torch.concat([batch["context"]["extrinsics"], batch["target"]["extrinsics"]], dim=1), index=rel_index).float()
        batch["context"]["extrinsics"] = rel_extrinsics[:, :v_c, ...]
        batch["target"]["extrinsics"] = rel_extrinsics[:, v_c:, ...]
        
        
        sampled_views, batch = self.sample(batch)
        sampled_views = sampled_views.clip(0.0, 1.0)
        context_views = batch["context"]["image"]
        target_views = batch["target"]["image"]
        b, v_t, *_ = sampled_views.shape
        b, v_c, *_ = context_views.shape
        
        for i in range(b):
            scene = batch["scene"][i]

            for index, color in zip(batch["target"]["index"][i], sampled_views[i]):
                save_image(color, output_dir / scene / f"color/{index:0>6}.png")

            for index, color in zip(batch["context"]["index"][i], batch["context"]["image"][i]):
                save_image(color, output_dir / scene / f"context/{index:0>6}.png")

    def test_video_anchored(self, batch, batch_idx, return_predictions: bool=False, limit_frames: int | None=None):
        if limit_frames is not None:
            batch = {
                "context": {
                    "image": batch["context"]["image"],
                    "extrinsics": batch["context"]["extrinsics"],
                    "intrinsics": batch["context"]["intrinsics"],
                    "near": batch["context"]["near"],
                    "far": batch["context"]["far"],
                    "index": batch["context"]["index"],
                },
                "target": {
                    "image": batch["target"]["image"][:, :limit_frames, ...],
                    "extrinsics": batch["target"]["extrinsics"][:,  :limit_frames, ...],
                    "intrinsics": batch["target"]["intrinsics"][:,  :limit_frames, ...],
                    "near": batch["target"]["near"][:,  :limit_frames, ...],
                    "far": batch["target"]["far"][:,  :limit_frames, ...],
                    "index": batch["target"]["index"][:,  :limit_frames, ...],
                },
                "scene": batch["scene"]
            }
        if self.global_rank == 0:
            print(
                f"test step {self.step_tracker.get_step()}; "
                f"scene = {batch['scene'][0]}; "
                f"context = {batch['context']['index'][0].tolist()} "
                f"target = {batch['target']['index'][0].tolist()}"
            )
        b, v_c = batch["context"]["extrinsics"].shape[:2]
        assert b == 1, "Batch Size must be 1 for sampling video"
        output_dir = self.output_dir / "video"

        index = 1 
        batch, rel_index = self.sample_indices(batch, index, False)
        b, v_c, c, h, w = batch["context"]["image"].shape
        b, num_t, c, h, w = batch["target"]["image"].shape
        rel_index = 0
        rel_extrinsics = absolute_to_relative_camera(torch.concat([batch["context"]["extrinsics"], batch["target"]["extrinsics"]], dim=1), index=rel_index).float()
        batch["context"]["extrinsics"] = rel_extrinsics[:, :v_c, ...]
        batch["target"]["extrinsics"] = rel_extrinsics[:, v_c:, ...]
        n_anchors = self.test_cfg.num_anchors_views
        
        print("Number of Anchors: ", n_anchors)
        
        # Sample anchor views
        anchor_step = len(batch["target"]["index"][0]) // n_anchors
        
        
        context = batch["context"]
        anchor_images = []
        start = anchor_step
        end = 5*anchor_step
        
        anchors_indices = batch["target"]["index"][:, start:end:anchor_step]
            
        indices = []
            
        print(f"{0} - Context Indices: ", context["index"][0].tolist())
        print(f"{0} - Generating Anchors: ", anchors_indices[0].tolist())
        
        anchor_batch = {
            "context": context,
            "target": {
                "image": batch["target"]["image"][:, start:end:anchor_step, ...],
                "extrinsics": batch["target"]["extrinsics"][:,start:end:anchor_step, ...],
                "intrinsics": batch["target"]["intrinsics"][:, start:end:anchor_step, ...],
                "near": batch["target"]["near"][:, start:end:anchor_step, ...],
                "far": batch["target"]["far"][:, start:end:anchor_step, ...],
                "index": anchors_indices,
            },
            "scene": batch["scene"]
        }
        rel_index = 0
        rel_extrinsics = absolute_to_relative_camera(torch.concat([anchor_batch["context"]["extrinsics"], anchor_batch["target"]["extrinsics"]], dim=1), index=rel_index).float()
        
        curr_batch = anchor_batch
        curr_batch["context"]["extrinsics"] = rel_extrinsics[:, :v_c, ...]
        curr_batch["target"]["extrinsics"] = rel_extrinsics[:, v_c:, ...]
        
        anchor_views, curr_batch = self.sample(curr_batch)
        anchor_views = anchor_views.clip(0.0, 1.0)
        anchor_images.append(anchor_views)
        
        context_image = torch.concat([batch["context"]["image"], anchor_views[:, -1:, ...]], dim=1)
        context_extrinsics = torch.concat([batch["context"]["extrinsics"], anchor_batch["target"]["extrinsics"][:, -1:, ...]], dim=1)
        context_intrinsics = torch.concat([batch["context"]["intrinsics"], anchor_batch["target"]["intrinsics"][:, -1:, ...]], dim=1)
        context_near = torch.concat([batch["context"]["near"], anchor_batch["target"]["near"][:, -1:, ...]], dim=1)
        context_far = torch.concat([batch["context"]["far"], anchor_batch["target"]["far"][:, -1:, ...]], dim=1)
        context_index = torch.concat([batch["context"]["index"], anchor_batch["target"]["index"][:, -1:, ...]], dim=1)
        
        context = {
            "image": context_image,
            "extrinsics": context_extrinsics,
            "intrinsics": context_intrinsics,
            "near": context_near,
            "far": context_far,
            "index": context_index,
        }
            
        for i in range(1, int(math.ceil((n_anchors - 4) / 3)) + 1):
            
            start = (i-1) * 3 * anchor_step
            end = i * 3 * anchor_step
            anchors_indices = batch["target"]["index"][:, start + 4*anchor_step:end + 4*anchor_step:anchor_step]
            
            
            b, v_c, c, h, w = context["image"].shape

            print(f"{i} - Generating Anchors: ", anchors_indices[1:])
            print(f"{i} - Context Indices: ", context["index"])
 
            anchor_batch = {
                "context": context,
                "target": {
                    "image": batch["target"]["image"][:, start:end:anchor_step, ...],
                    "extrinsics": batch["target"]["extrinsics"][:,start:end:anchor_step, ...],
                    "intrinsics": batch["target"]["intrinsics"][:, start:end:anchor_step, ...],
                    "near": batch["target"]["near"][:, start:end:anchor_step, ...],
                    "far": batch["target"]["far"][:, start:end:anchor_step, ...],
                    "index": anchors_indices,
                },
                "scene": batch["scene"]
            }
            rel_index = 1
            rel_extrinsics = absolute_to_relative_camera(torch.concat([anchor_batch["context"]["extrinsics"], anchor_batch["target"]["extrinsics"]], dim=1), index=rel_index).float()
            anchor_batch["context"]["extrinsics"] = rel_extrinsics[:, :v_c, ...]
            anchor_batch["target"]["extrinsics"] = rel_extrinsics[:, v_c:, ...]
            
            anchor_views, anchor_batch = self.sample(anchor_batch)
            anchor_views = anchor_views.clip(0.0, 1.0)

            anchor_images.append(anchor_views)
            context_image = torch.concat([batch["context"]["image"], anchor_views[:, -1:, ...]], dim=1)
            context_extrinsics = torch.concat([batch["context"]["extrinsics"], anchor_batch["target"]["extrinsics"][:, -1:, ...]], dim=1)
            context_intrinsics = torch.concat([batch["context"]["intrinsics"], anchor_batch["target"]["intrinsics"][:, -1:, ...]], dim=1)
            context_near = torch.concat([batch["context"]["near"], anchor_batch["target"]["near"][:, -1:, ...]], dim=1)
            context_far = torch.concat([batch["context"]["far"], anchor_batch["target"]["far"][:, -1:, ...]], dim=1)
            context_index = torch.concat([batch["context"]["index"], anchor_batch["target"]["index"][:, -1:, ...]], dim=1)
            
            context = {
                "image": context_image,
                "extrinsics": context_extrinsics,
                "intrinsics": context_intrinsics,
                "near": context_near,
                "far": context_far,
                "index": context_index,
            }
            
        
        
        anchor_images = torch.concat(anchor_images, dim=1)  
        end = -1
        if anchor_images.shape[1] < batch["target"]["extrinsics"][:, ::anchor_step, ...].shape[1]:
            end = anchor_images.shape[1]
        anchors_indices = batch["target"]["index"][:, ::anchor_step][:, 1:end+1]
        anchors_extrinsics = batch["target"]["extrinsics"][:, ::anchor_step, ...][:, 1:end+1]
        anchors_intrinsics = batch["target"]["intrinsics"][:, ::anchor_step, ...][:, 1:end+1]
        anchors_near = batch["target"]["near"][:, ::anchor_step, ...][:, 1:end+1]
        anchors_far = batch["target"]["far"][:, ::anchor_step, ...][:, 1:end+1]


        anchor_views = anchor_images
        
        context_views = batch["context"]["image"]
        target_views = batch["target"]["image"]
        b, v_t, *_ = anchor_views.shape
        b, v_c, *_ = context_views.shape
        
        scene = batch["scene"][0]
    
        samples = []
        # Save Anchors
        for index, color in zip(anchors_indices[0], anchor_views[0]):
            save_image(color, output_dir / scene / f"color/{index:0>6}.png")

        remaining_target_indices = batch["target"]["index"][0].tolist()
        
        # Remove Anchors from targets
        for idx in anchors_indices[0]:
            remaining_target_indices.pop(remaining_target_indices.index(idx.item()))
                    
        assigned_anchors = [nsmallest(1, anchors_indices[0], key=lambda x: abs(x - idx))[0] for idx in remaining_target_indices]

        anc_to_indices = {anc.item(): [] for anc in anchors_indices[0]}
        tmp = []
        for anc in anchors_indices[0]:
            
            for i, idx in enumerate(assigned_anchors):
                if idx == anc:
                    tmp.append(remaining_target_indices[i])

                if len(tmp) == 3:
                    anc_to_indices[anc.item()].append(tmp)
                    tmp = []

        for key in anc_to_indices.keys():
            i = anchors_indices[0].tolist().index(torch.tensor(key).cuda())
            
            
            context_image = torch.concat([batch["context"]["image"], anchor_views[:, i:i+1, ...]], dim=1)
            context_extrinsics = torch.concat([batch["context"]["extrinsics"], anchors_extrinsics[:, i:i+1, ...]], dim=1)
            context_intrinsics = torch.concat([batch["context"]["intrinsics"], anchors_intrinsics[:, i:i+1, ...]], dim=1)
            context_near = torch.concat([batch["context"]["near"], anchors_near[:, i:i+1, ...]], dim=1)
            context_far = torch.concat([batch["context"]["far"], anchors_far[:, i:i+1, ...]], dim=1)
            context_index = torch.concat([batch["context"]["index"], anchors_indices[:, i:i+1, ...]], dim=1)
            
            
            b, v_c, *_ = context_image.shape
            for target in anc_to_indices[key]:
                if len(target) == 0:
                    continue
                target_indices = []
                for ele in target:
                    target_indices.append(batch["target"]["index"][0].tolist().index(torch.tensor(ele).cuda()))

                curr_batch = {
                    "context": {
                        "image": context_image,
                        "extrinsics": context_extrinsics,
                        "intrinsics": context_intrinsics,
                        "near": context_near,
                        "far": context_far,
                        "index": context_index,
                    },
                    "target": {
                        "image": batch["target"]["image"][:, target_indices, ...],
                        "extrinsics": batch["target"]["extrinsics"][:,  target_indices, ...],
                        "intrinsics": batch["target"]["intrinsics"][:,  target_indices, ...],
                        "near": batch["target"]["near"][:,  target_indices, ...],
                        "far": batch["target"]["far"][:,  target_indices, ...],
                        "index": batch["target"]["index"][:,  target_indices, ...],
                    },
                    "scene": batch["scene"]
                }
                b, v_t, *_ = curr_batch["target"]["image"].shape
                rel_index = 1
                rel_extrinsics = absolute_to_relative_camera(torch.concat([curr_batch["context"]["extrinsics"], curr_batch["target"]["extrinsics"]], dim=1), index=rel_index).float()
                curr_batch["context"]["extrinsics"] = rel_extrinsics[:, :v_c, ...]
                curr_batch["target"]["extrinsics"] = rel_extrinsics[:, v_c:, ...]
                sampled_views, curr_batch = self.sample(curr_batch)
                sampled_views = sampled_views.clip(0.0, 1.0)
                context_views = batch["context"]["image"]
                target_views = batch["target"]["image"]
                b, v_t, *_ = sampled_views.shape
                
                scene = batch["scene"][0]

                for index, color in zip(target, sampled_views[0]):
                    save_image(color, output_dir / scene / f"color/{index:0>6}.png")
                
                    if return_predictions:
                        samples.append(color)
                        indices.append(index)
                        if (anchors_indices[0][i].item() - index) == 1:
                            samples.append(anchor_views[0, i, ...])
                            indices.append(anchors_indices[0][i].item())
        return samples, indices            
                
    def test_video_autoregressive(self, batch, batch_idx, limit_frames: int | None=None):
        if limit_frames is not None:
            batch = {
                "context": {
                    "image": batch["context"]["image"],
                    "extrinsics": batch["context"]["extrinsics"],
                    "intrinsics": batch["context"]["intrinsics"],
                    "near": batch["context"]["near"],
                    "far": batch["context"]["far"],
                    "index": batch["context"]["index"],
                },
                "target": {
                    "image": batch["target"]["image"][:, :limit_frames, ...],
                    "extrinsics": batch["target"]["extrinsics"][:,  :limit_frames, ...],
                    "intrinsics": batch["target"]["intrinsics"][:,  :limit_frames, ...],
                    "near": batch["target"]["near"][:,  :limit_frames, ...],
                    "far": batch["target"]["far"][:,  :limit_frames, ...],
                    "index": batch["target"]["index"][:,  :limit_frames, ...],
                },
                "scene": batch["scene"]
            }
        if self.global_rank == 0:
            print(
                f"test step {self.step_tracker.get_step()}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()} "
                f"target = {batch['target']['index'].tolist()}"
            )
        b, v_c = batch["context"]["extrinsics"].shape[:2]
        assert b == 1, "Batch Size must be 1 for sampling video"

        output_dir = self.output_dir / "video"
        index = 1 
        batch, rel_index = self.sample_indices(batch, index, False)
        b, v_c, c, h, w = batch["context"]["image"].shape
        b, num_t, c, h, w = batch["target"]["image"].shape
        
        n_anchors = 4
        # Sample anchor views
        initial_batch = {
            "context": batch["context"],
            "target": {
                "image": batch["target"]["image"][:, :n_anchors, ...],
                "extrinsics": batch["target"]["extrinsics"][:, :n_anchors, ...],
                "intrinsics": batch["target"]["intrinsics"][:, :n_anchors, ...],
                "near": batch["target"]["near"][:, :n_anchors, ...],
                "far": batch["target"]["far"][:, :n_anchors, ...],
                "index": batch["target"]["index"][:, :n_anchors],
            },
            "scene": batch["scene"]
        }
        b, v_c, c, h, w = initial_batch["context"]["image"].shape

        rel_index = 0
        rel_extrinsics = absolute_to_relative_camera(torch.concat([initial_batch["context"]["extrinsics"], initial_batch["target"]["extrinsics"]], dim=1), index=rel_index).float()
        initial_batch["context"]["extrinsics"] = rel_extrinsics[:, :v_c, ...]
        initial_batch["target"]["extrinsics"] = rel_extrinsics[:, v_c:, ...]

        initial_views, initial_batch = self.sample(initial_batch)
        initial_views = initial_views.clip(0.0, 1.0)
        context_views = batch["context"]["image"]
        target_views = batch["target"]["image"]
        b, v_t, *_ = initial_views.shape
        b, v_c, *_ = context_views.shape
        
        scene = batch["scene"][0]
        # context_index_str = "_".join(map(str, sorted(batch["context"]["index"][i].tolist())))
        # print(context_index_str)

        for index, color in zip(initial_batch["target"]["index"][0], initial_views[0]):
            save_image(color, output_dir / scene / f"color/{index:0>6}.png")
        
        remaining_target_indices = batch["target"]["index"][0].tolist()
        # remaining_target_indices_idx = [i for i in range(len(remaining_target_indices))]
        for idx in initial_batch["target"]["index"][0]:
            remaining_target_indices.pop(remaining_target_indices.index(idx.item()))
            # remaining_target_indices_idx.pop(remaining_target_indices.index(idx.item()))
            
        n_iterations = (len(remaining_target_indices) + 1) // 3

        start = n_anchors
        last_batch = {
            "image": initial_views[:, -1:, ...],
            "extrinsics": initial_batch["target"]["extrinsics"][:, -1:, ...],
            "intrinsics": initial_batch["target"]["intrinsics"][:, -1:, ...],
            "near": initial_batch["target"]["near"][:, -1:, ...],
            "far": initial_batch["target"]["far"][:, -1:, ...],
            "index": initial_batch["target"]["index"][:, -1:],
        }
        

        for i in range(1, n_iterations+1):
            
            end = start + 3
            context_image = torch.concat([batch["context"]["image"], last_batch["image"]], dim=1)
            context_extrinsics = torch.concat([batch["context"]["extrinsics"], last_batch["extrinsics"]], dim=1)
            context_intrinsics = torch.concat([batch["context"]["intrinsics"], last_batch["intrinsics"]], dim=1)
            context_near = torch.concat([batch["context"]["near"], last_batch["near"]], dim=1)
            context_far = torch.concat([batch["context"]["far"], last_batch["far"]], dim=1)
            context_index = torch.concat([batch["context"]["index"], last_batch["index"]], dim=1)

            
            b, v_c, *_ = context_image.shape
            print(start, end)
            curr_batch = {
                "context": {
                    "image": context_image,
                    "extrinsics": context_extrinsics,
                    "intrinsics": context_intrinsics,
                    "near": context_near,
                    "far": context_far,
                    "index": context_index,
                },
                "target": {
                    "image": batch["target"]["image"][:, start:end, ...],
                    "extrinsics": batch["target"]["extrinsics"][:,  start:end, ...],
                    "intrinsics": batch["target"]["intrinsics"][:,  start:end, ...],
                    "near": batch["target"]["near"][:,  start:end, ...],
                    "far": batch["target"]["far"][:,  start:end, ...],
                    "index": batch["target"]["index"][:,  start:end, ...],
                },
                "scene": batch["scene"]
            }

            print("Context Indices: ", curr_batch["context"]["index"])
            print("Target Indices: ", curr_batch["target"]["index"])

            b, v_t, *_ = curr_batch["target"]["image"].shape
            rel_index = 1
            rel_extrinsics = absolute_to_relative_camera(torch.concat([curr_batch["context"]["extrinsics"], curr_batch["target"]["extrinsics"]], dim=1), index=rel_index).float()
            curr_batch["context"]["extrinsics"] = rel_extrinsics[:, :v_c, ...]
            curr_batch["target"]["extrinsics"] = rel_extrinsics[:, v_c:, ...]
            sampled_views, _ = self.sample(curr_batch)
            sampled_views = sampled_views.clip(0.0, 1.0)
   
            b, v_t, *_ = sampled_views.shape
            
            scene = batch["scene"][0]

            for index, color in zip(curr_batch["target"]["index"][0], sampled_views[0]):
                save_image(color, output_dir / scene / f"color/{index:0>6}.png")

            last_batch = {
                "image": sampled_views[:, -1:, ...],
                "extrinsics": curr_batch["target"]["extrinsics"][:, -1:, ...],
                "intrinsics": curr_batch["target"]["intrinsics"][:, -1:, ...],
                "near": curr_batch["target"]["near"][:, -1:, ...],
                "far": curr_batch["target"]["far"][:, -1:, ...],
                "index": curr_batch["target"]["index"][:, -1:],
            }

            start = end
    
    def test_step(self, batch, batch_idx):
 
        b, v_c = batch["context"]["extrinsics"].shape[:2]
        
        if self.test_cfg.sampling_mode == "autoregressive":
            self.test_video_autoregressive(batch, batch_idx, limit_frames=self.test_cfg.limit_frames)
        elif self.test_cfg.sampling_mode == "anchored":
            self.test_video_anchored(batch, batch_idx, limit_frames=self.test_cfg.limit_frames)
        else:
            raise(Exception(f"Incorrect Mode {self.test_cfg.sampling_mode}"))
            # self.test_batch(batch, batch_idx)

    def on_test_end(self):
        output_dir = self.output_dir / "video"
        scenes = os.listdir(output_dir)
        print(f"Saving Scenes")
        for i, scene in enumerate(tqdm(scenes)):
            directory = output_dir / scene / "color"
            
            original_image_pil = [PIL.Image.open(f"{directory}/{fi}").convert("RGB") for fi in sorted(os.listdir(directory))]
            original_image_pil[0].save(
                output_dir / scene / "sampled.gif",
                save_all=True,
                append_images=original_image_pil[1:],
                duration=5,
                loop=0 # 0 means infinite loop
                )
            original_image_pil = [np.asarray(img) for img in original_image_pil]
            h_fps = ImageSequenceClip(original_image_pil, fps=25)
            l_fps = ImageSequenceClip(original_image_pil, fps=10)
            h_fps.write_videofile(str(output_dir / scene / "sampled_fps_25.mp4"),fps=25)
            l_fps.write_videofile(str(output_dir / scene / "sampled_fps_10.mp4"),fps=10)
            
        return super().on_test_end()
    
    @staticmethod
    def get_optimizer(
        optimizer_cfg: OptimizerCfg,
        params: Iterator[Parameter] | list[Dict[str, Any]],
        lr: float
    ) -> optim.Optimizer:
        return getattr(optim, optimizer_cfg.name)(
            params,
            lr=lr,
            **(optimizer_cfg.kwargs if optimizer_cfg.kwargs is not None else {})       
        )
   
    @staticmethod
    def get_lr_scheduler(
        opt: optim.Optimizer, 
        lr_scheduler_cfg: LRSchedulerCfg
    ) -> optim.lr_scheduler.LRScheduler:
        return getattr(optim.lr_scheduler, lr_scheduler_cfg.name)(
            opt,
            **(lr_scheduler_cfg.kwargs if lr_scheduler_cfg.kwargs is not None else {})     
        )

    def configure_optimizers(self):
        optimizer = self.get_optimizer(self.optimizer_cfg, self.denoiser.parameters(), self.lr)
        if self.optimizer_cfg.scheduler is not None:
            lr_scheduler_config = {
                "scheduler": self.get_lr_scheduler(optimizer, self.optimizer_cfg.scheduler),
                "frequency": self.optimizer_cfg.scheduler.frequency,
                "interval": self.optimizer_cfg.scheduler.interval
            }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
    
