import json
from dataclasses import asdict, dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset
import json 
from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.random_transform_shim import apply_random_transform_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler, ViewSamplerEvaluation
from torch.utils.data import Dataset

@dataclass
class DatasetRE10kV2Cfg(DatasetCfgCommon):
    name: Literal["re10k_non_iter"]
    root: Path | None
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool


class DatasetRE10kV2(Dataset):
    cfg: DatasetRE10kV2Cfg
    stage: Stage
    view_sampler: ViewSampler
    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetRE10kV2Cfg,
        stage: Stage,
        view_sampler: ViewSampler,
        force_shuffle: bool = False
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        self.force_shuffle = force_shuffle

        # Collect chunks.
        self.chunks = []
        with open(view_sampler.index_path) as f:
            indices = json.load(f)
        if cfg.root is None:
            raise Exception("Root directory of dataset is not defined. Please specify in your argument as dataset.root=<path-to-root-directory>")

        root = cfg.root / "test"
        root_chunks = sorted(
            [path for path in root.iterdir() if path.suffix == ".torch"]
        )
        self.chunks.extend(root_chunks)
        self.overfit_to_scene = list(indices.keys())
        
        with open(root / "index.json") as f:
            self.map_dict = json.load(f)
        self.chunks = [root / self.map_dict[name] for name in self.overfit_to_scene]
        
    def __getitem__(self, idx):
        


        chunk = torch.load(self.chunks[idx], weights_only=True)

        item = [x for x in chunk if x["key"] == self.overfit_to_scene[idx]]
        example = item[0]
        extrinsics, intrinsics = self.convert_poses(example["cameras"])
        scene = example["key"]
        num_views = extrinsics.shape[0]


        view_indices = self.view_sampler.sample(scene, num_views)

        
        for view_index in view_indices:
            sample = {"scene": scene}

            # Resize the world to make the baseline 1.
            context_extrinsics = extrinsics[view_index.context]
            if context_extrinsics.shape[0] == 2 and self.cfg.make_baseline_1:
                a, b = context_extrinsics[:, :3, 3]
                scale = (a - b).norm()
                if scale < self.cfg.baseline_epsilon:
                    print(
                        f"Skipped {scene} because of insufficient baseline "
                        f"{scale:.6f}"
                    )
                    continue
                extrinsics[:, :3, 3] /= scale
            else:
                scale = 1
            
            for view_type, indices in asdict(view_index).items():
                if indices is None:
                    continue
                # Load the images.
                images = [
                    example["images"][index.item()] for index in indices
                ]
                images = self.convert_images(images)

                # Skip the example if the images don't have the right shape.
                image_invalid = images.shape[1:] != (3, 360, 640)
                if image_invalid:
                    print(
                        f"Skipped bad example {scene}. "
                        f"{view_type.capitalize()} shape was {images.shape}."
                    )
                    break

                sample[view_type] = {
                    "extrinsics": extrinsics[indices],
                    "intrinsics": intrinsics[indices],
                    "image": images,
                    "near": self.get_bound("near", len(indices)) / scale,
                    "far": self.get_bound("far", len(indices)) / scale,
                    "index": indices
                }
            else:
                # This will only be executed if the inner for loop is exited normally
                if self.stage == "train" and self.cfg.augment:
                    sample = apply_augmentation_shim(sample)
                if self.stage in ["train", "val"]   and self.cfg.random_transform_extrinsics:
                    sample = apply_random_transform_shim(sample)
                return apply_crop_shim(sample, tuple(self.cfg.image_shape))

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics
    
    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)



    def __len__(self) -> int:

        return len(self.overfit_to_scene)
