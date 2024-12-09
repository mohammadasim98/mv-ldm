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


@dataclass
class DatasetRE10kCfg(DatasetCfgCommon):
    name: Literal["re10k"]
    root: Path | None
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool


class DatasetRE10k(IterableDataset):
    cfg: DatasetRE10kCfg
    stage: Stage
    view_sampler: ViewSampler
    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetRE10kCfg,
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
 
        if cfg.root is None:
            raise Exception("Root directory of dataset is not defined. Please specify in your argument as dataset.root=<path-to-root-directory>")

        root = cfg.root / self.data_stage
        root_chunks = sorted(
            [path for path in root.iterdir() if path.suffix == ".torch"]
        )
        self.chunks.extend(root_chunks)
        if self.cfg.overfit_to_scene is not None:
            # self.chunks = [self.index[name] for name in self.cfg.overfit_to_scene] * len(self.chunks)
        
            with open(root / "index.json") as f:
                self.map_dict = json.load(f)
            self.chunks = [root / self.map_dict[name] for name in self.cfg.overfit_to_scene]

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __iter__(self):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage in ("train", "val") or self.force_shuffle:
            self.chunks = self.shuffle(self.chunks)

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]

        for chunk_path in self.chunks:
            # Load the chunk.
            chunk = torch.load(chunk_path, weights_only=True)

            if self.cfg.overfit_to_scene is not None:
                item = [x for x in chunk if x["key"] in self.cfg.overfit_to_scene]
                # assert len(item) == len(self.cfg.overfit_to_scene)
                chunk = item

            if self.stage in ("train", "val", "test"):
                chunk = self.shuffle(chunk)

            for example in chunk:
                extrinsics, intrinsics = self.convert_poses(example["cameras"])
                scene = example["key"]
                num_views = extrinsics.shape[0]

                # Skip the example if the field of view is too wide.
                if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                    continue

                try:
                    view_indices = self.view_sampler.sample(scene, num_views)
                except ValueError:
                    # Skip because the example doesn't have enough frames.
                    continue
                
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
                        yield apply_crop_shim(sample, tuple(self.cfg.image_shape))

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

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for root in self.cfg.roots:
                # Load the root's index.
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v) for k, v in index.items()}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    def __len__(self) -> int:
        if isinstance(self.view_sampler, ViewSamplerEvaluation):
            return self.view_sampler.total_samples
        return len(self.index.keys())
