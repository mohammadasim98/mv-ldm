import os
from pathlib import Path

from pytorch_lightning import LightningModule

from ..misc.image_io import save_image


class GTSaver(LightningModule):

    def __init__(self, output_path: Path, save_context: bool = False) -> None:
        super(GTSaver, self).__init__()
        self.output_path = output_path
        self.save_context = save_context

    def test_step(self, batch, batch_idx):


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
                "image": batch["target"]["image"][:, :80, ...],
                "extrinsics": batch["target"]["extrinsics"][:,  :80, ...],
                "intrinsics": batch["target"]["intrinsics"][:,  :80, ...],
                "near": batch["target"]["near"][:,  :80, ...],
                "far": batch["target"]["far"][:,  :80, ...],
                "index": batch["target"]["index"][:,  :80, ...],
            },
            "scene": batch["scene"]
        }
        

        b, cv, _, _, _ = batch["context"]["image"].shape
        _, v, _, _, _ = batch["target"]["image"].shape

        # if len(batch["target"]["index"][0]) <= 150:
        #     scene = batch["scene"][0]
        #     _len = len(batch["target"]["index"][0])
        #     print(f"Skipping scene {scene} with len {_len}")
        #     return
        
        cv = 1

        for j in range(b):
            scene = batch["scene"][j]
            target_dir_path = self.output_path / scene / "color"
            for i in range(v):
                true_index = batch["target"]["index"][j, i]
                gt_image = batch["target"]["image"][j, i]
                save_image(
                    gt_image,
                    target_dir_path / f"{true_index:0>6}.png"
                )
        # for j in range(b):
        #     scene = batch["scene"][j]
        #     context_dir_path = self.output_path / scene / "context"
        #     true_index = batch["context"]["index"][j, 0]
        #     gt_image = batch["context"]["image"][j, 0]
        #     save_image(
        #         gt_image,
        #         context_dir_path / f"{true_index:0>6}.png"
        #     )
        # # Save the second context view as the target view  
        # for j in range(b):
        #     scene = batch["scene"][j]
        #     context_dir_path = self.output_path / scene / "context"
        #     true_index = batch["context"]["index"][j, 1]
        #     gt_image = batch["context"]["image"][j, 1]
        #     save_image(
        #         gt_image,
        #         context_dir_path / f"{true_index:0>6}.png"
        #     )
