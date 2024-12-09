import os
from pathlib import Path

import torch
from pytorch_lightning import LightningModule
from tabulate import tabulate

from ..misc.image_io import load_image, save_image
from ..visualization.annotation import add_label
from ..visualization.layout import add_border, hcat
from .evaluation_cfg import EvaluationCfg
from .metrics import compute_dists, compute_lpips, compute_psnr, compute_ssim
from torchmetrics.image.fid import FrechetInceptionDistance
from einops import rearrange

class MetricComputer(LightningModule):
    cfg: EvaluationCfg

    def __init__(self, cfg: EvaluationCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.fid = FrechetInceptionDistance(feature=64, normalize=True)

    def on_test_epoch_start(self) -> None:
        self.scores = dict(dists={}, lpips={}, ssim={}, psnr={}, fid={})

    def test_step(self, batch, batch_idx):
        scene = batch["scene"]
        b, cv, _, _, _ = batch["context"]["image"].shape
        _, v, _, _, _ = batch["target"]["image"].shape

        # Skip scenes.
        # for method in self.cfg.methods:
        #     print(method.path / scene)
        #     if not (method.path / scene).exists():
        #         # print(method.path)
        #         print(f'Skipping "{method.path / scene}".')
        #         return


        # Load the images.
        all_images = {}
        try:
            for method in self.cfg.methods:

                
                images = [torch.stack([
                    load_image(method.path / scene[j] / f"color/{index.item():0>6}.png")
                    for index in batch["target"]["index"][j]
                ]) for j in range(b)]
                all_images[method.key] = torch.stack(images).to(self.device)
        except:
            print(f'Skipping "{scene}".')
            # exit()
            return

        # Compute metrics.
        all_metrics = {}
        
        for key, sampled_images in all_images.items():
            for j in range(b):
                rgb_gt = batch["target"]["image"][j]
                images = sampled_images[j]

                self.fid.update(rgb_gt, real=True)
                self.fid.update(images, real=False)
                fid = self.fid.compute()
                self.fid.reset()
                dists = compute_dists(rgb_gt, images).mean()
                lpips = compute_lpips(rgb_gt, images).mean()
                ssim = compute_ssim(rgb_gt, images).mean()
                psnr = compute_psnr(rgb_gt, images).mean()
                for metric, score in zip(
                    ("dists", "lpips", "ssim", "psnr", "fid"),
                    (dists, lpips, ssim, psnr, fid)
                ):
                    if scene[j] not in self.scores[metric]:
                        self.scores[metric][scene[j]] = {}
                    self.scores[metric][scene[j]][key] = score.item()
                all_metrics = {
                    **all_metrics,
                    f"dists_{key}": dists,
                    f"lpips_{key}": lpips,
                    f"ssim_{key}": ssim,
                    f"psnr_{key}": psnr,
                    f"fid_{key}": fid,
                }
        self.log_dict(all_metrics)
        self.print_preview_metrics(all_metrics)

        # Skip the rest if no side-by-side is needed.
        if self.cfg.side_by_side_path is None:
            return

        # Create side-by-side.
        scene_key = f"{batch_idx:0>6}_{scene}"
        for i in range(v):
            true_index = batch["target"]["index"][0, i]
            row = [add_label(batch["target"]["image"][0, i], "Ground Truth")]
            for method in self.cfg.methods:
                image = all_images[method.key][i]
                image = add_label(image, method.name)
                row.append(image)
            start_frame = batch["target"]["index"][0, 0]
            end_frame = batch["target"]["index"][0, -1]
            label = f"Scene {batch['scene'][0]} (frames {start_frame} to {end_frame})"
            row = add_border(add_label(hcat(*row), label, font_size=16))
            save_image(
                row,
                self.cfg.side_by_side_path / scene_key / f"{true_index:0>6}.png",
            )

        # Create an animation.
        if self.cfg.animate_side_by_side:
            (self.cfg.side_by_side_path / "videos").mkdir(exist_ok=True, parents=True)
            command = (
                'ffmpeg -y -framerate 30 -pattern_type glob -i "*.png"  -c:v libx264 '
                '-pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"'
            )
            os.system(
                f"cd {self.cfg.side_by_side_path / scene_key } && {command} "
                f"{Path.cwd()}/{self.cfg.side_by_side_path}/videos/{scene_key}.mp4"
            )

    def print_preview_metrics(self, metrics: dict[str, float]) -> None:
        if getattr(self, "running_metrics", None) is None:
            self.running_metrics = metrics
            self.running_metric_steps = 1
        else:
            s = self.running_metric_steps
            self.running_metrics = {
                k: ((s * v) + metrics[k]) / (s + 1)
                for k, v in self.running_metrics.items()
            }
            self.running_metric_steps += 1

        table = []
        for method in self.cfg.methods:
            row = [
                f"{self.running_metrics[f'{metric}_{method.key}']:.3f}"
                for metric in ("psnr", "lpips", "dists", "ssim", "fid")
            ]
            table.append((method.key, *row))

        table = tabulate(table, ["Method", "PSNR (dB)", "LPIPS", "DISTS", "SSIM", "FID"])
        print(table)
