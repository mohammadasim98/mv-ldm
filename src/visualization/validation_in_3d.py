import torch
from jaxtyping import Float, Shaped
from torch import Tensor

from ..visualization.drawing.cameras import draw_cameras


def pad(images: list[Shaped[Tensor, "..."]]) -> list[Shaped[Tensor, "..."]]:
    shapes = torch.stack([torch.tensor(x.shape) for x in images])
    padded_shape = shapes.max(dim=0)[0]
    results = [
        torch.ones(padded_shape.tolist(), dtype=x.dtype, device=x.device)
        for x in images
    ]
    for image, result in zip(images, results):
        slices = [slice(0, x) for x in image.shape]
        result[slices] = image[slices]
    return results


def render_cameras(batch: dict, resolution: int) -> Float[Tensor, "3 3 height width"]:
    # Define colors for context and target views.
    num_context_views = batch["context"]["extrinsics"].shape[1]
    num_target_views = batch["target"]["extrinsics"].shape[1]
    color = torch.ones(
        (num_target_views + num_context_views, 3),
        dtype=torch.float32,
        device=batch["target"]["extrinsics"].device,
    )
    color[num_context_views:, 1:] = 0

    return draw_cameras(
        resolution,
        torch.cat(
            (batch["context"]["extrinsics"][0], batch["target"]["extrinsics"][0])
        ),
        torch.cat(
            (batch["context"]["intrinsics"][0], batch["target"]["intrinsics"][0])
        ),
        color,
        torch.cat((batch["context"]["near"][0], batch["target"]["near"][0])),
        torch.cat((batch["context"]["far"][0], batch["target"]["far"][0])),
        frustum_scale=0.05
    )
