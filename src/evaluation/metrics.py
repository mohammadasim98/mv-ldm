from functools import cache

import torch
from einops import reduce, repeat, rearrange 
from jaxtyping import Float
from lpips import LPIPS
from DISTS_pytorch import DISTS
from skimage.metrics import structural_similarity
from torch import Tensor
import torch.nn.functional as F
from torch import Tensor

from ..geometry.epipolar_lines import project_rays
from ..geometry.projection import get_world_rays, sample_image_grid

@torch.no_grad()
def compute_psnr(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    return -10 * mse.log10()


@cache
def get_dists(device: torch.device) -> DISTS:
    return DISTS().to(device)


@torch.no_grad()
def compute_dists(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    value = get_dists(predicted.device).forward(ground_truth, predicted, require_grad=False)
    if value.ndim == 0:
        return value.unsqueeze(0)
    return value


@cache
def get_lpips(device: torch.device) -> LPIPS:
    return LPIPS(net="vgg").to(device)


@torch.no_grad()
def compute_lpips(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    value = get_lpips(predicted.device).forward(ground_truth, predicted, normalize=True)
    return value[:, 0, 0, 0]


@torch.no_grad()
def compute_ssim(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ssim = [
        structural_similarity(
            gt.detach().cpu().float().numpy(),
            hat.detach().cpu().float().numpy(),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        for gt, hat in zip(ground_truth, predicted)
    ]
    return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)








def generate_image_rays(
        shape: tuple[int, ...],
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
    b, v, *_ = extrinsics.shape
    device, dtype = extrinsics.device, extrinsics.dtype
    xy, _ = sample_image_grid(shape, device=device, dtype=dtype)
    origins, directions = get_world_rays(
        rearrange(xy, "h w xy -> (h w) xy"),
        rearrange(extrinsics, "b v i j -> b v () i j"),
        rearrange(intrinsics, "b v i j -> b v () i j"),
    )
    return repeat(xy, "h w xy -> b v (h w) xy", b=b, v=v), origins, directions


def get_score(
        batch,
        device: torch.device=torch.cuda
    ):
    
    b, v_t, *_ =  batch["target"]["image"].shape
    ref_index = 0
    proj_image = batch["target"]["image"]
    proj_extrinsics = batch["target"]["extrinsics"]
    proj_intrinsics = batch["target"]["intrinsics"]
    proj_near = batch["target"]["near"]
    proj_far = batch["target"]["far"]
    proj_index = batch["target"]["index"]
    
    b, vp, *_ = proj_extrinsics.shape
    
    ref_image = batch["context"]["image"][:, ref_index:ref_index+1, ...]
    ref_extrinsics = batch["context"]["extrinsics"][:, ref_index:ref_index+1, ...]
    ref_intrinsics = batch["context"]["intrinsics"][:, ref_index:ref_index+1, ...]
    ref_near = batch["context"]["near"][:, ref_index:ref_index+1, ...]
    ref_far = batch["context"]["far"][:, ref_index:ref_index+1, ...]
    ref_index = batch["context"]["index"][:, ref_index:ref_index+1]

    *_, h, w = proj_image.shape
    
    b, vr, *_ = ref_extrinsics.shape
    b, vp, *_ = proj_extrinsics.shape

    # Generate the rays that are projected onto other views.
    _, proj_origins, proj_directions = generate_image_rays(
        (h, w), proj_extrinsics, proj_intrinsics
    )

    ref_extrinsics = repeat(ref_extrinsics, "b vr ... -> b vp vr ...", vp=vp)
    ref_intrinsics = repeat(ref_intrinsics, "b vr ... -> b vp vr ...", vp=vp)

    
    # Select the camera extrinsics and intrinsics to project onto. For each context
    # view, this means all other context views in the batch.
    projection = project_rays(
        rearrange(proj_origins, "b vp r xyz -> b vp () r xyz"),
        rearrange(proj_directions, "b vp r xyz -> b vp () r xyz"),
        rearrange(ref_extrinsics, "b vp vr i j -> b vp vr () i j"),
        rearrange(ref_intrinsics, "b vp vr i j -> b vp vr () i j"),
        rearrange(proj_near, "b vp -> b vp () ()"),
        rearrange(proj_far, "b vp -> b vp () ()"),
    )
    # Generate sample points.
    res_scale = rearrange(torch.tensor([w, h]).to(device), "xy -> () () () () () xy")
    s = 32
    sample_depth = (torch.arange(s, device=ref_extrinsics.device) + 0.5) / s
    sample_depth = rearrange(sample_depth, "s -> s ()")
    xy_min = projection["xy_min"].nan_to_num(posinf=0, neginf=0) 
    xy_min = xy_min * projection["overlaps_image"][..., None]
    xy_min = rearrange(xy_min, "b vp vr r xy -> b vp vr r () xy")
    xy_max = projection["xy_max"].nan_to_num(posinf=0, neginf=0) 
    xy_max = xy_max * projection["overlaps_image"][..., None]
    xy_max = rearrange(xy_max, "b vp vr r xy -> b vp vr r () xy")
    xy_sample = xy_min + sample_depth * (xy_max - xy_min)

    xy_sample = rearrange(xy_sample, "b vp vr (h w) s xy -> b vp vr h w s xy", b=b, vp=vp, h=h, w=w)
    print(xy_sample.shape)
    # xy_sample = xy_sample * res_scale
    
    # min_max_mask = (xy_max == xy_min)
    # overlap_mask = (min_max_mask[..., 0] == min_max_mask[..., 1])

    # mask = (samples == 0.0).float()
    # samples += mask * torch.randn_like(samples).to(device)
    weights = (1 - sample_depth)
    weights_sum = weights.sum()
    weights = rearrange(weights, "s () -> () () () () () () s")
    proj_noise = reduce(samples * weights, "b vp vr c h w s -> b vp c h w", reduction="sum")
    proj_noise = proj_noise / weights_sum
    overlap_mask = rearrange(projection["overlaps_image"], "b vp c (h w) -> b vp c h w", h=h, w=w)
    
    # Set the non-overlap region to 0
    proj_noise *= overlap_mask
    
    # Add noise to non-overlap region
    proj_noise += ~overlap_mask * torch.randn_like(proj_noise).to(device)

    # Add a small post noise globally
    if perturb:
        proj_noise = (1 - perturb_factor)  * proj_noise + perturb_factor * torch.randn_like(proj_noise).to(device)
    
    return proj_noise, batch