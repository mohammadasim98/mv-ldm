import torch
from jaxtyping import Float
from torch import Tensor
from einops import rearrange, repeat

from ..types import AnyExample, AnyViews
from ...misc.rotation_utils import IsotropicGaussianSO3

def random_transform_extrinsics(
    extrinsics: Float[Tensor, "view 4 4"],
    generator: torch.Generator | None = None,
) ->  Float[Tensor, "view 4 4"]:    

    v, *_ = extrinsics.shape
    eps = torch.ones((1,)).to(extrinsics.device)
    random_rot = IsotropicGaussianSO3(eps).sample((1, )).squeeze(0)
    
    random_trans = torch.randn((1, 3, 1), generator=generator).to(extrinsics.device)
    
    
    rot = extrinsics[..., :3, :3]
    trans = extrinsics[..., :3, 3].unsqueeze(-1)
    
    new_trans = random_trans + trans
    new_rot = random_rot @ rot

    new_extrinsics = torch.concat([new_rot, new_trans], dim=-1)
    last_row = torch.tensor([0.0, 0.0, 0.0, 1.0])
    last_row = repeat(last_row, "d -> v () d", v=v)
    
    
    new_extrinsics = torch.concat([new_extrinsics, last_row], dim=-2)
    return new_extrinsics


def apply_random_transform_shim(
    example: AnyExample,
    generator: torch.Generator | None = None,
) -> AnyExample:
    """Randomly augment the training images."""
    # Do not augment with 50% chance.
    # if torch.rand(tuple(), generator=generator) < 0.5:
    #     return example
    extrinsics = []
    shapes = []
    for view in ("context", "target"):
        if view in example:
            extrinsics.append(example[view]["extrinsics"])
            shapes.append(example[view]["extrinsics"].shape)
    if len(extrinsics) == 0:
        return example       
    
    extrinsics = torch.concat(extrinsics, dim=0)    
    
    modified_extrinsics = random_transform_extrinsics(extrinsics)

    extrinsics = []
    start = 0
    for shape in shapes:
        end = start + shape[0]
        extrinsics.append(modified_extrinsics[start:end])
        start = end
    for view in ("context", "target"):
        if view in example:
            example[view]["extrinsics"] = extrinsics.pop(0)

    return example


    