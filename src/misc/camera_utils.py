

import torch
from jaxtyping import Float
from torch import Tensor

def absolute_to_relative_camera(
    tform: Float[Tensor, "batch v 4 4"],
    index: int
):

    _, v, *_ = tform.shape
    ref_tform = tform[:, index:index+1, ...]
    ref_tform = ref_tform.expand(-1, v, -1, -1) 

    # tform = tform.inverse() @ ref_tform
    tform = torch.linalg.inv(ref_tform) @ tform
    
    
    # new_tform = torch.zeros_like(ref_tform)
    # new_tform[..., :3, :3] = ref_tform[..., :3, :3].transpose(-1, -2)
    # new_tform[..., :3, 3] = -ref_tform[..., :3, 3]
    
    # tform = new_tform @ tform
    
    
    return tform
