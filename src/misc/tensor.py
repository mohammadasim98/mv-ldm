import torch
from torch import Tensor


def cumsum(
    x: Tensor, 
    dim: int = 0
) -> Tensor:
    """In contrast to :meth:`torch.cumsum`, prepends the output with zero"""
    size = x.size()[:dim] + (x.size(dim) + 1, ) + x.size()[dim + 1:]
    out = x.new_empty(size)

    out.narrow(dim, 0, 1).zero_()
    torch.cumsum(x, dim=dim, out=out.narrow(dim, 1, x.size(dim)))
    return out


def right_broadcasting_gather_(
    input: Tensor,
    dim: int,
    index: Tensor
) -> Tensor:
    """
    Accepts input and index shapes that are broadcastable to the right (unlike PyTorch to the left)!
    """
    input_dims = input.shape
    dims = index.shape
    nrdim = index.ndim - 1
    if dim < 0:
        dim = input.ndim + dim
    max_dim = max(dims[:dim] + dims[dim:])
    dummy_idx = torch.arange(max_dim, device=index.device)
    dummy_zero = dummy_idx[:1]
    return input[
        [(dummy_zero if input_dims[i] == 1 else dummy_idx[:dims[i]].view(*(i*(1,)), -1, *((nrdim - i)*(1,)))).expand(dims) for i in range(dim)] 
        + [index] 
        + [(dummy_zero if input_dims[i] == 1 else dummy_idx[:dims[i]].view(*(i*(1,)), -1, *((nrdim - i)*(1,)))).expand(dims) for i in range(dim+1, index.ndim)]
        + (input.ndim - index.ndim) * [slice(None)]
    ]


def unsqueeze_multi_dims(
    t: Tensor,
    n: int,
    i: int | None = None,
) -> Tensor:
    """
    Arguments:
        t: [d_{0}, ..., d_{n}]
    Returns:
        [d_{0}, ..., d_{i-1}, *(n * (1,)), d_{i}, ..., d_{n}]
    """
    if i is None:
        i = t.ndim
    if i < 0:
        i += t.ndim + 1
        assert i >= 0
    return t[i * (slice(None),) + n * (None,)]


def unsqueeze_as(
    a: Tensor,
    b: Tensor,
    i: int | None = None
) -> Tensor:
    """
    Arguments:
        a: [d_{0}, ..., d_{i}]
        b: [d_{0}, ... d_{n}]
        with i <= n
    Returns:
        a: [d_{0}, ..., d_{i}, *((n-i) * (1,))]
    """
    return unsqueeze_multi_dims(a, b.ndim-a.ndim, i)
