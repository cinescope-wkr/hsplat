import torch
from torch import Tensor

from ._backend import _C

# simple CUDA test function
def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _C.add(a, b)

### BEGIN FAST CGH FUNCTIONS

def cgh_gaussians_naive(
    fx             : torch.Tensor,
    fy             : torch.Tensor,
    fz             : torch.Tensor,
    wvl            : float,
    R              : torch.Tensor,
    A_inv_T        : torch.Tensor,
    A_det          : torch.Tensor,
    c              : torch.Tensor,
    du             : torch.Tensor,
    local_AS_shift : torch.Tensor,
    opacity        : torch.Tensor,
    color          : torch.Tensor
) -> torch.Tensor:
    return _C.cgh_gaussians_naive(
        fx, 
        fy, 
        fz, 
        wvl, 
        R, 
        A_inv_T, 
        A_det, 
        c, 
        du, 
        local_AS_shift,
        opacity,
        color
    )