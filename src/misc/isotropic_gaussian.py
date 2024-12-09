
import torch
from math import pi, sqrt, log, exp
from .so3 import so3_exp_map
from .rotation_conversions import matrix_to_axis_angle
from torch.distributions import Distribution, constraints, Normal, MultivariateNormal

class IsotropicGaussianSO3(Distribution):
    arg_constraints = {'eps': constraints.positive}

    def __init__(self, eps: torch.Tensor, mean: torch.Tensor = torch.eye(3)):
        self.eps = eps # (B,)
        self._mean = mean.to(self.eps) # (3, 3)
        self._mean_inv = self._mean.transpose(-1, -2)  # orthonormal so inverse = Transpose
        pdf_sample_locs = pi * torch.linspace(0, 1.0, 1000) ** 3.0  # Pack more samples near 0
        pdf_sample_locs = pdf_sample_locs.to(self.eps).unsqueeze(-1) # (1000, 1)
        # As we're sampling using axis-angle form
        # and need to account for the change in density
        # Scale by 1-cos(t)/pi for sampling
        with torch.no_grad():
            pdf_sample_vals = self._eps_ft(pdf_sample_locs) * ((1 - pdf_sample_locs.cos()) / pi)
        # Set to 0.0, otherwise there's a divide by 0 here
        pdf_sample_vals[(pdf_sample_locs == 0).expand_as(pdf_sample_vals)] = 0.0

        # Trapezoidal integration
        pdf_val_sums = pdf_sample_vals[:-1, ...] + pdf_sample_vals[1:, ...]

        pdf_loc_diffs = torch.diff(pdf_sample_locs, dim=0)
        self.trap = (pdf_loc_diffs * pdf_val_sums / 2).cumsum(dim=0)
        self.trap = self.trap/self.trap[-1,None]
        self.trap_loc = pdf_sample_locs[1:]
        self.trap = self.trap.permute(1, 0)
        self.trap_loc = self.trap_loc.permute(1, 0)
        self.trap_loc = self.trap_loc.expand(self.eps.shape[0], -1)
        super().__init__()

    def sample(self, sample_shape=torch.Size(), generator: torch.Generator | None = None):
       
        # Consider axis-angle form.
        axes = torch.randn((*self.eps.shape, *sample_shape, 3)).to(self.eps)
        axes = axes / axes.norm(dim=-1, keepdim=True)
        
        # Inverse transform sampling based on numerical approximation of CDF
        unif = torch.rand((*self.eps.shape, *sample_shape), device=self.trap.device, generator=generator)
        trap = self.trap[..., None]
        idx_1 = (trap <= unif[:,  None, ...]).sum(dim=1)
        idx_0 = torch.clamp(idx_1 - 1, min=0)
        trap_start = torch.gather(self.trap, 1, idx_0)
        trap_end = torch.gather(self.trap, 1, idx_1)
        trap_diff = torch.clamp((trap_end - trap_start), min=1e-6)
        weight = torch.clamp(((unif - trap_start) / trap_diff), 0, 1)


        angle_start = torch.gather(self.trap_loc, 1, idx_0)
        angle_end = torch.gather(self.trap_loc, 1, idx_1)

        angles = torch.lerp(angle_start, angle_end, weight)[..., None]

        axes = axes * angles

        R = so3_exp_map(axes.reshape(-1, 3))
        R = R.reshape(*self.eps.shape, *sample_shape, 3, 3)
        out = self._mean @ R

        return out

    def _eps_ft(self, t: torch.Tensor) -> torch.Tensor:

        """Sampling from PDF described by equation 10 and 11 in the appendix

        """
        var_d = self.eps.double()**2
        t_d = t.double()
        vals = sqrt(pi) * var_d ** (-3 / 2) * torch.exp(var_d / 4) * torch.exp(-((t_d / 2) ** 2) / var_d) \
               * (t_d - torch.exp((-pi ** 2) / var_d)
                  * ((t_d - 2 * pi) * torch.exp(pi * t_d / var_d) + (
                            t_d + 2 * pi) * torch.exp(-pi * t_d / var_d))
                  ) / (2 * torch.sin(t_d / 2))
        vals[vals.isinf()] = 0.0
        vals[vals.isnan()] = 0.0

        # using the value of the limit t -> 0 to fix nans at 0
        t_big, _ = torch.broadcast_tensors(t_d, var_d)
        # Just trust me on this...
        # This doesn't fix all nans as a lot are still too big to flit in float32 here
        vals[t_big == 0] = sqrt(pi) * (var_d * torch.exp(2 * pi ** 2 / var_d)
                                       - 2 * var_d * torch.exp(pi ** 2 / var_d)
                                       + 4 * pi ** 2 * var_d * torch.exp(pi ** 2 / var_d)
                                       ) * torch.exp(var_d / 4 - (2 * pi ** 2) / var_d) / var_d ** (5 / 2)
        return vals.float()

    def log_prob(self, rotations):
        skew_vec = matrix_to_axis_angle(rotations)
        angles = skew_vec.norm(p=2, dim=-1, keepdim=True)
        probs = self._eps_ft(angles)
        return probs.log()

    @property
    def mean(self):
        return self._mean
