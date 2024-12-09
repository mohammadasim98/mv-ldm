
import torch
from math import pi, sqrt, log, exp
from torch.distributions import Distribution, constraints, Normal, MultivariateNormal
from jaxtyping import Float
from torch import Tensor
from typing import Literal

def axis_angle_to_quaternion(
    axis_angle: Float[Tensor, "*#batch 3"]
) -> Float[Tensor, "*#batch 4"]:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def axis_angle_to_matrix(
    axis_angle: Float[Tensor, "*#batch 3"]
) -> Float[Tensor, "*#batch 3 3"]:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def _axis_angle_rotation(
    axis: Literal["X", "Y", "Z"], 
    angle: Float[Tensor, "*#batch"]
) -> Float[Tensor, "*#batch 3 3"]:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(
    euler_angles: Float[Tensor, "*#batch 3"], 
    convention: str
) -> Float[Tensor, "*#batch 3 3"]:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def quaternion_to_axis_angle(
    quaternions: Float[Tensor, "*#batch 4"]
) -> Float[Tensor, "*#batch 3"]:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def quaternion_to_matrix(
    quaternions: Float[Tensor, "*#batch 4"]
) -> Float[Tensor, "*#batch 3 3"]:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def standardize_quaternion(
    quaternions: Float[Tensor, "*#batch 4"]
) -> Float[Tensor, "*#batch 4"]:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def matrix_to_axis_angle(
    matrix: Float[Tensor, "*#batch 3 3"]
) -> Float[Tensor, "*#batch 3"]:
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def _angle_from_tan(
    axis: Literal["X", "Y", "Z"], 
    other_axis: Literal["X", "Y", "Z"], 
    data: Float[Tensor, "*#batch 3 3"], 
    horizontal: bool, 
    tait_bryan: bool
) -> Float[Tensor, "*#batch"]:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _index_from_letter(
    letter: Literal["X", "Y", "Z"]
) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")


def matrix_to_euler_angles(
    matrix: Float[Tensor, "*#batch 3 3"], 
    convention: str
) -> Float[Tensor, "*#batch 3"]:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)


def _sqrt_positive_part(
    x: Float[Tensor, "*#batch"]
) -> Float[Tensor, "*#batch"]:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(
    matrix: Float[Tensor, "*#batch 3 3"]
) -> Float[Tensor, "*#batch 4"]:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)

# Matrix and vector operators
def hat(
    v: Float[Tensor, "*#batch 3"]
) -> Float[Tensor, "*#batch 3 3"]:
    """
    Compute the Hat operator [1] of a batch of 3D vectors.

    Args:
        v: Batch of vectors of shape `(minibatch , 3)`.

    Returns:
        Batch of skew-symmetric matrices of shape
        `(minibatch, 3 , 3)` where each matrix is of the form:
            `[    0  -v_z   v_y ]
             [  v_z     0  -v_x ]
             [ -v_y   v_x     0 ]`

    Raises:
        ValueError if `v` is of incorrect shape.

    [1] https://en.wikipedia.org/wiki/Hat_operator
    """
    h = torch.zeros(v.shape[:-1] + (3, 3), dtype=v.dtype, device=v.device)

    x, y, z = v.unbind(1)

    h[..., 0, 1] = -z
    h[..., 0, 2] = y
    h[..., 1, 0] = z
    h[..., 1, 2] = -x
    h[..., 2, 0] = -y
    h[..., 2, 1] = x

    return h


# Logarithmic and exponential mappings
def so3_log_map(
    R: Float[Tensor, "*#batch 3 3"]
) -> Float[Tensor, "*#batch 3"]:
    """
    Convert a batch of 3x3 rotation matrices `R`
    to a batch of 3-dimensional matrix logarithms of rotation matrices
    The conversion has a singularity around `(R=I)`.
    """
    return matrix_to_axis_angle(R)


def so3_exp_map(
    log_rot: Float[Tensor, "*#batch 3"]
) -> Float[Tensor, "*#batch 3 3"]:
    return axis_angle_to_matrix(log_rot)


def so3_lerp(
    rot_a: Float[Tensor, "*#batch 3 3"], 
    rot_b: Float[Tensor, "*#batch 3 3"], 
    weight: Float[Tensor, "*#batch"]
) -> Float[Tensor, "*#batch 3 3"]:
    """Weighted interpolation between rot_a and rot_b"""
    rot_full = rot_a.transpose(-1, -2) @ rot_b
    angle_full = matrix_to_axis_angle(rot_full)
    angle_inter = weight * angle_full
    rot_inter = axis_angle_to_matrix(angle_inter)
    return rot_a @ rot_inter


def so3_scale(
    rot_mat: Float[Tensor, "*#batch 3 3"], 
    scalars: Float[Tensor, "*#batch"]
) -> Float[Tensor, "*#batch 3 3"]:
    """Scale the magnitude of a rotation matrix, geodesic flow from I to R by scalars amount
    e.g. a 45 degree rotation scaled by a factor of 2 gives a 90 degree rotation.

    This is the same as taking matrix powers, but pytorch only supports integer exponents

    So instead, we take advantage of the properties of rotation matrices
    to calculate logarithms easily and multiply instead.
    """
    logs = so3_log_map(rot_mat)
    scaled_logs = logs * scalars.unsqueeze(-1)
    out = so3_exp_map(scaled_logs)
    return out.real

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
