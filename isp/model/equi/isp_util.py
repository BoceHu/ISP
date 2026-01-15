import torch
import math
import healpy as hp
import numpy as np
from e3nn import o3
import e3nn.nn
import torch.nn as nn


def s2_healpix_grid(rec_level: int = 0, max_beta: float = np.pi / 6):
    """Returns healpix grid up to a max_beta"""
    n_side = 2**rec_level
    m = hp.query_disc(nside=n_side, vec=(0, 0, 1), radius=max_beta)
    beta, alpha = hp.pix2ang(n_side, m)
    alpha = torch.from_numpy(alpha)
    beta = torch.from_numpy(beta)
    return torch.stack((alpha, beta)).float()


def s2_near_identity_grid(
    max_beta: float = math.pi / 8, n_alpha: int = 8, n_beta: int = 3
):
    """
    :return: rings around the north pole
    size of the kernel = n_alpha * n_beta
    """
    beta = torch.arange(1, n_beta + 1) * max_beta / n_beta
    alpha = torch.linspace(0, 2 * math.pi, n_alpha + 1)[:-1]
    a, b = torch.meshgrid(alpha, beta, indexing="ij")
    b = b.flatten()
    a = a.flatten()
    return torch.stack((a, b))


def so3_healpix_grid(rec_level: int = 3):
    """Returns healpix grid over so3
    https://github.com/google-research/google-research/blob/4808a726f4b126ea38d49cdd152a6bb5d42efdf0/implicit_pdf/models.py#L272

    alpha: 0-2pi around Y
    beta: 0-pi around X
    gamma: 0-2pi around Y

    rec_level | num_points | bin width (deg)
    ----------------------------------------
         0    |         72 |    60
         1    |        576 |    30
         2    |       4608 |    15
         3    |      36864 |    7.5
         4    |     294912 |    3.75
         5    |    2359296 |    1.875

    :return: tensor of shape (3,npix)
    """
    n_side = 2**rec_level
    npix = hp.nside2npix(n_side)
    beta, alpha = hp.pix2ang(n_side, torch.arange(npix))
    gamma = torch.linspace(0, 2 * np.pi, 6 * n_side + 1)[:-1]

    alpha = alpha.repeat(len(gamma))
    beta = beta.repeat(len(gamma))
    gamma = torch.repeat_interleave(gamma, npix)
    return torch.stack((alpha, beta, gamma)).float()


def so3_near_identity_grid(
    max_beta=np.pi / 8, max_gamma=2 * np.pi, n_alpha=8, n_beta=3, n_gamma=None
):
    """
    :return: rings of rotations around the identity, all points (rotations) in
    a ring are at the same distance from the identity
    size of the kernel = n_alpha * n_beta * n_gamma
    """
    if n_gamma is None:
        n_gamma = n_alpha  # similar to regular representations
    beta = torch.arange(1, n_beta + 1) * max_beta / n_beta
    alpha = torch.linspace(0, 2 * np.pi, n_alpha)[:-1]
    pre_gamma = torch.linspace(-max_gamma, max_gamma, n_gamma)
    A, B, preC = torch.meshgrid(alpha, beta, pre_gamma, indexing="ij")
    C = preC - A
    A = A.flatten()
    B = B.flatten()
    C = C.flatten()
    return torch.stack((A, B, C))


def s2_irreps(lmax):
    return o3.Irreps([(1, (l, 1)) for l in range(lmax + 1)])


def so3_irreps(lmax):
    return o3.Irreps([(2 * l + 1, (l, 1)) for l in range(lmax + 1)])


def rotate_s2(s2_signal, alpha=0.0, beta=0.0, gamma=0.0):
    """alpha beta gamma in radians"""
    lmax = int(s2_signal.shape[-1] ** 0.5) - 1
    irreps = s2_irreps(lmax)
    alpha = torch.tensor(alpha, dtype=torch.float32)
    beta = torch.tensor(beta, dtype=torch.float32)
    gamma = torch.tensor(gamma, dtype=torch.float32)
    return torch.einsum(
        "ij,...j->...i", irreps.D_from_angles(alpha, beta, gamma), s2_signal
    )


class S2Conv(torch.nn.Module):
    """S2 group convolution which outputs signal over SO(3) irreps

    :f_in: feature dimensionality of input signal
    :f_out: feature dimensionality of output signal
    :lmax: maximum degree of harmonics used to represent input and output signals
           technically, you can have different degrees for input and output, but
           we do not explore that in our work
    :kernel_grid: spatial locations over which the filter is defined (alphas, betas)
                  we find that it is better to parametrize filter in spatial domain
                  and project to harmonics at every forward pass.
    """

    def __init__(self, f_in, f_out, lmax, kernel_grid) -> None:
        super().__init__()
        # self.register_parameter(
        #     "w", torch.nn.Parameter(torch.randn(f_in, f_out, kernel_grid.shape[1]))
        # )  # [f_in, f_out, n_s2_pts]

        weight = torch.zeros((f_in, f_out, kernel_grid.shape[1]), dtype=torch.float32)
        self.w = nn.Parameter(data=weight, requires_grad=True)
        torch.nn.init.normal_(self.w)
        self.register_buffer(
            "Y",
            o3.spherical_harmonics_alpha_beta(
                range(lmax + 1), *kernel_grid, normalization="component"
            ),
        )  # [n_s2_pts, psi]
        self.lin = o3.Linear(
            s2_irreps(lmax),
            so3_irreps(lmax),
            f_in=f_in,
            f_out=f_out,
            internal_weights=False,
        )

    def forward(self, x):
        psi = torch.einsum("ni,xyn->xyi", self.Y, self.w) / self.Y.shape[0] ** 0.5
        return self.lin(x, weight=psi)


def flat_wigner(lmax, alpha, beta, gamma):
    return torch.cat(
        [
            (2 * l + 1) ** 0.5 * o3.wigner_D(l, alpha, beta, gamma).flatten(-2)
            for l in range(lmax + 1)
        ],
        dim=-1,
    )


class SO3Conv(nn.Module):
    """SO3 group convolution

    :f_in: feature dimensionality of input signal
    :f_out: feature dimensionality of output signal
    :lmax: maximum degree of harmonics used to represent input and output signals
           technically, you can have different degrees for input and output, but
           we do not explore that in our work
    :kernel_grid: spatial locations over which the filter is defined (alphas, betas, gammas)
                  we find that it is better to parametrize filter in spatial domain
                  and project to harmonics at every forward pass
    """

    def __init__(
        self,
        f_in: int,
        f_out: int,
        lmax_in: int,
        kernel_grid: tuple,
        lmax_out: int = None,
    ):
        super().__init__()

        # filter weight parametrized over spatial grid on SO3
        # self.register_parameter(
        #     "w", torch.nn.Parameter(torch.randn(f_in, f_out, kernel_grid.shape[1]))
        # )  # [f_in, f_out, n_so3_pts]

        if lmax_out == None:
            lmax_out = lmax_in

        weight = torch.zeros((f_in, f_out, kernel_grid.shape[1]), dtype=torch.float32)
        self.w = nn.Parameter(data=weight, requires_grad=True)
        torch.nn.init.normal_(self.w)

        # wigner D matrices used to project spatial signal to irreps of SO(3)
        self.register_buffer(
            "D", flat_wigner(lmax_in, *kernel_grid)
        )  # [n_so3_pts, sum_l^L (2*l+1)**2]

        # defines group convolution using appropriate irreps
        self.lin = o3.Linear(
            so3_irreps(lmax_in),
            so3_irreps(lmax_out),
            f_in=f_in,
            f_out=f_out,
            internal_weights=False,
        )

    def forward(self, x):
        """Perform SO3 group convolution to produce signal over irreps of SO(3).
        First project filter into fourier domain then perform convolution

        :x: tensor of shape (B, f_in, sum_l^L (2*l+1)**2), signal over SO3 irreps
        :return: tensor of shape (B, f_out, sum_l^L (2*l+1)**2)
        """
        psi = torch.einsum("ni,xyn->xyi", self.D, self.w) / self.D.shape[0] ** 0.5
        return self.lin(x, weight=psi)


class SO3ToS2Conv(nn.Module):
    """SO3 to S2 Linear Layer"""

    def __init__(self, f_in: int, f_out: int, lmax_in: int, lmax_out: int, kernel_grid):
        super().__init__()
        self.lin = o3.Linear(
            so3_irreps(lmax_in),
            s2_irreps(lmax_out),
            f_in=f_in,
            f_out=f_out,
        )

    def forward(self, x):
        return self.lin(x)
