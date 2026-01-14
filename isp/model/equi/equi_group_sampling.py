import sys
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import e3nn
from e3nn import o3
import healpy as hp
import numpy as np
from functorch.einops import rearrange
import escnn.nn as enn
from escnn import gspaces, group

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "."))


def s2_irreps(lmax):
    return o3.Irreps([(1, (l, 1)) for l in range(lmax + 1)])


def so3_irreps(lmax):
    return o3.Irreps([(2 * l + 1, (l, 1)) for l in range(lmax + 1)])


def flat_wigner(lmax, alpha, beta, gamma):
    return torch.cat(
        [
            (2 * l + 1) ** 0.5 * o3.wigner_D(l, alpha, beta, gamma).flatten(-2)
            for l in range(lmax + 1)
        ],
        dim=-1,
    )


class SO3ToIco(torch.nn.Module):
    def __init__(self, Lmax, Cin, Cout):
        super().__init__()

        # Sample signal at Ico Group elements
        ico_gspace = gspaces.icoOnR3()
        ico_gspace = gspaces.no_base_space(ico_gspace.fibergroup)

        # get ico group elements
        ico_matrices = torch.from_numpy(
            np.float32([g.to("MAT").T for g in ico_gspace.testing_elements])
        )

        alpha, beta, gamma = o3.matrix_to_angles(ico_matrices)
        self.register_buffer("ico_wigners", flat_wigner(Lmax, alpha, beta, gamma))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape B, Cin, sum_{l=0}^Lmax (2*l+1)^2
        return: shape B, Cout*G
        """
        # perform IFFT onto ico group elements
        x_on_ico = torch.einsum("bhcj,ij->bhci", x, self.ico_wigners)

        return x_on_ico


class SO3ToC8(torch.nn.Module):
    def __init__(self, Lmax, Cin, Cout):
        super().__init__()

        # Sample signal at C8 Group elements
        c8_gspace = gspaces.rot2dOnR2(8)
        c8_gspace = gspaces.no_base_space(c8_gspace.fibergroup)

        # get C8 group elements
        c8_matrices = torch.from_numpy(
            np.float32([g.to("MAT").T for g in c8_gspace.testing_elements])
        )
        temp_matrics = torch.zeros(8, 3, 3)
        temp_matrics[:, :2, :2] = c8_matrices
        temp_matrics[:, 2, 2] = 1
        c8_matrices = temp_matrics

        alpha, beta, gamma = o3.matrix_to_angles(c8_matrices)
        self.register_buffer("c8_wigners", flat_wigner(Lmax, alpha, beta, gamma))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape B, Cin, sum_{l=0}^Lmax (2*l+1)^2
        return: shape B, Cout*G
        """
        # perform IFFT onto C8 group elements
        x_on_c8 = torch.einsum("bhcj,ij->bhci", x, self.c8_wigners)

        return x_on_c8


class EquiGroupSamplingIco(nn.Module):
    def __init__(self, lmax=6, f_out=128):
        super().__init__()
        self.lmax = lmax

        self.proj_1 = o3.Linear("1x0e+3x1e", so3_irreps(1), f_in=1, f_out=f_out // 8)
        self.proj_2 = o3.Linear(
            so3_irreps(lmax), so3_irreps(lmax), f_in=f_out // 8, f_out=f_out // 2
        )
        self.so3_act = e3nn.nn.SO3Activation(1, lmax, act=torch.relu_, resolution=8)

        self.m1 = SO3ToIco(Lmax=lmax, Cin=f_out // 2, Cout=f_out // 2)
        self.m2 = SO3ToIco(Lmax=lmax, Cin=f_out, Cout=f_out)

    def forward(self, x, trajectory):
        """
        x: (b, h, c, f)
        trajectory: (b, t, 10)
        """
        traj_irreps = torch.zeros(trajectory.shape[0], trajectory.shape[1], 1, 10)
        traj_irreps[..., 0, 1:10] = trajectory[:, :, :9]
        traj_irreps[..., 0, 0] = trajectory[:, :, -1]
        traj_irreps = traj_irreps.to(x.device)

        traj_irreps = self.proj_1(traj_irreps)
        traj_irreps = self.so3_act(traj_irreps)
        traj_irreps = self.proj_2(traj_irreps)

        traj_irreps = self.m1(traj_irreps)
        x = self.m2(x)

        results = {"global_cond": x, "trajectory": traj_irreps}

        return results


class EquiGroupSamplingC8(nn.Module):
    def __init__(self, lmax=6, f_out=128):
        super().__init__()
        self.lmax = lmax

        self.proj_1 = o3.Linear("1x0e+3x1e", so3_irreps(1), f_in=1, f_out=f_out // 8)
        self.proj_2 = o3.Linear(
            so3_irreps(lmax), so3_irreps(lmax), f_in=f_out // 8, f_out=f_out // 2
        )
        self.so3_act = e3nn.nn.SO3Activation(1, lmax, act=torch.relu_, resolution=8)

        self.m1 = SO3ToC8(Lmax=lmax, Cin=f_out // 2, Cout=f_out // 2)
        self.m2 = SO3ToC8(Lmax=lmax, Cin=f_out, Cout=f_out)

    def forward(self, x, trajectory):
        """
        x: (b, h, c, f)
        trajectory: (b, t, 10)
        """
        traj_irreps = torch.zeros(trajectory.shape[0], trajectory.shape[1], 1, 10)
        traj_irreps[..., 0, 1:10] = trajectory[:, :, :9]
        traj_irreps[..., 0, 0] = trajectory[:, :, -1]
        traj_irreps = traj_irreps.to(x.device)

        traj_irreps = self.proj_1(traj_irreps)
        traj_irreps = self.so3_act(traj_irreps)
        traj_irreps = self.proj_2(traj_irreps)
        traj_irreps = self.m1(traj_irreps)

        x = self.m2(x)

        results = {"global_cond": x, "trajectory": traj_irreps}

        return results
