import sys
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import escnn
import e3nn
from e3nn import o3
import healpy as hp
import numpy as np
from einops import rearrange

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

from isp_equi_img_encoder import EquiResnet18, EquiResnet34, EquiResnet50, EquiResnet1x1
from isp_img_encoder import Resnet18, Resnet34, Resnet50
from isp_util import (
    s2_healpix_grid,
    s2_near_identity_grid,
    s2_irreps,
    so3_irreps,
    so3_near_identity_grid,
    S2Conv,
    SO3Conv,
    rotate_s2,
)


class ImageEncoder(nn.Module):
    def __init__(
        self,
        encoder: str = "equiresnet50",
        out_fdim: int = 512,
        out_shape=(7, 7),
        N=8,
        obs_channel=3,
        initialize=True,
    ):
        super().__init__()
        assert out_shape[0] == out_shape[1] == 7, "Only support 9x9 output shape"
        assert encoder in [
            "equiresnet18",
            "equiresnet34",
            "equiresnet50",
            "resnet18",
            "resnet34",
            "resnet50",
            "equi1x1",
        ], "Only support equiresnet18, 34, 50 and resnet18, 34, 50, and equi1x1"

        self.output_shape = (out_fdim, out_shape[0], out_shape[1])

        encoder_factory = {
            "equiresnet18": EquiResnet18,
            "equiresnet34": EquiResnet34,
            "equiresnet50": EquiResnet50,
            "resnet18": Resnet18,
            "resnet34": Resnet34,
            "resnet50": Resnet50,
            "equi1x1": EquiResnet1x1,
        }
        self.layers = encoder_factory[encoder](
            obs_channel=obs_channel, N=N, initialize=initialize, out_fdim=out_fdim
        )

    def forward(self, img):
        out = self.layers(img)
        # if type(out) is not torch.Tensor:
        #     return out.tensor

        return out


class Image2SphereProjector(nn.Module):
    """Define orthographic projection from image space to half of sphere, returning
    coefficients of spherical harmonics

    :fmap_shape: shape of incoming feature map (channels, height, width)
    :lmax: maximum degree of harmonics
    :coverage: fraction of feature map that is projected onto sphere
    :sigma: stdev of gaussians used to sample points in image space
    :max_beta: maximum azimuth angle projected onto sphere (np.pi/2 corresponds to half sphere)
    :taper_beta: if less than max_beta, taper magnitude of projected features beyond this angle
    :rec_level: recursion level of healpy grid where points are projected
    """

    def __init__(
        self,
        fmap_shape,
        lmax: int,
        coverage: float = 1,
        sigma: float = 0.2,
        max_beta: float = np.radians(90),
        taper_beta: float = np.radians(90),
        rec_level: int = 2,
    ):
        super().__init__()
        self.lmax = lmax

        # determine sampling locations for orthographic projection
        self.kernel_grid = s2_healpix_grid(max_beta=max_beta, rec_level=rec_level)
        self.xyz = o3.angles_to_xyz(*self.kernel_grid)

        # orthographic projection
        max_radius = torch.linalg.norm(self.xyz[:, [0, 2]], dim=1).max()
        sample_x = coverage * self.xyz[:, 2] / max_radius  # range -1 to 1
        sample_y = coverage * self.xyz[:, 0] / max_radius

        gridx, gridy = torch.meshgrid(
            2 * [torch.linspace(-1, 1, fmap_shape[1])], indexing="ij"
        )
        scale = 1 / np.sqrt(2 * np.pi * sigma**2)
        data = scale * torch.exp(
            -(
                (gridx.unsqueeze(-1) - sample_x).pow(2)
                + (gridy.unsqueeze(-1) - sample_y).pow(2)
            )
            / (2 * sigma**2)
        )
        data = data / data.sum((0, 1), keepdims=True)

        # apply mask to taper magnitude near border if desired
        betas = self.kernel_grid[1]
        if taper_beta < max_beta:
            mask = (
                ((betas - max_beta) / (taper_beta - max_beta))
                .clamp(max=1)
                .view(1, 1, -1)
            )
        else:
            mask = torch.ones_like(data)

        data = (mask * data).unsqueeze(0).unsqueeze(0).to(torch.float32)

        self.weight = nn.Parameter(data=data, requires_grad=True)

        self.n_pts = self.weight.shape[-1]
        self.ind = torch.arange(self.n_pts)

        self.register_buffer(
            "Y",
            o3.spherical_harmonics_alpha_beta(
                range(lmax + 1), *self.kernel_grid, normalization="component"
            ),
        )

        self.angles = torch.linspace(0, 2 * torch.pi, 9)[:-1]

        self.rotation_matrices = torch.stack(
            [
                torch.tensor(
                    [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0]],
                    dtype=torch.float32,
                )
                for a in self.angles
            ]
        )
        grid = F.affine_grid(
            self.rotation_matrices,
            size=(8, 1, fmap_shape[1], fmap_shape[1]),
            align_corners=True,
        )
        grid = grid.unsqueeze(1).expand(-1, self.n_pts, -1, -1, -1)
        grid = grid.reshape(8 * self.n_pts, fmap_shape[1], fmap_shape[1], 2)
        self.register_buffer("grid", grid)

    def forward(self, x):
        """
        :x: float tensor of shape (B, C, H, W)
        :return: feature vector of shape (B,P,C) where P is number of points on S2
        """
        if type(x) is not torch.Tensor:
            x = x.tensor

        x = torch.einsum("bixy,bixyz->biz", x, self.weight)
        x = x.relu_()
        x = torch.einsum("ni,xyn->xyi", self.Y, x) / self.ind.shape[0] ** 0.5
        return x


class I2SPolicy(nn.Module):
    """
    Instantiate I2S-style network for predicting distributions over SO(3) from
    single image
    """

    def __init__(
        self,
        encoder="equiresnet50",
        lmax=6,
        s2_fdim=512,
        so3_fdim=24,
        N=8,
        initialize=True,
        device="cuda:0",
        f_out=64,
    ):

        super().__init__()
        self.encoder = ImageEncoder(
            encoder=encoder, out_fdim=s2_fdim, initialize=initialize, N=N
        )
        self.device = device
        self.encoder_name = encoder
        self.lmax = lmax

        # Projector from image feature map to spherical harmonics
        self.projector = Image2SphereProjector(
            fmap_shape=self.encoder.output_shape, lmax=lmax, rec_level=3
        )

        # S2 convolution setup
        s2_kernel_grid = s2_healpix_grid(max_beta=np.inf, rec_level=1)
        self.s2_conv = S2Conv(s2_fdim, so3_fdim, lmax, s2_kernel_grid)
        self.s2_act = e3nn.nn.SO3Activation(lmax, lmax, act=torch.relu_, resolution=8)
        # self.s2_act = e3nn.nn.S2Activation(s2_irreps(lmax), torch.tanh, 80, lmax_out=lmax)
        self.irreps = o3.Irreps([(1, (l, 1)) for l in range(lmax + 1)])

        # so3_kernel_grid = so3_healpix_grid(rec_level=3)
        so3_kernel_grid = so3_near_identity_grid()
        self.so3_conv_1 = SO3Conv(so3_fdim, 64, lmax, so3_kernel_grid, lmax)
        self.so3_act_1 = e3nn.nn.SO3Activation(lmax, 6, act=torch.relu_, resolution=8)
        self.so3_conv_2 = SO3Conv(64, f_out, 6, so3_kernel_grid, 6)

        # align the harmonic frame to the general frame (This is for any fixed general frame)
        alpha = torch.tensor(0, dtype=torch.float32)
        beta = torch.tensor(np.pi, dtype=torch.float32)
        gamma = torch.tensor(3 * np.pi / 2, dtype=torch.float32)
        self.register_buffer(
            "camera_cali", self.irreps.D_from_angles(alpha, beta, gamma)
        )
        ###############################################
        # Frame alignment for projected spherical harmonics
        #
        # Important:
        # The I2S projected spherical harmonics (SO(3) irreps) used here
        # are defined in a projector-specific axis convention, e.g., using the y-axis as the polar axis.
        #
        # In contrast, the input images follow the OpenGL camera convention, and robot actions
        # are defined in the gripper/world convention. These axis conventions generally differ,
        # e.g., using the z-axis as the polar axis.
        #
        # Therefore, we must explicitly align the harmonic
        # frame with the physical coordinate frames so that SO(3) actions in representation space
        # correspond to the intended physical axes.
        #
        # For instance, if the spherical harmonics assume the y-axis as the "up"/polar axis,
        # while the gripper pose (quaternion) is defined with the z-axis as up in the world frame,
        # directly applying the quaternion rotation would rotate features around mismatched axes,
        # leading to semantically incorrect equivariant transformations.
        #
        # This alignment step ensures that the representation space correctly captures the intended
        # physical rotations, allowing equivariant features to accurately transform under the
        # gripper's actual world orientation.
        ###############################################

    def forward(self, x, quaternion):
        """Returns so3 irreps

        :x: image, tensor of shape (B, 3, 76, 76)
        :quaternion: (x,y,z,w) tensor of shape (B, 4) represent the pose of the gripper in the world frame
        """
        # input size: 256 3 76 76
        x = self.encoder(x)  # image to feature map

        assert x.shape[2] == 7, "Only support 7x7 output shape"
        # output size: 256 1024 7 7
        x = self.projector(x)  # feature map to s2 signal on sphere

        #################################
        # This aligns i2s harmonics to OPENGL (camera) frame
        # x = rotate_s2(x.cpu(), gamma=3 * np.pi / 2).to(x.device)
        # This aligns camera frame to gripper frame
        # x = rotate_s2(x.cpu(), beta=np.pi).to(x.device)
        #################################
        # combine signals to camera and camera to gripper frame. This changes the polar axis of the harmonics to the z-axis.
        x = torch.einsum("ij,...j->...i", self.camera_cali, x)

        #  Equivariance Correction by the gripper pose in the world frame.
        x = torch.einsum(
            "bij,bfj->bfi",
            self.irreps.D_from_quaternion(
                torch.cat([quaternion[:, 3:], quaternion[:, :3]], dim=1).cpu()
            ).to(x.device),
            x,
        )

        # 256, 1024, 49
        x = self.s2_conv(x)
        x = self.s2_act(x)
        x = self.so3_conv_1(x)
        x = self.so3_act_1(x)
        out_irreps = self.so3_conv_2(x)

        return out_irreps
