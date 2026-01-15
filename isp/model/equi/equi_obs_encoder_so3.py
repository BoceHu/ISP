import torch
import e3nn
import e3nn.nn
from e3nn import o3
import healpy as hp
import numpy as np
from torchvision import models as vision_models
from escnn import gspaces, nn
from escnn.group import CyclicGroup
from einops import rearrange
from robomimic.models.base_nets import SpatialSoftmax
from isp.model.common.module_attr_mixin import ModuleAttrMixin
import isp.model.vision.crop_randomizer as dmvc
from isp.model.common.rotation_transformer import RotationTransformer
from isp.model.equi.i2s_policy import I2SPolicy
from isp.model.equi.isp_util import (
    s2_irreps,
    so3_irreps,
    s2_healpix_grid,
    SO3Conv,
    so3_near_identity_grid,
)


class ISPObsEnc(ModuleAttrMixin):
    def __init__(
        self,
        encoder="equiresnet50",
        obs_shape=(3, 84, 84),
        crop_shape=(76, 76),
        n_hidden=128,
        N=8,
        initialize=True,
        lmax=6,
        s2_fdim=512,
        so3_fdim=24,
    ):
        super().__init__()
        self.lmax = lmax

        self.enc_obs = I2SPolicy(
            encoder=encoder,
            lmax=lmax,
            s2_fdim=s2_fdim,
            so3_fdim=so3_fdim,
            N=N,
            initialize=initialize,
            f_out=n_hidden,
        )

        self.robot_state_embed = None
        so3_kernel_grid = so3_near_identity_grid(n_alpha=4)
        self.enc_out = SO3Conv(n_hidden + 32, n_hidden, lmax, so3_kernel_grid, lmax)

        self.quaternion_to_sixd = RotationTransformer("quaternion", "rotation_6d")
        self.quaternion_to_matrix = RotationTransformer("quaternion", "matrix")

        self.crop_randomizer = dmvc.CropRandomizer(
            input_shape=obs_shape,
            crop_height=crop_shape[0],
            crop_width=crop_shape[1],
        )

        self.proj_1 = o3.Linear("2x0e+3x1e", so3_irreps(1), f_in=1, f_out=32)
        self.proj_2 = o3.Linear(so3_irreps(lmax), so3_irreps(lmax), f_in=32, f_out=32)
        self.so3_act = e3nn.nn.SO3Activation(1, lmax, act=torch.relu, resolution=8)

    def getMatrixRotation(self, quat):
        # data is in xyzw, but rotation transformer takes wxyz
        return self.quaternion_to_matrix.forward(quat[:, [3, 0, 1, 2]])

    def getSixDRotation(self, quat):
        # data is in xyzw, but rotation transformer takes wxyz
        return self.quaternion_to_sixd.forward(quat[:, [3, 0, 1, 2]])

    def forward(self, nobs):
        ee_pos = nobs["robot0_eef_pos"]  # 128, 2, 3
        ee_quat = nobs["robot0_eef_quat"]  # 128, 2, 4
        ee_q = nobs["robot0_gripper_qpos"]  # 128, 2, 2
        ih = nobs["robot0_eye_in_hand_image"]  # 128, 2, 3, 84, 84
        # B, T, C, H, W
        batch_size = ih.shape[0]
        ih = rearrange(ih, "b t c h w -> (b t) c h w")
        ee_pos = rearrange(ee_pos, "b t d -> (b t) d")
        ee_quat = rearrange(ee_quat, "b t d -> (b t) d")
        ee_q = rearrange(ee_q, "b t d -> (b t) d")
        ih = self.crop_randomizer(ih)
        ee_rot_xy = self.getMatrixRotation(ee_quat).transpose(2, 1)[:, :2, :]

        enc_128_out = self.enc_obs(ih, ee_quat)  # 256 128 455

        ee_rot_xy = ee_rot_xy.reshape(enc_128_out.shape[0], 2 * 3)
        robot_state_embed = torch.zeros(
            [enc_128_out.shape[0], 1, 11], device=enc_128_out.device
        )

        robot_state_embed[:, 0, :2] = ee_q
        robot_state_embed[:, 0, 2:5] = ee_pos
        robot_state_embed[:, 0, 5:11] = ee_rot_xy

        robot_state_embed = self.proj_1(robot_state_embed)
        robot_state_embed = self.so3_act(robot_state_embed)
        robot_state_embed = self.proj_2(robot_state_embed)
        features = torch.cat([enc_128_out, robot_state_embed], dim=1)
        out = self.enc_out(features)
        return rearrange(out, "(b t) c f -> b t c f", b=batch_size)
