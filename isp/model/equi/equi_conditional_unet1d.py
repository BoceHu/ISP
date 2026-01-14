from typing import Union
import torch
from escnn import gspaces, nn
from einops import rearrange, repeat
import numpy as np

from isp.model.diffusion.conditional_unet1d import ConditionalUnet1D


class EquiDiffusionUNet(torch.nn.Module):
    def __init__(
        self,
        act_emb_dim,
        local_cond_dim,
        global_cond_dim,
        diffusion_step_embed_dim,
        down_dims,
        kernel_size,
        n_groups,
        cond_predict_scale,
        N,
        lmax,
    ):
        super().__init__()
        self.unet = ConditionalUnet1D(
            input_dim=act_emb_dim,
            local_cond_dim=local_cond_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )
        self.N = N
        ico_gspace = gspaces.icoOnR3()
        self.ico_gspace = gspaces.no_base_space(ico_gspace.fibergroup)
        self.act_type = nn.FieldType(
            self.ico_gspace, act_emb_dim * [self.ico_gspace.regular_repr]
        )

        self.out_layer = nn.Linear(self.act_type, self.getOutFieldType())

        g_list = list(self.ico_gspace.testing_elements)
        per_mat = torch.zeros(N, N)
        for i in range(N):
            to_mat = nn.FieldType(
                self.ico_gspace, [self.ico_gspace.irrep(1)]
            ).fiber_representation(g_list[i])
            from_mats = []
            for j in range(N):
                from_mat = g_list[j].to("MAT")
                from_mats.append(from_mat)
            arg_min = (
                (to_mat - torch.tensor(np.stack(from_mats)))
                .reshape(N, -1)
                .abs()
                .sum(dim=-1)
                .argmin()
            )
            per_mat[i, arg_min] = 1
        per_mat = per_mat.T
        self.register_buffer("per_mat", per_mat)

    def getOutFieldType(self):
        return nn.FieldType(
            self.ico_gspace,
            3 * [self.ico_gspace.irrep(1)]  # 9
            + 1 * [self.ico_gspace.trivial_repr],  # 1
        )

    def getOutput(self, conv_out):
        """
        conv_out: (B, T, 4, 4)
        """
        xyz = conv_out[:, 0:3]
        rot_col_1 = conv_out[:, 3:6]
        rot_col_2 = conv_out[:, 6:9]
        g = conv_out[:, 9:10]
        action = torch.cat((xyz, rot_col_1, rot_col_2, g), dim=1)

        return action

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        local_cond=None,
        global_cond=None,
        **kwargs
    ):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        global_cond = torch.einsum("b h c f, f g -> b h c g", global_cond, self.per_mat)
        sample = torch.einsum("b t c f, f g -> b t c g", sample, self.per_mat)
        B, T = sample.shape[:2]
        sample = rearrange(sample, "b t c f -> (b f) t c", f=self.N)

        if type(timestep) == torch.Tensor and len(timestep.shape) == 1:
            timestep = repeat(timestep, "b -> (b f)", f=self.N)
        if local_cond is not None:
            local_cond = rearrange(local_cond, "b t (c f) -> (b f) t c", f=self.N)
        if global_cond is not None:
            global_cond = rearrange(global_cond, "b h c f -> (b f) (h c)", f=self.N)
        out = self.unet(sample, timestep, local_cond, global_cond, **kwargs)
        out = rearrange(out, "(b f) t c -> (b t) (c f)", f=self.N)
        out = nn.GeometricTensor(out, self.act_type)
        out = self.out_layer(out).tensor.reshape(B * T, -1)
        out = self.getOutput(out)

        out = rearrange(out, "(b t) n -> b t n", b=B)
        return out

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
