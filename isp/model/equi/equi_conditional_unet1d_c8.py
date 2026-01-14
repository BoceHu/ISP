from typing import Union
import torch
from escnn import gspaces, nn
from einops import rearrange, repeat

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

        cyc_gspace = gspaces.rot2dOnR2(N)
        self.cyc_gspace = gspaces.no_base_space(cyc_gspace.fibergroup)
        self.act_type = nn.FieldType(
            self.cyc_gspace, act_emb_dim * [self.cyc_gspace.regular_repr]
        )

        self.out_layer = nn.Linear(self.act_type, self.getOutFieldType())

    def getOutFieldType(self):
        return nn.FieldType(
            self.cyc_gspace,
            3 * [self.cyc_gspace.irrep(1)]
            + 4 * [self.cyc_gspace.trivial_repr],  # 8  # 2
        )

    def getOutput(self, conv_out):
        xy = conv_out[:, 0:2]
        x1 = conv_out[:, 2:3]
        y1 = conv_out[:, 3:4]
        x2 = conv_out[:, 4:5]
        y2 = conv_out[:, 5:6]
        z1 = conv_out[:, 6:7]
        z2 = conv_out[:, 7:8]
        z = conv_out[:, 8:9]
        g = conv_out[:, 9:10]

        action = torch.cat((xy, z, x1, y1, z1, x2, y2, z2, g), dim=1)
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
