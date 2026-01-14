import copy

import timm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging
from einops import rearrange
import escnn
from escnn import gspaces
from escnn.group import CyclicGroup
import numpy as np
from isp.model.common.module_attr_mixin import ModuleAttrMixin
from isp.common.pytorch_util import replace_submodules
from robomimic.models.base_nets import SpatialSoftmax

logger = logging.getLogger(__name__)


class AttentionPool2d(nn.Module):
    def __init__(
        self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(0)


class TrainRandomTestCenterCrop(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.random_crop = torchvision.transforms.RandomCrop(size=size)
        self.center_crop = torchvision.transforms.CenterCrop(size=size)

    def forward(self, x):
        if self.training:
            return self.random_crop(x)
        else:
            return self.center_crop(x)


class TimmObsEncoder(ModuleAttrMixin):
    def __init__(
        self,
        shape_meta: dict,
        model_name: str,
        pretrained: bool,
        frozen: bool,
        global_pool: str,
        transforms: list,
        # replace BatchNorm with GroupNorm
        use_group_norm: bool = False,
        # use single rgb model for all rgb inputs
        share_rgb_model: bool = False,
        # renormalize rgb input with imagenet normalization
        # assuming input in [0,1]
        imagenet_norm: bool = False,
        feature_aggregation: str = "spatial_embedding",
        downsample_ratio: int = 32,
        position_encording: str = "learnable",
    ):
        """
        Assumes rgb input: B,T,C,H,W
        Assumes low_dim input: B,T,D
        """
        super().__init__()

        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        self.low_dim_horizon = shape_meta["obs"]["robot0_eef_pos"]["horizon"]

        assert global_pool == ""

        model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            global_pool=global_pool,  # '' means no pooling
            num_classes=0,  # remove classification layer
        )
        # model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

        if frozen:
            assert pretrained
            for param in model.parameters():
                param.requires_grad = False

        feature_dim = None

        # if model_name.startswith('resnet18'):
        #     model.layer4[1].act2 = torch.nn.Identity()
        if model_name.startswith("resnet"):
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 512
            elif downsample_ratio == 16:
                modules = list(model.children())[:-3]
                model = torch.nn.Sequential(*modules)
                feature_dim = 256
            else:
                raise NotImplementedError(
                    f"Unsupported downsample_ratio: {downsample_ratio}"
                )
        elif model_name.startswith("convnext"):
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 1024
            else:
                raise NotImplementedError(
                    f"Unsupported downsample_ratio: {downsample_ratio}"
                )

        if use_group_norm:
            model = replace_submodules(
                root_module=model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=(
                        (x.num_features // 16)
                        if (x.num_features % 16 == 0)
                        else (x.num_features // 8)
                    ),
                    num_channels=x.num_features,
                ),
            )

        image_shape = None
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            type = attr.get("type", "low_dim")
            if type == "rgb":
                assert image_shape is None or image_shape == shape[1:]
                image_shape = shape[1:]
        if transforms is not None and not isinstance(transforms[0], torch.nn.Module):
            assert transforms[0].type == "RandomCrop"
            ratio = transforms[0].ratio
            transforms = [
                torchvision.transforms.RandomCrop(size=int(image_shape[0] * ratio)),
                torchvision.transforms.Resize(size=image_shape[0], antialias=True),
            ] + transforms[1:]
        transform = (
            nn.Identity() if transforms is None else torch.nn.Sequential(*transforms)
        )

        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            type = attr.get("type", "low_dim")
            key_shape_map[key] = shape
            if type == "rgb":
                rgb_keys.append(key)

                this_model = model if share_rgb_model else copy.deepcopy(model)
                key_model_map[key] = this_model

                this_transform = transform
                key_transform_map[key] = this_transform
            elif type == "low_dim":
                if not attr.get("ignore_by_policy", False):
                    low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        example_obs = torch.zeros(
            (1, 3) + image_shape, dtype=self.dtype, device=self.device
        )

        example_obs = transform(example_obs)
        example_output = model(example_obs)

        feature_dim = example_output.shape[1]
        feature_map_shape = example_output.shape[2:]

        # feature_map_shape = [x // downsample_ratio for x in image_shape]

        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)
        print("rgb keys:         ", rgb_keys)
        print("low_dim_keys keys:", low_dim_keys)

        self.model_name = model_name
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.feature_aggregation = feature_aggregation
        if model_name.startswith("vit"):
            # assert self.feature_aggregation is None # vit uses the CLS token
            if self.feature_aggregation == "all_tokens":
                # Use all tokens from ViT
                pass
            elif self.feature_aggregation is not None:
                logger.warn(
                    f"vit will use the CLS token. feature_aggregation ({self.feature_aggregation}) is ignored!"
                )
                self.feature_aggregation = None

        if self.feature_aggregation == "soft_attention":
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 1, bias=False), nn.Softmax(dim=1)
            )
        elif self.feature_aggregation == "spatial_embedding":
            self.spatial_embedding = torch.nn.Parameter(
                torch.randn(feature_map_shape[0] * feature_map_shape[1], feature_dim)
            )
        elif self.feature_aggregation == "transformer":
            if position_encording == "learnable":
                self.position_embedding = torch.nn.Parameter(
                    torch.randn(
                        feature_map_shape[0] * feature_map_shape[1] + 1, feature_dim
                    )
                )
            elif position_encording == "sinusoidal":
                num_features = feature_map_shape[0] * feature_map_shape[1] + 1
                self.position_embedding = torch.zeros(num_features, feature_dim)
                position = torch.arange(0, num_features, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, feature_dim, 2).float()
                    * (-math.log(2 * num_features) / feature_dim)
                )
                self.position_embedding[:, 0::2] = torch.sin(position * div_term)
                self.position_embedding[:, 1::2] = torch.cos(position * div_term)
            self.aggregation_transformer = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4),
                num_layers=4,
            )
        elif self.feature_aggregation == "attention_pool_2d":
            self.attention_pool_2d = AttentionPool2d(
                spacial_dim=feature_map_shape[0],
                embed_dim=feature_dim,
                num_heads=feature_dim // 64,
                output_dim=feature_dim,
            )
        elif self.feature_aggregation == "adaptive_avg_pool2d":
            self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))
        elif self.feature_aggregation == "spatial_softmax":
            if model_name == "equi_resnet18":
                num_kp = 256
            else:
                num_kp = feature_dim // 2
            self.spatial_softmax = SpatialSoftmax(
                input_shape=(feature_dim, *feature_map_shape), num_kp=num_kp
            )
        elif self.feature_aggregation == "c8_spatial_softmax":
            num_kp = (feature_dim // 8) // 4
            self.spatial_softmax = SpatialSoftmax(
                input_shape=(feature_dim // 8, *feature_map_shape), num_kp=num_kp
            )
        elif self.feature_aggregation == "c8_spatial_softmax_2":
            num_kp = feature_dim // 8
            self.spatial_softmax = SpatialSoftmax(
                input_shape=(feature_dim // 8, *feature_map_shape), num_kp=num_kp
            )
        elif self.feature_aggregation == "c8_spatial_softmax_3":
            num_kp = feature_dim // 8
            self.spatial_softmax = SpatialSoftmax(
                input_shape=(feature_dim // 8, *feature_map_shape), num_kp=num_kp
            )
        elif self.feature_aggregation == "c8_spatial_softmax_4":
            num_kp = (feature_dim // 8) // 4
            self.spatial_softmax = SpatialSoftmax(
                input_shape=(feature_dim // 8, *feature_map_shape), num_kp=num_kp
            )
            rot_mats = []
            for i in range(8):
                rot_mat = torch.eye(2)
                rot_mat[0, 0] = np.cos(-i * np.pi / 4)
                rot_mat[0, 1] = -np.sin(-i * np.pi / 4)
                rot_mat[1, 0] = np.sin(-i * np.pi / 4)
                rot_mat[1, 1] = np.cos(-i * np.pi / 4)
                rot_mats.append(rot_mat)
            self.register_buffer("rot_mats", torch.stack(rot_mats, dim=0))

        elif self.feature_aggregation == "equi_layer":
            group = escnn.gspaces.rot2dOnR2(8)
            self.equi_layer = escnn.nn.R2Conv(
                escnn.nn.FieldType(group, feature_dim // 8 * [group.regular_repr]),
                escnn.nn.FieldType(
                    group,
                    64 * [group.irrep(0)]
                    + 64 * [group.irrep(1)]
                    + 64 * [group.irrep(2)]
                    + 64 * [group.irrep(3)]
                    + 64 * [group.irrep(4)],
                ),
                kernel_size=feature_map_shape[0],
            )
        elif self.feature_aggregation == "equi_layer_2":
            group = escnn.gspaces.rot2dOnR2(8)
            self.equi_layer = escnn.nn.R2Conv(
                escnn.nn.FieldType(group, feature_dim * [group.trivial_repr]),
                escnn.nn.FieldType(
                    group,
                    64 * [group.irrep(0)]
                    + 64 * [group.irrep(1)]
                    + 64 * [group.irrep(2)]
                    + 64 * [group.irrep(3)]
                    + 64 * [group.irrep(4)],
                ),
                kernel_size=feature_map_shape[0],
            )
        elif self.feature_aggregation == "equi_layer_3":
            group = escnn.gspaces.rot2dOnR2(8)
            self.equi_layer = escnn.nn.R2Conv(
                escnn.nn.FieldType(group, feature_dim * [group.trivial_repr]),
                escnn.nn.FieldType(group, 128 * [group.regular_repr]),
                kernel_size=feature_map_shape[0],
            )
        elif self.feature_aggregation == "equi_net":
            assert feature_map_shape[0] == 7
            group = escnn.gspaces.rot2dOnR2(8)
            self.equi_layer = escnn.nn.SequentialModule(
                escnn.nn.R2Conv(
                    escnn.nn.FieldType(group, feature_dim * [group.trivial_repr]),
                    escnn.nn.FieldType(group, 128 * [group.regular_repr]),
                    kernel_size=3,
                ),
                # 5x5
                escnn.nn.ReLU(escnn.nn.FieldType(group, 128 * [group.regular_repr])),
                escnn.nn.R2Conv(
                    escnn.nn.FieldType(group, 128 * [group.regular_repr]),
                    escnn.nn.FieldType(group, 128 * [group.regular_repr]),
                    kernel_size=3,
                ),
                # 3x3
                escnn.nn.ReLU(escnn.nn.FieldType(group, 128 * [group.regular_repr])),
                escnn.nn.R2Conv(
                    escnn.nn.FieldType(group, 128 * [group.regular_repr]),
                    escnn.nn.FieldType(group, 128 * [group.regular_repr]),
                    kernel_size=3,
                ),
            )
        elif self.feature_aggregation == "equi_net_reg":
            assert feature_map_shape[0] == 7
            group = escnn.gspaces.rot2dOnR2(8)
            self.equi_layer = escnn.nn.SequentialModule(
                escnn.nn.R2Conv(
                    escnn.nn.FieldType(group, feature_dim // 8 * [group.regular_repr]),
                    escnn.nn.FieldType(group, 128 * [group.regular_repr]),
                    kernel_size=3,
                ),
                # 5x5
                escnn.nn.ReLU(escnn.nn.FieldType(group, 128 * [group.regular_repr])),
                escnn.nn.R2Conv(
                    escnn.nn.FieldType(group, 128 * [group.regular_repr]),
                    escnn.nn.FieldType(group, 128 * [group.regular_repr]),
                    kernel_size=3,
                ),
                # 3x3
                escnn.nn.ReLU(escnn.nn.FieldType(group, 128 * [group.regular_repr])),
                escnn.nn.R2Conv(
                    escnn.nn.FieldType(group, 128 * [group.regular_repr]),
                    escnn.nn.FieldType(group, 128 * [group.regular_repr]),
                    kernel_size=3,
                ),
            )

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def aggregate_feature(self, feature):
        if self.model_name.startswith("vit"):
            assert self.feature_aggregation is None  # vit uses the CLS token
            return feature[:, 0, :]

        # resnet
        assert len(feature.shape) == 4
        if self.feature_aggregation == "attention_pool_2d":
            return self.attention_pool_2d(feature)
        elif self.feature_aggregation == "adaptive_avg_pool2d":
            return self.adaptive_avg_pool2d(feature).squeeze(-1).squeeze(-1)
        elif self.feature_aggregation == "spatial_softmax":
            return self.spatial_softmax(feature).reshape(feature.shape[0], -1)
        elif self.feature_aggregation == "c8_spatial_softmax":
            feature = rearrange(feature, "b (c f) h w -> (b f) c h w", f=8)
            feature = self.spatial_softmax(feature)
            feature = rearrange(feature, "(b f) c d -> b (c f d)", f=8)
            return feature
        elif self.feature_aggregation == "c8_spatial_softmax_2":
            feature = rearrange(feature, "b (c f) h w -> b c f h w", f=8)
            feature = feature.mean(dim=2)
            feature = self.spatial_softmax(feature)
            feature = rearrange(feature, "b c d -> b (c d)")
            return feature
        elif self.feature_aggregation == "c8_spatial_softmax_3":
            feature = rearrange(feature, "b (c f) h w -> (b f) c h w", f=8)
            feature = self.spatial_softmax(feature)
            feature = rearrange(feature, "(b f) c d -> b (c d) f", f=8)
            feature = feature.mean(dim=2)
            return feature
        elif self.feature_aggregation == "c8_spatial_softmax_4":
            feature = rearrange(feature, "b (c f) h w -> (b f) c h w", f=8)
            feature = self.spatial_softmax(feature)
            feature = rearrange(feature, "(b f) c d -> b c f d", f=8)
            feature = torch.einsum("b c f d, f d m -> b c f m", feature, self.rot_mats)
            # feature = feature @ rot_mats
            feature = rearrange(feature, "b c f d -> b (c d f)")
            return feature
        elif self.feature_aggregation in ["equi_layer", "equi_net_reg"]:
            group = escnn.gspaces.rot2dOnR2(8)
            feature = escnn.nn.GeometricTensor(
                feature,
                escnn.nn.FieldType(group, feature.shape[1] // 8 * [group.regular_repr]),
            )
            feature = self.equi_layer(feature)
            feature = feature.tensor.squeeze(-1).squeeze(-1)
            return feature
        elif self.feature_aggregation in ["equi_layer_2", "equi_layer_3", "equi_net"]:
            group = escnn.gspaces.rot2dOnR2(8)
            feature = escnn.nn.GeometricTensor(
                feature,
                escnn.nn.FieldType(group, feature.shape[1] * [group.trivial_repr]),
            )
            feature = self.equi_layer(feature)
            feature = feature.tensor.squeeze(-1).squeeze(-1)
            return feature
        elif self.feature_aggregation == "i2s":
            return self.i2s(feature)
        feature = torch.flatten(feature, start_dim=-2)  # B, 512, 7*7
        feature = torch.transpose(feature, 1, 2)  # B, 7*7, 512

        if self.feature_aggregation == "avg":
            return torch.mean(feature, dim=[1])
        elif self.feature_aggregation == "max":
            return torch.amax(feature, dim=[1])
        elif self.feature_aggregation == "soft_attention":
            weight = self.attention(feature)
            return torch.sum(feature * weight, dim=1)
        elif self.feature_aggregation == "spatial_embedding":
            return torch.mean(feature * self.spatial_embedding, dim=1)
        elif self.feature_aggregation == "transformer":
            zero_feature = torch.zeros(
                feature.shape[0], 1, feature.shape[-1], device=feature.device
            )
            if self.position_embedding.device != feature.device:
                self.position_embedding = self.position_embedding.to(feature.device)
            feature_with_pos_embedding = (
                torch.concat([zero_feature, feature], dim=1) + self.position_embedding
            )
            feature_output = self.aggregation_transformer(feature_with_pos_embedding)
            return feature_output[:, 0]
        else:
            assert self.feature_aggregation is None
            return feature

    def forward(self, obs_dict):
        features = list()
        batch_size = next(iter(obs_dict.values())).shape[0]

        # process rgb input
        for key in self.rgb_keys:
            img = obs_dict[key]
            if img.shape[1] > self.shape_meta["obs"][key]["horizon"]:
                img = img[:, -self.shape_meta["obs"][key]["horizon"] :, ...]
            B, T = img.shape[:2]
            assert B == batch_size
            assert img.shape[2:] == self.key_shape_map[key]
            img = img.reshape(B * T, *img.shape[2:])
            img = self.key_transform_map[key](img)
            raw_feature = self.key_model_map[key](img)
            feature = self.aggregate_feature(raw_feature)
            assert len(feature.shape) == 2 and feature.shape[0] == B * T
            features.append(feature.reshape(B, -1))

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if data.shape[1] > self.shape_meta["obs"][key]["horizon"]:
                data = data[:, -self.shape_meta["obs"][key]["horizon"] :, ...]
            B, T = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            features.append(data.reshape(B, -1))

        # concatenate all features
        result = torch.cat(features, dim=-1)

        return result

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            this_obs = torch.zeros(
                (1, attr["horizon"]) + shape, dtype=self.dtype, device=self.device
            )
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        assert len(example_output.shape) == 2
        assert example_output.shape[0] == 1

        return example_output.shape[1:]


if __name__ == "__main__":
    timm_obs_encoder = TimmObsEncoder(
        shape_meta=None,
        model_name="resnet18.a1_in1k",
        pretrained=False,
        global_pool="",
        transforms=None,
    )


class C8EquivariantTimmObsEncoder(TimmObsEncoder):
    def __init__(
        self,
        shape_meta: dict,
        model_name: str,
        pretrained: bool,
        frozen: bool,
        global_pool: str,
        transforms: list,
        use_group_norm: bool = False,
        share_rgb_model: bool = False,
        imagenet_norm: bool = False,
        feature_aggregation: str = "spatial_embedding",
        downsample_ratio: int = 32,
        position_encording: str = "learnable",
        N: int = 8,  # Number of rotations for C8 group
        output_type: str = "reg",  # 'reg' for regular representation, 'std' for standard representation
    ):
        """
        C8-equivariant version of TimmObsEncoder using frame averaging.
        Applies frame averaging only to RGB inputs, while low-dim inputs are handled normally.
        Assumes RGB input is in trivial representation.
        output_type can be:
            - 'reg': Output is in regular representation (rotated input leads to permuted output)
            - 'std': Output is in standard representation (rotated input leads to rotated output via 2x2 matrices)
        """
        super().__init__(
            shape_meta=shape_meta,
            model_name=model_name,
            pretrained=pretrained,
            frozen=frozen,
            global_pool=global_pool,
            transforms=transforms,
            use_group_norm=use_group_norm,
            share_rgb_model=share_rgb_model,
            imagenet_norm=imagenet_norm,
            feature_aggregation=feature_aggregation,
            downsample_ratio=downsample_ratio,
            position_encording=position_encording,
        )

        assert output_type in [
            "reg",
            "std",
        ], f"output_type must be 'reg' or 'std', got {output_type}"
        self.output_type = output_type

        # C8 group setup
        self.N = N
        self.group = gspaces.no_base_space(CyclicGroup(self.N))

        # Precompute rotation matrices for grid_sample
        angles = torch.linspace(0, 2 * math.pi, self.N + 1)[
            :-1
        ]  # N angles from 0 to 2π
        self.rotation_matrices = torch.zeros(self.N, 2, 3)

        for i, angle in enumerate(angles):
            cos_val = math.cos(-angle.item())
            sin_val = math.sin(-angle.item())

            self.rotation_matrices[i, 0, 0] = cos_val
            self.rotation_matrices[i, 0, 1] = -sin_val
            self.rotation_matrices[i, 1, 0] = sin_val
            self.rotation_matrices[i, 1, 1] = cos_val

        # Register buffer for rotation matrices
        self.register_buffer("rotation_matrices_buffer", self.rotation_matrices)

        # Get the group elements for transformations
        self.group_elements = list(self.group.testing_elements)

        # Initialize permutation matrices for frame averaging
        permutation_matrices = torch.zeros(self.N, self.N, self.N)
        for r in range(self.N):
            for i in range(self.N):
                j = (i + r) % self.N
                permutation_matrices[r, i, j] = 1.0

        # Register as buffer
        self.register_buffer("permutation_matrices", permutation_matrices)

        # Pre-compute the flattened permutation matrices for batch operations
        perm_matrices_flat = permutation_matrices.reshape(self.N, -1)
        self.register_buffer("perm_matrices_flat", perm_matrices_flat)

        # Pre-compute indices for selecting the appropriate permutation matrix for each rotation
        indices_template = torch.arange(self.N)
        self.register_buffer("indices_template", indices_template)

        # Pre-compute the selected permutation matrices for batch size 1
        selected_perm_matrices_template = perm_matrices_flat[indices_template].reshape(
            self.N, self.N, self.N
        )
        self.register_buffer(
            "selected_perm_matrices_template", selected_perm_matrices_template
        )

        # For standard representation, pre-compute 2x2 rotation matrices
        if output_type == "std":
            std_rotation_matrices = torch.zeros(self.N, 2, 2)
            for i, angle in enumerate(angles):
                cos_val = math.cos(-angle.item())
                sin_val = math.sin(-angle.item())
                std_rotation_matrices[i, 0, 0] = cos_val
                std_rotation_matrices[i, 0, 1] = -sin_val
                std_rotation_matrices[i, 1, 0] = sin_val
                std_rotation_matrices[i, 1, 1] = cos_val
            self.register_buffer("std_rotation_matrices", std_rotation_matrices)

    def rotate_rgb_batch(self, img_batch):
        """
        Apply all N rotations to a batch of images efficiently in a single operation.

        Args:
            img_batch: [B, C, H, W] tensor of RGB images

        Returns:
            Batch of rotated images with shape [B*N, C, H, W], where N is the number of rotations
            The output is organized as [img0_rot0, img0_rot1, ..., img0_rotN-1, img1_rot0, ...]
        """
        B, C, H, W = img_batch.shape

        # Create an expanded batch by repeating each image N times
        # We need to ensure images and their rotations are grouped together
        # [B, C, H, W] -> [B*N, C, H, W] where each block of N images contains all rotations of a single input

        # First, expand each image to N copies
        # This creates [B, N, C, H, W] where each original image is repeated N times
        expanded = img_batch.unsqueeze(1).expand(-1, self.N, -1, -1, -1)

        # Reshape to [B*N, C, H, W]
        img_batch_expanded = expanded.reshape(B * self.N, C, H, W)

        # Now create the rotation matrices
        # For each image block of N copies, we need to apply a different rotation to each copy

        # Create pattern of indices for the N rotations, repeated for each image in the batch
        # For B=2, N=3 this would be [0,1,2, 0,1,2]
        rotation_indices = torch.arange(self.N, device=self.device).repeat(B)

        # Use these indices to select the correct rotation matrix for each image
        # [B*N, 2, 3]
        rotation_matrices = self.rotation_matrices_buffer[rotation_indices]

        # Generate sampling grid for all rotations at once
        grid = torch.nn.functional.affine_grid(
            rotation_matrices, size=(B * self.N, C, H, W), align_corners=True
        )

        # Apply all transformations in a single grid_sample operation
        rotated_imgs = torch.nn.functional.grid_sample(
            img_batch_expanded, grid, align_corners=True, padding_mode="zeros"
        )

        return rotated_imgs

    def forward(self, obs_dict):
        features = list()
        batch_size = next(iter(obs_dict.values())).shape[0]

        # process rgb input
        for key in self.rgb_keys:
            img = obs_dict[key]
            if img.shape[1] > self.shape_meta["obs"][key]["horizon"]:
                img = img[:, -self.shape_meta["obs"][key]["horizon"] :, ...]
            B, T = img.shape[:2]
            assert B == batch_size
            assert img.shape[2:] == self.key_shape_map[key]
            img = img.reshape(B * T, *img.shape[2:])

            # Apply transformations first
            img = self.key_transform_map[key](img)

            # Apply rotations to get [B*T*N, C, H, W]
            rotated_img = self.rotate_rgb_batch(img)

            # Process through vision model - output shape [B*T*N, feature_dim]
            raw_feature = self.key_model_map[key](rotated_img)

            # Aggregate feature maps - output shape [B*T*N, feature_dim]
            feature = self.aggregate_feature(raw_feature)

            # First reshape to [B*T, N, feature_dim]
            feature = feature.reshape(B * T, self.N, -1)

            # Apply frame averaging, resulting in [B*T, feature_dim*blocks]
            # where each block is of size N and preserves equivariance
            avg_feature = self._apply_frame_averaging(feature, B * T)

            # Reshape to [B, T*feature_dim*blocks]
            avg_feature = avg_feature.reshape(B, T * avg_feature.shape[1])
            features.append(avg_feature)

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if data.shape[1] > self.shape_meta["obs"][key]["horizon"]:
                data = data[:, -self.shape_meta["obs"][key]["horizon"] :, ...]
            B, T = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            features.append(data.reshape(B, -1))

        # concatenate all features
        result = torch.cat(features, dim=1)

        return result

    def _apply_frame_averaging(self, features, batch_size):
        """
        Apply frame averaging to features using permutation matrices.

        Args:
            features: Tensor of shape [B, N, feature_dim]
            batch_size: Batch size

        Returns:
            Tensor of shape [B, feature_dim] with frame averaging applied
        """
        if self.output_type == "reg":
            # Regular representation - use permutation matrices
            feature_dim = features.shape[2]
            blocks = feature_dim // self.N
            features = features.reshape(batch_size, self.N, blocks, self.N)

            all_features_flat = features.reshape(-1, blocks, self.N)
            selected_perm_matrices = self.selected_perm_matrices_template.repeat(
                batch_size, 1, 1
            )
            aligned_features_flat = torch.bmm(all_features_flat, selected_perm_matrices)
            aligned_features = aligned_features_flat.reshape(
                batch_size, self.N, blocks, self.N
            )
            avg_features = torch.mean(aligned_features, dim=1)  # [B, blocks, N]
            return avg_features.reshape(batch_size, blocks * self.N)
        else:
            # Standard representation - use 2x2 rotation matrices
            feature_dim = features.shape[2]
            assert (
                feature_dim % 2 == 0
            ), "For standard representation, feature_dim must be even"
            blocks = feature_dim // 2

            # Reshape into pairs [B, N, blocks, 2]
            features = features.reshape(batch_size, self.N, blocks, 2)

            # Flatten for batch matrix multiplication [B*N, blocks, 2]
            all_features_flat = features.reshape(-1, blocks, 2)

            # Get rotation matrices for each rotation [B*N, 2, 2]
            selected_rot_matrices = self.std_rotation_matrices.repeat(batch_size, 1, 1)

            # Apply rotations to each pair [B*N, blocks, 2]
            aligned_features_flat = torch.bmm(all_features_flat, selected_rot_matrices)

            # Reshape back to [B, N, blocks, 2]
            aligned_features = aligned_features_flat.reshape(
                batch_size, self.N, blocks, 2
            )

            # Average over rotations [B, blocks, 2]
            avg_features = torch.mean(aligned_features, dim=1)

            # Flatten back to [B, feature_dim]
            return avg_features.reshape(batch_size, blocks * 2)
