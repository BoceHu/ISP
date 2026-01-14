import copy
from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from einops import reduce
import torchvision
import pytorch3d.transforms as pytorch3d_transforms
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from isp.model.common.normalizer import LinearNormalizer
from isp.policy.base_image_policy import BaseImagePolicy
from isp.model.diffusion.mask_generator import LowdimMaskGenerator

try:
    import robomimic.models.base_nets as rmbn

    if not hasattr(rmbn, "CropRandomizer"):
        raise ImportError("CropRandomizer is not in robomimic.models.base_nets")
except ImportError:
    import robomimic.models.obs_core as rmbn
from isp.model.equi.equi_obs_encoder_so2_pretrain import ISPObsEnc
from isp.model.equi.equi_conditional_unet1d_c8 import EquiDiffusionUNet
from isp.model.vision.rot_randomizer import (
    RotRandomizer,
    RotRandomizerForPrediction,
)
from isp.model.equi.equi_group_sampling import EquiGroupSamplingC8
from isp.model.common.rotation_transformer import RotationTransformer


class ISPSO2PretrainedPolicy(BaseImagePolicy):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        # task params
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        # image
        crop_shape=(76, 76),
        # arch
        N=8,
        enc_n_hidden=128,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        rot_aug=False,
        image_shape=(84, 84),
        # i2s params
        lmax=6,
        s2_fdim=1024,
        so3_fdim=48,
        initialize=True,
        encoder="equiresnet50",
        **kwargs,
    ):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta["obs"]

        self.enc = ISPObsEnc(
            obs_shape=obs_shape_meta["robot0_eye_in_hand_image"]["shape"],
            crop_shape=crop_shape,
            n_hidden=enc_n_hidden,
            N=8,
            initialize=initialize,
            lmax=lmax,
            s2_fdim=s2_fdim,
            so3_fdim=so3_fdim,
            encoder=encoder,
        )

        obs_feature_dim = enc_n_hidden
        global_cond_dim = obs_feature_dim * n_obs_steps

        self.equi_sampler = EquiGroupSamplingC8(lmax=lmax, f_out=obs_feature_dim)

        self.diff = EquiDiffusionUNet(
            act_emb_dim=64,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            N=N,
            lmax=lmax,
        )

        print("Enc params: %e" % sum(p.numel() for p in self.enc.parameters()))
        print(
            "Equi sampler params: %e"
            % sum(p.numel() for p in self.equi_sampler.parameters())
        )

        print("Diff params: %e" % sum(p.numel() for p in self.diff.parameters()))

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.rot_randomizer = RotRandomizer()
        self.rot_randomizer2 = RotRandomizerForPrediction()

        self.horizon = horizon
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.crop_shape = crop_shape
        self.obs_feature_dim = obs_feature_dim
        self.rot_aug = rot_aug

        print("Data Augmentation: ", self.rot_aug)
        print("n_obs_steps: ", n_obs_steps)
        print("This is using a pretrained equivariant ResNet")

        self.kwargs = kwargs

        self.noise_scheduler = noise_scheduler
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        self.register_buffer(
            "ws_center", torch.tensor([0, 0, 0.8], dtype=torch.float32)
        )

        self.sixd2mat = RotationTransformer("rotation_6d", "matrix")

    # ========= training  ============
    def set_ws_center(self, ws_center):
        self.ws_center.copy_(ws_center)

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        eps: float,
    ) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            weight_decay=weight_decay,
            lr=learning_rate,
            betas=betas,
            eps=eps,
        )
        return optimizer

    # ========= inference  ============
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        model = self.diff
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            results = self.equi_sampler(global_cond, trajectory)
            model_output = model(
                results["trajectory"],
                t,
                local_cond=local_cond,
                global_cond=results["global_cond"],
            )

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, generator=generator, **kwargs
            ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        obs_dict = copy.deepcopy(obs_dict)
        obs_dict["robot0_eef_pos"] -= self.ws_center
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        global_cond = self.enc(nobs)
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=None,
            global_cond=global_cond,
            **self.kwargs,
        )

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        # Converting XY-axes (first two columns) of the end-effector to 6D representation (first two rows)
        # I already tested with this, the network is E2E equivariant.
        # TODO: Check whether this is necessary to keep the correct equivariance (Can we remove this?).
        rot_mat = pytorch3d_transforms.rotation_6d_to_matrix(
            naction_pred[..., 3:9].reshape(-1, 6)
        )
        rot_6d = rot_mat.transpose(2, 1)[:, :2, :]
        rot_6d = rot_6d.reshape(B, T, 6)

        naction_pred = torch.cat(
            (naction_pred[..., :3], rot_6d, naction_pred[..., 9:]), dim=-1
        )
        action_pred = self.normalizer["action"].unnormalize(naction_pred)
        action_pred[:, :, :3] += self.ws_center

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {"action": action, "action_pred": action_pred}
        return result

    # ========= training  ============

    def compute_loss(self, batch):
        # normalize input
        batch = copy.deepcopy(batch)
        batch["action"][:, :, :3] -= self.ws_center
        batch["obs"]["robot0_eef_pos"] -= self.ws_center
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])

        if self.rot_aug:
            nobs, nactions = self.rot_randomizer(nobs, nactions)

        # Converting 6D representation (first two rows) to XY- axes of the end-effector (first two columns) so that it is compatible to SO(3) equ
        # I already tested with this, the network is E2E equivariant.
        # Check whether this is necessary to keep the correct equivariance (Can we remove this?).
        rot_mat = pytorch3d_transforms.rotation_6d_to_matrix(
            nactions[:, :, 3:9].reshape(-1, 6)
        )
        rot_xy = rot_mat.transpose(2, 1)[:, :2, :].reshape(
            nactions.shape[0], nactions.shape[1], 6
        )
        nactions = torch.cat((nactions[:, :, :3], rot_xy, nactions[:, :, 9:]), dim=-1)

        trajectory = nactions
        cond_data = trajectory

        global_cond = self.enc(nobs)

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=trajectory.device,
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        results = self.equi_sampler(global_cond, noisy_trajectory)

        # Predict the noise residual
        pred = self.diff(
            results["trajectory"],
            timesteps,
            local_cond=None,
            global_cond=results["global_cond"],
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss
