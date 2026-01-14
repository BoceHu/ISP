import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import random
import numpy as np
from einops import rearrange
import math

from isp.model.common.rotation_transformer import RotationTransformer


class RotRandomizer(nn.Module):
    """
    Continuously and randomly rotate the input tensor during training.
    Does not rotate the tensor during evaluation.
    """

    def __init__(self, min_angle=-180, max_angle=180):
        """
        Args:
            min_angle (float): Minimum rotation angle.
            max_angle (float): Maximum rotation angle.
        """
        super().__init__()
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.tf = RotationTransformer("quaternion", "matrix")
        self.sixd2mat = RotationTransformer("rotation_6d", "matrix")

    def forward(self, nobs, naction):
        """
        Randomly rotates the inputs if in training mode.
        Keeps inputs unchanged if in evaluation mode.

        Args:
            inputs (torch.Tensor): input tensors

        Returns:
            torch.Tensor: rotated or unrotated tensors based on the mode
        """
        if self.training:
            pos = nobs["robot0_eef_pos"]
            # x, y, z, w -> w, x, y, z
            quat = nobs["robot0_eef_quat"][:, :, [3, 0, 1, 2]]
            batch_size = pos.shape[0]
            T = pos.shape[1]

            for i in range(1000):
                # angles = torch.rand(batch_size) * 2 * np.pi - np.pi
                # rotation_matrix = torch.zeros((batch_size, 3, 3), device=obs.device)
                # rotation_matrix[:, 2, 2] = 1
                #
                # angles[torch.rand(batch_size) < 1/64] = 0
                # rotation_matrix[:, 0, 0] = torch.cos(angles)
                # rotation_matrix[:, 0, 1] = -torch.sin(angles)
                # rotation_matrix[:, 1, 0] = torch.sin(angles)
                # rotation_matrix[:, 1, 1] = torch.cos(angles)

                # angles_x = torch.rand(batch_size) * 2 * np.pi - np.pi  # X
                # angles_y = torch.rand(batch_size) * 2 * np.pi - np.pi  # Y
                angles_z = torch.rand(batch_size) * 2 * np.pi - np.pi  # Z

                mask = torch.rand(batch_size) < 1 / 64
                # angles_x[mask] = 0
                # angles_y[mask] = 0
                angles_z[mask] = 0

                # rot_x = torch.zeros((batch_size, 3, 3), device=pos.device)
                # rot_x[:, 0, 0] = 1
                # rot_x[:, 1, 1] = torch.cos(angles_x)
                # rot_x[:, 1, 2] = -torch.sin(angles_x)
                # rot_x[:, 2, 1] = torch.sin(angles_x)
                # rot_x[:, 2, 2] = torch.cos(angles_x)

                # rot_y = torch.zeros((batch_size, 3, 3), device=pos.device)
                # rot_y[:, 0, 0] = torch.cos(angles_y)
                # rot_y[:, 0, 2] = torch.sin(angles_y)
                # rot_y[:, 1, 1] = 1
                # rot_y[:, 2, 0] = -torch.sin(angles_y)
                # rot_y[:, 2, 2] = torch.cos(angles_y)

                rot_z = torch.zeros((batch_size, 3, 3), device=pos.device)
                rot_z[:, 0, 0] = torch.cos(angles_z)
                rot_z[:, 0, 1] = -torch.sin(angles_z)
                rot_z[:, 1, 0] = torch.sin(angles_z)
                rot_z[:, 1, 1] = torch.cos(angles_z)
                rot_z[:, 2, 2] = 1
                rotation_matrix = rot_z

                rotated_naction = naction.clone()
                rotated_naction[:, :, 0:3] = (
                    rotation_matrix @ naction[:, :, 0:3].permute(0, 2, 1)
                ).permute(0, 2, 1)

                rot_mat = self.sixd2mat.forward(rotated_naction[:, :, 3:9])
                rot_mat = rotation_matrix.unsqueeze(1) @ rot_mat
                rotated_6d = self.sixd2mat.inverse(rot_mat)
                rotated_naction[:, :, 3:9] = rotated_6d

                # rotated_naction[:, :, [3, 6]] = (
                #             rotation_matrix[:, :2, :2] @ naction[:, :, [3, 6]].permute(0, 2, 1)).permute(0, 2, 1)
                # rotated_naction[:, :, [4, 7]] = (
                #             rotation_matrix[:, :2, :2] @ naction[:, :, [4, 7]].permute(0, 2, 1)).permute(0, 2, 1)
                # rotated_naction[:, :, [5, 8]] = (
                #             rotation_matrix[:, :2, :2] @ naction[:, :, [5, 8]].permute(0, 2, 1)).permute(0, 2, 1)

                rotated_pos = (rotation_matrix @ pos.permute(0, 2, 1)).permute(0, 2, 1)
                rot = self.tf.forward(quat)
                rotated_rot = rotation_matrix.unsqueeze(1) @ rot
                rotated_quat = self.tf.inverse(rotated_rot)

                if (
                    rotated_pos.min() >= -1
                    and rotated_pos.max() <= 1
                    and rotated_naction[:, :, :2].min() > -1
                    and rotated_naction[:, :, :2].max() < 1
                ):
                    break
            if i == 999:
                return nobs, naction

            nobs["robot0_eef_pos"] = rotated_pos
            # w, x, y, z -> x, y, z, w
            nobs["robot0_eef_quat"] = rotated_quat[:, :, [1, 2, 3, 0]]
            naction = rotated_naction

        return nobs, naction

    def __repr__(self):
        """Pretty print the network."""
        header = "{}".format(str(self.__class__.__name__))
        msg = header + "(min_angle={}, max_angle={})".format(
            self.min_angle, self.max_angle
        )
        return msg


class RotRandomizerForPrediction(nn.Module):
    """
    Continuously and randomly rotate the input tensor during training.
    Does not rotate the tensor during evaluation.
    """

    def __init__(self, min_angle=-180, max_angle=180):
        """
        Args:
            min_angle (float): Minimum rotation angle.
            max_angle (float): Maximum rotation angle.
        """
        super().__init__()
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.tf = RotationTransformer("quaternion", "matrix")
        self.rotation_matrices = None

    def forward(self, nobs):
        """
        Randomly rotates the inputs if in training mode.
        Keeps inputs unchanged if in evaluation mode.

        Args:
            inputs (torch.Tensor): input tensors

        Returns:
            torch.Tensor: rotated or unrotated tensors based on the mode
        """
        pos = nobs["robot0_eef_pos"]
        # x, y, z, w -> w, x, y, z
        quat = nobs["robot0_eef_quat"][:, :, [3, 0, 1, 2]]
        batch_size = pos.shape[0]

        for i in range(1000):

            angles_x = torch.rand(batch_size) * 2 * np.pi - np.pi  # X
            angles_y = torch.rand(batch_size) * 2 * np.pi - np.pi  # Y
            angles_z = torch.rand(batch_size) * 2 * np.pi - np.pi  # Z

            mask = torch.rand(batch_size) < 1 / 64
            angles_x[mask] = 0
            angles_y[mask] = 0
            angles_z[mask] = 0

            rot_x = torch.zeros((batch_size, 3, 3), device=pos.device)
            rot_x[:, 0, 0] = 1
            rot_x[:, 1, 1] = torch.cos(angles_x)
            rot_x[:, 1, 2] = -torch.sin(angles_x)
            rot_x[:, 2, 1] = torch.sin(angles_x)
            rot_x[:, 2, 2] = torch.cos(angles_x)

            rot_y = torch.zeros((batch_size, 3, 3), device=pos.device)
            rot_y[:, 0, 0] = torch.cos(angles_y)
            rot_y[:, 0, 2] = torch.sin(angles_y)
            rot_y[:, 1, 1] = 1
            rot_y[:, 2, 0] = -torch.sin(angles_y)
            rot_y[:, 2, 2] = torch.cos(angles_y)

            rot_z = torch.zeros((batch_size, 3, 3), device=pos.device)
            rot_z[:, 0, 0] = torch.cos(angles_z)
            rot_z[:, 0, 1] = -torch.sin(angles_z)
            rot_z[:, 1, 0] = torch.sin(angles_z)
            rot_z[:, 1, 1] = torch.cos(angles_z)
            rot_z[:, 2, 2] = 1
            rotation_matrix = rot_z @ rot_y @ rot_x
            self.rotation_matrix = rotation_matrix

            rotated_pos = (rotation_matrix @ pos.permute(0, 2, 1)).permute(0, 2, 1)
            rot = self.tf.forward(quat)
            rotated_rot = rotation_matrix.unsqueeze(1) @ rot
            rotated_quat = self.tf.inverse(rotated_rot)

            if rotated_pos.min() >= -1 and rotated_pos.max() <= 1:
                break
        if i == 999:
            identity_matrix = (
                torch.eye(3, device=pos.device).unsqueeze(0).repeat(batch_size, 1, 1)
            )
            self.rotation_matrices = identity_matrix
            return nobs, identity_matrix

        nobs["robot0_eef_pos"] = rotated_pos
        # w, x, y, z -> x, y, z, w
        nobs["robot0_eef_quat"] = rotated_quat[:, :, [1, 2, 3, 0]]

        return nobs, self.rotation_matrix

    def __repr__(self):
        """Pretty print the network."""
        header = "{}".format(str(self.__class__.__name__))
        msg = header + "(min_angle={}, max_angle={})".format(
            self.min_angle, self.max_angle
        )
        return msg


class RotRandomizer2(nn.Module):
    """
    Continuously and randomly rotate the input tensor during training.
    Does not rotate the tensor during evaluation.
    """

    def __init__(self, min_angle=-180, max_angle=180):
        """
        Args:
            min_angle (float): Minimum rotation angle.
            max_angle (float): Maximum rotation angle.
        """
        super().__init__()
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.tf = RotationTransformer("quaternion", "matrix")
        self.sixd2mat = RotationTransformer("rotation_6d", "matrix")

    def forward(self, nobs, naction):
        """
        Randomly rotates the inputs if in training mode.
        Keeps inputs unchanged if in evaluation mode.

        Args:
            inputs (torch.Tensor): input tensors

        Returns:
            torch.Tensor: rotated or unrotated tensors based on the mode
        """
        if self.training:
            pos = nobs["robot0_eef_pos"]
            # x, y, z, w -> w, x, y, z
            quat = nobs["robot0_eef_quat"][:, :, [3, 0, 1, 2]]
            batch_size = pos.shape[0]
            T = pos.shape[1]

            for i in range(1000):
                # angles = torch.rand(batch_size) * 2 * np.pi - np.pi
                # rotation_matrix = torch.zeros((batch_size, 3, 3), device=obs.device)
                # rotation_matrix[:, 2, 2] = 1
                #
                # angles[torch.rand(batch_size) < 1/64] = 0
                # rotation_matrix[:, 0, 0] = torch.cos(angles)
                # rotation_matrix[:, 0, 1] = -torch.sin(angles)
                # rotation_matrix[:, 1, 0] = torch.sin(angles)
                # rotation_matrix[:, 1, 1] = torch.cos(angles)

                # angles_x = torch.rand(batch_size) * 2 * np.pi - np.pi  # X
                angles_y = torch.tensor(torch.pi, device=pos.device)  # Y
                # angles_z = torch.rand(batch_size) * 2 * np.pi - np.pi  # Z

                # mask = torch.rand(batch_size) < 1 / 64
                # angles_x[mask] = 0
                # angles_y[mask] = 0
                # angles_z[mask] = 0

                # rot_x = torch.zeros((batch_size, 3, 3), device=pos.device)
                # rot_x[:, 0, 0] = 1
                # rot_x[:, 1, 1] = torch.cos(angles_x)
                # rot_x[:, 1, 2] = -torch.sin(angles_x)
                # rot_x[:, 2, 1] = torch.sin(angles_x)
                # rot_x[:, 2, 2] = torch.cos(angles_x)

                rot_y = torch.zeros((batch_size, 3, 3), device=pos.device)
                rot_y[:, 0, 0] = torch.cos(angles_y)
                rot_y[:, 0, 2] = torch.sin(angles_y)
                rot_y[:, 1, 1] = 1
                rot_y[:, 2, 0] = -torch.sin(angles_y)
                rot_y[:, 2, 2] = torch.cos(angles_y)

                # rot_z = torch.zeros((batch_size, 3, 3), device=pos.device)
                # rot_z[:, 0, 0] = torch.cos(angles_z)
                # rot_z[:, 0, 1] = -torch.sin(angles_z)
                # rot_z[:, 1, 0] = torch.sin(angles_z)
                # rot_z[:, 1, 1] = torch.cos(angles_z)
                # rot_z[:, 2, 2] = 1
                # rotation_matrix = rot_z @ rot_y @ rot_x
                rotation_matrix = rot_y

                rotated_naction = naction.clone()
                rotated_naction[:, :, 0:3] = (
                    rotation_matrix @ naction[:, :, 0:3].permute(0, 2, 1)
                ).permute(0, 2, 1)

                rot_mat = self.sixd2mat.forward(rotated_naction[:, :, 3:9])
                rot_mat = rotation_matrix.unsqueeze(1) @ rot_mat
                rotated_6d = self.sixd2mat.inverse(rot_mat)
                rotated_naction[:, :, 3:9] = rotated_6d

                # rotated_naction[:, :, [3, 6]] = (
                #             rotation_matrix[:, :2, :2] @ naction[:, :, [3, 6]].permute(0, 2, 1)).permute(0, 2, 1)
                # rotated_naction[:, :, [4, 7]] = (
                #             rotation_matrix[:, :2, :2] @ naction[:, :, [4, 7]].permute(0, 2, 1)).permute(0, 2, 1)
                # rotated_naction[:, :, [5, 8]] = (
                #             rotation_matrix[:, :2, :2] @ naction[:, :, [5, 8]].permute(0, 2, 1)).permute(0, 2, 1)

                rotated_pos = (rotation_matrix @ pos.permute(0, 2, 1)).permute(0, 2, 1)
                rot = self.tf.forward(quat)
                rotated_rot = rotation_matrix.unsqueeze(1) @ rot
                rotated_quat = self.tf.inverse(rotated_rot)

                break
            if i == 999:
                return nobs, naction

            nobs["robot0_eef_pos"] = rotated_pos
            # w, x, y, z -> x, y, z, w
            nobs["robot0_eef_quat"] = rotated_quat[:, :, [1, 2, 3, 0]]
            naction = rotated_naction

        return nobs, naction

    def __repr__(self):
        """Pretty print the network."""
        header = "{}".format(str(self.__class__.__name__))
        msg = header + "(min_angle={}, max_angle={})".format(
            self.min_angle, self.max_angle
        )
        return msg
