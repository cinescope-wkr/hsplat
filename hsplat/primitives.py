"""
Define primitives for CGH

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

Article: 
S. Choi, B. Chao, J. Yang, M. Gopakumar, G. Wetzstein
"Gaussian Wave Splatting for Computer-generated Holography",
ACM Transactions on Graphics (Proc. SIGGRAPH 2025)
"""
import math
import sys
import time
import types
from typing import Any, List, Tuple, Union, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pytorch3d.transforms import (
    quaternion_to_matrix,
    matrix_to_quaternion,
    quaternion_multiply,
)
from torch import Tensor

import utils

import logging
logger = logging.getLogger(__name__)

def get_obj_from_module(module: types.ModuleType, obj_name: str) -> Any:
    """Traverses the object name and returns the last (rightmost) Python object."""
    if obj_name == "":
        return module
    obj = module
    for part in obj_name.split("."):
        obj = getattr(obj, part)
    return obj


def get_module_from_obj_name(name: str) -> Tuple[types.ModuleType, str]:
    """Splits the object name into the module and the actual object name."""
    parts = name.split(".")
    module_name = ".".join(parts[:-1])
    obj_name = parts[-1]
    if module_name:
        module = sys.modules[module_name]
    else:
        module = sys.modules["__main__"]  # Use the current module
    return module, obj_name


def get_obj_by_name(name: str) -> Any:
    """Finds the Python object with the given name."""
    module, obj_name = get_module_from_obj_name(name)
    return get_obj_from_module(module, obj_name)


def call_func_by_name(*args, func_name: str = None, **kwargs) -> Any:
    """Finds the Python object with the given name and calls it as a function."""
    assert func_name is not None
    func_obj = get_obj_by_name(func_name)
    assert callable(func_obj)
    return func_obj(*args, **kwargs)


def construct_class_by_name(*args, class_name: str = None, **kwargs) -> Any:
    """Finds the Python class with the given name and constructs it with the given arguments."""
    return call_func_by_name(*args, func_name=class_name, **kwargs)


class Primitives(nn.Module):
    def __init__(self, primitives=None):
        """
        Base class for managing a collection of various primitives.
        """
        super().__init__()
        self.primitives = []

    def add_primitive(self, primitive):
        """Add a primitive to the collection."""
        self.primitives.append(primitive)

    def __iter__(self):
        """Iterate through the primitives in the collection."""
        return iter(self.primitives)

    def __getitem__(self, idx):
        """Make the class subscriptable by returning the primitive at the given index."""
        return self.primitives[idx]

    def __len__(self):
        logger.debug(f"Debug: self.primitives has {len(self.primitives)} elements")
        return len(self.primitives)

    def to(self, *args, **kwargs):
        slf = super().to(*args, **kwargs)
        for primitive in slf.primitives:
            primitive.to(*args, **kwargs)
        return slf

    def select_random_points(self, num_points):
        indices = torch.randperm(len(self.primitives))[:num_points]
        return Primitives([self.primitives[i] for i in indices])

    def sort(self, order="front2back"):
        pass


class Primitive(nn.Module):
    def __init__(self, position, normal=None, amplitude=1.0, phase=0.0):
        """
        Base class for a primitive.

        Parameters
        ----------
        position : array-like or Tensor
            Position of the primitive in 3D space.
        normal : torch.Tensor or None
            3D normal vector. Default is None for primitives with no specific normal.
        amplitude : float
            Amplitude of the wavefront contribution.
        phase : float
            Phase of the wavefront contribution.
        """
        super().__init__()
        self.position = position
        self.normal = normal
        self.amplitude = torch.tensor(amplitude, dtype=torch.float32)
        self.phase = torch.tensor(phase, dtype=torch.float32)

    def get_wavefront_contribution(self):
        """
        Compute the wavefront contribution for the primitive.
        This is a placeholder method meant to be overridden in derived classes.
        """
        raise NotImplementedError("This method should be implemented in derived classes.")

    @property
    def normal(self):
        return self._normal

    @normal.setter
    def normal(self, value):
        if value is not None:
            if torch.is_tensor(value):
                self._normal = value
            else:
                self._normal = torch.tensor(value, dtype=torch.float32)
        else:
            self._normal = None

    def shade_illumination(self, illu):
        # set amplitude to the dot product of the normal and the illumination
        if self.normal is not None:
            if self.normal[2] < 0:
                self.amplitude = 0.0
            else:
                if illu is not None:
                    illu = illu / torch.norm(illu)
                    if illu.shape != self.normal.shape:
                        self.normal = self.normal.reshape(illu.shape)
                    self.amplitude = torch.dot(self.normal, illu) * self.amplitude
                if self.amplitude < 0:
                    self.amplitude = 0.0
                self.amplitude = (2 * self.amplitude + 0.1)
        return self.amplitude

    def to(self, *args, **kwargs):
        slf = super().to(*args, **kwargs)
        slf.position = slf.position.to(*args, **kwargs)
        if slf.normal is not None:
            slf.normal = slf.normal.to(*args, **kwargs)
        slf.amplitude = slf.amplitude.to(*args, **kwargs)
        slf.phase = slf.phase.to(*args, **kwargs)
        return slf

    def get_sh_color(self, direction=torch.tensor([0, 0, 1])):
        raise NotImplementedError("This method should be implemented in derived classes.")


class Point(Primitive):
    def __init__(
        self, pos, opacity, color=None, amplitude=1.0, phase=0.0, normal=None
    ):
        """
        Point primitive, always with zero orientation.

        Parameters
        ----------
        pos : torch.Tensor
            Position of the point in 3D space.
        opacity : float
            Opacity value.
        color : optional
            Color information.
        amplitude : float
            Amplitude of the wavefront contribution.
        phase : float
            Phase of the wavefront contribution.
        normal : torch.Tensor or None
            Optional normal vector.
        """
        super().__init__(pos, normal=normal, amplitude=amplitude, phase=phase)
        self.color = color
        self.opacity = opacity

    def get_wavefront_contribution(self):
        return self.amplitude * torch.exp(1j * self.phase)

    def z(self):
        return self.position[2]

    def get_sh_color(self, direction=torch.tensor([0, 0, 1])):
        return self.color


class Polygon(Primitive):
    def __init__(
        self,
        mean,
        opacity,
        amplitude=1.0,
        phase=0.0,
        normal=None,
        quat=None,
        color=None,
    ):
        """
        Polygon primitive with an affine transformation matrix.

        Parameters
        ----------
        mean : torch.Tensor
            3D vertex positions of the triangle, each as a 3D vector.
        amplitude : float
            Amplitude of the wavefront contribution.
        phase : float
            Phase of the wavefront contribution.
        normal : optional
            Normal vector.
        quat : optional
            Quaternion.
        color : optional
            Color tensor.
        """
        v1 = mean[0, ...]
        v2 = mean[1, ...]
        v3 = mean[2, ...]
        self.centroid = (v1 + v2 + v3) / 3
        super().__init__(self.centroid, amplitude=amplitude, phase=phase)

        v1 = v1.reshape(3, 1)
        v2 = v2.reshape(3, 1)
        v3 = v3.reshape(3, 1)
        self.X = torch.cat([v1, v2, v3], dim=1)
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        self.opacity = opacity

        if normal is None:
            self.normal = torch.cross(v2 - v3, v1 - v3)
            self.normal = self.normal / torch.norm(self.normal)
        else:
            self.normal = normal

        self.quat = quat
        self.color = color

    def z(self):
        return self.X[2:3, 0]

    def center_z_coord(self):
        z = self.z()
        self.X[2:3, :] = self.X[2:3, :]
        self.v1[2] -= z
        self.v2[2] -= z
        self.v3[2] -= z

    def compute_affine_matrix(self, canonical_points, actual_points):
        """
        Compute the affine transformation matrix.

        Parameters
        ----------
        canonical_points : torch.Tensor
            Canonical points of the polygon.
        actual_points : torch.Tensor
            Actual points after transformation.

        Returns
        -------
        affine_matrix : torch.Tensor
            The affine transformation matrix (2x3).
        """
        affine_matrix, _ = torch.lstsq(actual_points.t(), canonical_points.t())
        return affine_matrix[:2]

    def get_wavefront_contribution(self):
        return self.amplitude * torch.exp(1j * self.phase)

    def get_sh_color(self, direction=torch.tensor([0, 0, 1])):
        return torch.tensor(0.5).to(direction.device)

    def __name__(self):
        return "polygon"


class Gaussian(Primitive):
    def __init__(
        self,
        mean,
        opacity,
        quat,
        scale,
        sh0,
        shN=None,
        color=None,
        cov2d=None,
        amplitude=1.0,
        phase=0.0,
    ):
        """
        Gaussian primitive with spherical harmonics.

        Parameters
        ----------
        mean : torch.Tensor
            Position of the Gaussian disk in 3D space.
        opacity : float
            Opacity value.
        quat : torch.Tensor
            Quaternion.
        scale : torch.Tensor
            Scale.
        sh0 : torch.Tensor
            SH coefficient (constant term).
        shN : torch.Tensor, optional
            Higher SH coefficients.
        color : optional
            Color tensor.
        cov2d : torch.Tensor, optional
            2d covariance.
        amplitude : float
            Amplitude.
        phase : float
            Phase.
        """
        super().__init__(mean, amplitude=1.0, phase=phase)
        self.mean = mean
        self.opacity = opacity
        self.quat = quat
        self.scale = scale
        self.sh0 = sh0
        self.shN = shN
        self.device = mean.device
        self.cov2d = cov2d
        self.color = color

    def __name__(self):
        return "gaussian"

    def z(self):
        # so that the z coordinate is not changed
        return self.mean[..., 2].clone()

    def center_z_coord(self):
        self.mean[..., 2] = 0.0

    def get_sh_color(self, direction=torch.tensor([0, 0, 1])):
        """Evaluate color value using SH coefficients at a given direction in PyTorch."""
        sh0 = self.sh0
        shN = self.shN
        dev = sh0.device
        direction = direction.to(dev)
        direction = direction / torch.norm(direction)

        # Precompute SH basis for L = 0, 1, 2, 3
        sh_basis = utils.compute_sh_basis(3, direction)

        # Combine SH coefficients with SH basis
        if shN is None:
            sh_coeffs = sh0
        else:
            sh_coeffs = torch.cat([sh0, shN])
        sh_basis = torch.tensor(sh_basis[: len(sh_coeffs)]).to(sh_coeffs.device)
        color = torch.dot(sh_coeffs, sh_basis)
        return color

    def to(self, *args, **kwargs):
        slf = super().to(*args, **kwargs)
        slf.mean = slf.mean.to(*args, **kwargs)
        if slf.opacity is not None and torch.is_tensor(slf.opacity):
            slf.opacity = slf.opacity.to(*args, **kwargs)
        if slf.quat is not None and torch.is_tensor(slf.quat):
            slf.quat = slf.quat.to(*args, **kwargs)
        if slf.scale is not None and torch.is_tensor(slf.scale):
            slf.scale = slf.scale.to(*args, **kwargs)
        if slf.sh0 is not None and torch.is_tensor(slf.sh0):
            slf.sh0 = slf.sh0.to(*args, **kwargs)
        if slf.shN is not None and torch.is_tensor(slf.shN):
            slf.shN = slf.shN.to(*args, **kwargs)
        if slf.cov2d is not None and torch.is_tensor(slf.cov2d):
            slf.cov2d = slf.cov2d.to(*args, **kwargs)
        if slf.color is not None and torch.is_tensor(slf.color):
            slf.color = slf.color.to(*args, **kwargs)
        return slf


class Gaussians(Primitives):
    def __init__(
        self,
        means,
        opacities,
        quats,
        scales,
        sh0=None,
        shN=None,
        colors=None,
        cov2ds=None,
    ):
        super().__init__()
        self.means = means
        self.opacities = opacities
        self.quats = quats
        self.scales = scales
        self.sh0 = sh0
        self.shN = shN
        self.colors = colors
        self.cov2ds = cov2ds
        # add channel dimension if needed
        if self.means.dim() == 1:
            self.means = self.means.unsqueeze(0)
            self.opacities = self.opacities.unsqueeze(0)
            self.quats = self.quats.unsqueeze(0)
            self.scales = self.scales.unsqueeze(0)
            if self.sh0 is not None:
                self.sh0 = self.sh0.unsqueeze(0)
            if self.shN is not None:
                self.shN = self.shN.unsqueeze(0)
            if self.colors is not None:
                self.colors = self.colors.unsqueeze(0)
            if self.cov2ds is not None:
                self.cov2ds = self.cov2ds.unsqueeze(0)

    def to(self, *args, **kwargs):
        slf = super().to(*args, **kwargs)
        slf.means = slf.means.to(*args, **kwargs)
        slf.opacities = slf.opacities.to(*args, **kwargs)
        slf.quats = slf.quats.to(*args, **kwargs)
        slf.scales = slf.scales.to(*args, **kwargs)
        if slf.sh0 is not None and torch.is_tensor(slf.sh0):
            slf.sh0 = slf.sh0.to(*args, **kwargs)
        if slf.shN is not None and torch.is_tensor(slf.shN):
            slf.shN = slf.shN.to(*args, **kwargs)
        if slf.colors is not None and torch.is_tensor(slf.colors):
            slf.colors = slf.colors.to(*args, **kwargs)
        if slf.cov2ds is not None and torch.is_tensor(slf.cov2ds):
            slf.cov2ds = slf.cov2ds.to(*args, **kwargs)
        return slf

    def __iter__(self):
        """Iterate through the primitives in the collection."""
        if self.sh0 is None:
            self.sh0 = [None] * len(self.means)
        if self.shN is None:
            self.shN = [None] * len(self.means)
        if self.colors is None:
            self.colors = [None] * len(self.means)
        if self.cov2ds is None:
            self.cov2ds = [None] * len(self.means)

        for mean, opacity, quat, scale, sh0, shN, cov2d, color in zip(
            self.means,
            self.opacities,
            self.quats,
            self.scales,
            self.sh0,
            self.shN,
            self.cov2ds,
            self.colors,
        ):
            yield Gaussian(
                mean=mean,
                opacity=opacity,
                quat=quat,
                scale=scale,
                sh0=sh0,
                shN=shN,
                color=color,
                cov2d=cov2d,
            )

    def __getitem__(self, index):
        # Return new Gaussians object with only the selected indices
        means = self.means[index]
        opacities = self.opacities[index]
        quats = self.quats[index]
        scales = self.scales[index]
        sh0 = self.sh0[index] if self.sh0 is not None else None
        shN = self.shN[index] if self.shN is not None else None
        colors = self.colors[index] if self.colors is not None else None
        cov2ds = self.cov2ds[index] if self.cov2ds is not None else None
        return Gaussians(
            means=means,
            opacities=opacities,
            quats=quats,
            scales=scales,
            sh0=sh0,
            shN=shN,
            colors=colors,
            cov2ds=cov2ds,
        )

    def __name__(self):
        return "gaussians"

    def __len__(self):
        logger.debug(f"Debug: self.primitives has {len(self.means)} elements")
        return len(self.means)

    def sort(self, order="front2back"):
        """
        Sort the polygons based on a specified order.

        Args
        ----
            order (str): The sorting criterion.
                    Options:
                    - 'z_mean': Mean of the z-coordinates.
                    - 'opacity': Based on opacity values.
                    - 'normal_z': Based on the z-component of the normal vector.
        """
        if order == "back2front":
            key = self.means[..., 2]  # background to foreground
        elif order == "front2back":
            key = -self.means[..., 2]  # close to background
        elif order == "normal_z":
            raise ValueError(f"Unknown sorting order: {order}")
        else:
            raise ValueError(f"Unknown sorting order: {order}")

        sorted_indices = torch.argsort(key)
        self.means = self.means[sorted_indices]
        self.opacities = self.opacities[sorted_indices]
        self.quats = self.quats[sorted_indices]
        self.scales = self.scales[sorted_indices]
        if self.sh0 is not None and torch.is_tensor(self.sh0):
            self.sh0 = self.sh0[sorted_indices]
        if self.shN is not None and torch.is_tensor(self.shN):
            self.shN = self.shN[sorted_indices]
        if self.cov2ds is not None and torch.is_tensor(self.cov2ds):
            self.cov2ds = self.cov2ds[sorted_indices]
        if self.colors is not None and torch.is_tensor(self.colors):
            self.colors = self.colors[sorted_indices]

    def cull_elements(self, standard, threshold):
        if standard == "z_smaller":
            indices = self.means[..., 2] < threshold
        elif standard == "z_larger":
            indices = self.means[..., 2] > threshold
        elif standard == "outside_canvas":
            indices = self.means[..., 0].abs() < threshold[1]
            indices2 = self.means[..., 1].abs() < threshold[0]
            indices = indices & indices2
        elif standard == "large_angle":
            theta_x, theta_y, theta_z = utils.quaternion_to_euler_angles_zyx(
                self.quats
            )
            indice_x = theta_x.abs() < threshold
            indice_y = theta_y.abs() < threshold
            indices = indice_x & indice_y
        elif standard == "bbox":
            x = self.means[..., 0]
            y = self.means[..., 1]
            z = self.means[..., 2]
            indices = (
                (x >= -threshold)
                & (x <= threshold)
                & (y >= -threshold)
                & (y <= threshold)
                & (z >= -threshold)
                & (z <= threshold)
            )
        elif standard == "around_90":
            theta_x, theta_y, theta_z = utils.quaternion_to_euler_angles_zyx(
                self.quats
            )
            indices = (
                ((theta_x.abs() - 90 * math.pi / 180).abs() > threshold)
                & ((theta_y.abs() - 90 * math.pi / 180).abs() > threshold)
            )
        elif standard == "gsplat_culling":
            indices = threshold > 0
        elif standard == "small_scales":
            indices = (self.scales[:, :2] > threshold).all(dim=-1)
        else:
            raise ValueError(f"Unknown sorting order: {standard}")

        self.means = self.means[indices]
        self.opacities = self.opacities[indices]
        self.quats = self.quats[indices]
        self.scales = self.scales[indices]
        if self.sh0 is not None:
            self.sh0 = self.sh0[indices]
        if self.shN is not None:
            self.shN = self.shN[indices]
        if self.cov2ds is not None:
            self.cov2ds = self.cov2ds[indices]
        if self.colors is not None:
            self.colors = self.colors[indices]

    def remap_depth_range(
        self, depth_range, sigma=3, clamp_normalized_depth=True, original_depth_range=None
    ):
        logger.info(f"[Gaussians] Remapping {sigma} * IQR depth range to {depth_range}")
        if depth_range is None:
            z_mid = self.means[..., 2].median()
            self.means[..., 2] -= z_mid
        else:
            z_min, z_max = depth_range
            depth_values = self.means[..., 2]
            logger.info(f"[Gaussians] Depth range: {depth_values.min()} - {depth_values.max()}")

            if original_depth_range is None:
                Q1 = torch.quantile(depth_values, 0.25)
                Q3 = torch.quantile(depth_values, 0.75)
                IQR = Q3 - Q1
                logger.info(f"[Gaussians] IQR: {IQR}")

                lower_bound = Q1 - 0.5 * sigma * IQR
                upper_bound = Q3 + 0.5 * sigma * IQR

                non_outliers = depth_values[
                    (depth_values >= lower_bound) & (depth_values <= upper_bound)
                ]

                min_val = torch.min(non_outliers)
                max_val = torch.max(non_outliers)
                logger.info(f"[Gaussians] Non-outlier depth range: {min_val} - {max_val}")
            else:
                min_val, max_val = original_depth_range
                logger.info(f"[Gaussians] Using original depth range: {min_val} - {max_val}")

            normalized_depths = (depth_values - min_val) / (max_val - min_val)
            if clamp_normalized_depth:
                logger.info("[Gaussians] Clamping normalized depths")
                normalized_depths = torch.clamp(normalized_depths, 0, 1)
            else:
                logger.info("[Gaussians] Not clamping normalized depths")

            remapped_depths = normalized_depths * (z_max - z_min) + z_min

            self.means[..., 2] = remapped_depths
            self.min_depth = min_val
            self.max_depth = max_val

    def transform_perspective(self, K, pixel_pitch):
        logger.info("[Gaussians] Transforming gaussians to perspective pixel space")

        N = self.means.shape[0]
        fx = K[0, 0] * pixel_pitch
        fy = K[1, 1] * pixel_pitch

        self.scales[..., 2] = 0  # make sure Gaussians are 2D
        x, y, z = self.means[..., 0], self.means[..., 1], self.means[..., 2]
        z += 1e-8  # avoid division by zero

        J = torch.zeros((N, 2, 3), device=self.means.device, dtype=self.means.dtype)
        J[:, 0, 0] = fx / z
        J[:, 1, 1] = fy / z
        J[:, 0, 2] = -fx * x / z**2
        J[:, 1, 2] = -fy * y / z**2

        R = quaternion_to_matrix(self.quats)
        D = torch.diag_embed(self.scales ** 2)
        self.cov2ds = J @ R @ D @ R.mT @ J.mT

        scales_2d_sq, rotations_2d = torch.linalg.eigh(self.cov2ds)

        rotations_3d = torch.zeros((N, 3, 3), device=self.means.device, dtype=self.means.dtype)
        rotations_3d[:, :2, :2] = rotations_2d
        rotations_3d[:, 2, 2] = 1.0
        rotations_3d[torch.det(rotations_3d) < 0, :, 0] *= -1
        self.quats = matrix_to_quaternion(rotations_3d)
        self.quats = self.quats / torch.norm(self.quats, dim=-1, keepdim=True)

        scales_2d = torch.sqrt(scales_2d_sq)
        scales_3d = torch.zeros((N, 3), device=self.means.device, dtype=self.means.dtype)
        scales_3d[:, :2] = scales_2d
        self.scales = scales_3d

        self.means = torch.stack([fx * x / z, fy * y / z, z], dim=-1)

    def set_scales(self, scales):
        self.scales = scales

    def set_quats(self, quats):
        self.quats = quats

    def flip_z(self):
        self.means[..., 2] *= -1

    def zero_z(self):
        self.means[..., 2] = 0

    def sample_points(self, num_points):
        if num_points is None or num_points >= self.means.shape[0]:
            logger.info(
                "[Gaussians] Number of points to sample is greater than the number of Gaussians. Skipping sampling."
            )
            return list(range(self.means.shape[0]))
        np.random.seed(0)
        sampled_idx = np.random.choice(self.means.shape[0], num_points, replace=False)
        self.means = self.means[sampled_idx]
        self.opacities = self.opacities[sampled_idx]
        self.quats = self.quats[sampled_idx]
        self.scales = self.scales[sampled_idx]
        if self.sh0 is not None:
            self.sh0 = self.sh0[sampled_idx]
        if self.shN is not None:
            self.shN = self.shN[sampled_idx]
        if self.colors is not None:
            self.colors = self.colors[sampled_idx]
        if self.cov2ds is not None:
            self.cov2ds = self.cov2ds[sampled_idx]
        return sampled_idx


class Polygons(Primitives):
    def __init__(self, means, opacities, amplitudes, quats=None, colors=None):
        super().__init__()
        self.means = means  # [N, 3, 3] - vertices of the triangle
        self.opacities = opacities
        self.amplitudes = amplitudes
        self.normals = torch.linalg.cross(
            means[:, 1, ...] - means[:, 2, ...], means[:, 0, ...] - means[:, 2, ...]
        )
        self.normals = self.normals / torch.norm(self.normals, dim=-1, keepdim=True)
        self.quats = quats
        self.colors = colors

    def to(self, *args, **kwargs):
        slf = super().to(*args, **kwargs)
        slf.means = slf.means.to(*args, **kwargs)
        slf.opacities = slf.opacities.to(*args, **kwargs)
        slf.amplitudes = slf.amplitudes.to(*args, **kwargs)
        slf.normals = slf.normals.to(*args, **kwargs)
        if slf.quats is not None:
            slf.quats = slf.quats.to(*args, **kwargs)
        if slf.colors is not None:
            slf.colors = slf.colors.to(*args, **kwargs)
        return slf

    def __name__(self):
        return "polygons"

    def __len__(self):
        logger.debug(f"Debug: self.primitives has {len(self.means)} elements")
        return len(self.means)

    def __iter__(self):
        """Iterate through the primitives in the collection."""
        for mean, opacity, amplitude, normal, quat, color in zip(
            self.means, self.opacities, self.amplitudes, self.normals, self.quats, self.colors
        ):
            yield Polygon(
                mean, opacity, amplitude=amplitude, normal=normal, quat=quat, color=color
            )

    def __getitem__(self, index):
        self.means = self.means[index]
        self.opacities = self.opacities[index]
        self.amplitudes = self.amplitudes[index]
        if self.normals is not None:
            self.normals = self.normals[index]
        if self.quats is not None:
            self.quats = self.quats[index]
        if self.colors is not None:
            self.colors = self.colors[index]
        return self

    def z(self):
        return self.means[..., 2].mean(dim=1)

    def sort(self, order="front2back"):
        """
        Sort the polygons based on a specified order.

        Args
        ----
            order (str): The sorting criterion.
                    Options:
                    - 'z_mean': Mean of the z-coordinates.
                    - 'opacity': Based on opacity values.
                    - 'normal_z': Based on the z-component of the normal vector.
        """
        if order == "back2front":
            key = self.means[..., 2].mean(dim=1)
        elif order == "front2back":
            key = -self.means[..., 2].mean(dim=1)
        elif order == "opacity":
            key = self.opacities.squeeze()
        elif order == "normal_z":
            key = self.normals[..., 2]
        else:
            raise ValueError(f"Unknown sorting order: {order}")

        sorted_indices = torch.argsort(key)
        self.means = self.means[sorted_indices]
        self.opacities = self.opacities[sorted_indices]
        self.amplitudes = self.amplitudes[sorted_indices]
        self.normals = self.normals[sorted_indices]
        if self.quats is not None:
            self.quats = self.quats[sorted_indices]
        if self.colors is not None:
            self.colors = self.colors[sorted_indices]

    def cull_elements(self, standard, threshold):
        if standard == "z_smaller":
            indices = self.means[..., 2].mean(dim=-1) < threshold
        elif standard == "z_larger":
            indices = self.means[..., 2].mean(dim=-1) > threshold
        elif standard == "outside_canvas":
            indices = self.means[..., 0].mean(dim=-1) < threshold[1]
            indices2 = self.means[..., 1].mean(dim=-1) < threshold[0]
            indices = indices & indices2
        elif standard == "large_angle":
            theta_x, theta_y, theta_z = utils.quaternion_to_euler_angles_zyx(self.quats)
            indice_x = theta_x.abs() < threshold
            indice_y = theta_y.abs() < threshold
            indices = indice_x & indice_y
        elif standard == "around_90":
            theta_x, theta_y, theta_z = utils.quaternion_to_euler_angles_zyx(self.quats)
            indices = (
                ((theta_x.abs() - 90 * math.pi / 180).abs() > threshold)
                & ((theta_y.abs() - 90 * math.pi / 180).abs() > threshold)
            )
        elif standard == "gsplat_culling":
            indices = threshold > 0
        else:
            raise ValueError(f"Unknown sorting order: {standard}")

        self.means = self.means[indices]
        self.opacities = self.opacities[indices]
        self.amplitudes = self.amplitudes[indices]
        self.normals = self.normals[indices]
        if self.quats is not None:
            self.quats = self.quats[indices]
        if self.colors is not None:
            self.colors = self.colors[indices]

    def reset_normals(self):
        self.normals = torch.linalg.cross(
            self.means[:, 1, ...] - self.means[:, 2, ...],
            self.means[:, 0, ...] - self.means[:, 2, ...],
        )
        self.normals = self.normals / torch.norm(self.normals, dim=-1, keepdim=True)

    def remap_depth_range(self, depth_range, sigma=3, original_depth_range=None):
        logger.info(f"[Polygons] Remapping {sigma} * IQR depth range to {depth_range}")

        if depth_range is None:
            z_mid = self.means[..., 2].median()
            self.means[..., 2] -= z_mid
        else:
            z_min, z_max = depth_range
            depth_values = self.means[..., 2].view(-1)
            logger.info(f"[Polygons] Depth range: {depth_values.min()} - {depth_values.max()}")

            if original_depth_range is None:
                Q1 = torch.quantile(depth_values, 0.25)
                Q3 = torch.quantile(depth_values, 0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 0.5 * sigma * IQR
                upper_bound = Q3 + 0.5 * sigma * IQR

                non_outliers = depth_values[
                    (depth_values >= lower_bound) & (depth_values <= upper_bound)
                ]
                min_val = torch.min(non_outliers)
                max_val = torch.max(non_outliers)
            else:
                min_val, max_val = original_depth_range
                logger.info(f"[Polygons] Using original depth range: {min_val} - {max_val}")

            eps = 1e-5
            normalized_depths = torch.zeros_like(depth_values)

            mask_middle = (depth_values > min_val) & (depth_values < max_val)
            mask_small = depth_values < min_val
            mask_large = depth_values > max_val

            normalized_depths[mask_middle] = utils.scale_to_range(
                depth_values[mask_middle], eps, 1 - eps
            )
            if depth_values.min() < min_val:
                normalized_depths[mask_small] = utils.scale_to_range(
                    depth_values[mask_small], 0, eps
                )
            if depth_values.max() > max_val:
                normalized_depths[mask_large] = utils.scale_to_range(
                    depth_values[mask_large], 1 - eps, 1
                )
            remapped_depths = normalized_depths * (z_max - z_min) + z_min
            self.means[..., 2] = remapped_depths.view(-1, 3)
        self.reset_normals()

    def transform_perspective(self, K, pixel_pitch):
        logger.info("[Polygons] Transforming gaussians to perspective pixel space")

        fx = K[0, 0] * pixel_pitch
        fy = K[1, 1] * pixel_pitch
        x, y, z = self.means[..., 0], self.means[..., 1], self.means[..., 2]
        z += 1e-8

        self.means = torch.stack(
            [fx * x / z, fy * y / z, z],
            dim=-1,
        )

        self.reset_normals()

    def flip_z(self):
        self.means[..., 2] *= -1
        self.reset_normals()

    def zero_z(self):
        self.means[..., 2] = 0
        self.reset_normals()


class Points(Primitives):
    def __init__(
        self,
        means,
        opacities,
        phases=None,
        colors=None,
        sh0=None,
        shN=None,
        quats=None,
        scales=None,
        normals=None,
    ):
        super().__init__()
        self.means = means
        self.opacities = opacities
        self.phases = phases
        self.normals = normals
        self.colors = colors

    def to(self, *args, **kwargs):
        slf = super().to(*args, **kwargs)
        slf.means = slf.means.to(*args, **kwargs)
        slf.opacities = slf.opacities.to(*args, **kwargs)
        if slf.phases is not None:
            slf.phases = self.phases.to(*args, **kwargs)
        if slf.normals is not None:
            slf.normals = slf.normals.to(*args, **kwargs)
        if slf.colors is not None:
            slf.colors = slf.colors.to(*args, **kwargs)
        return slf

    def __name__(self):
        return "points"

    def __len__(self):
        return len(self.means)

    def __iter__(self):
        """Iterate through the primitives in the collection."""
        for mean, opacity, phase, color in zip(
            self.means, self.opacities, self.phases, self.colors
        ):
            yield Point(mean, opacity, color=color, phase=phase)

    def cull_elements(self, standard, threshold):
        if standard == "z_smaller":
            indices = self.means[..., 2] < threshold
        elif standard == "z_larger":
            indices = self.means[..., 2] > threshold
        elif standard == "outside_canvas":
            indices = self.means[..., 0].abs() < threshold[1]
            indices2 = self.means[..., 1].abs() < threshold[0]
            indices = indices & indices2
        elif standard == "large_angle":
            theta_x, theta_y, theta_z = utils.quaternion_to_euler_angles_zyx(
                self.quats
            )
            indice_x = theta_x.abs() < threshold
            indice_y = theta_y.abs() < threshold
            indices = indice_x & indice_y
        elif standard == "bbox":
            x = self.means[..., 0]
            y = self.means[..., 1]
            z = self.means[..., 2]
            indices = (
                (x >= -threshold)
                & (x <= threshold)
                & (y >= -threshold)
                & (y <= threshold)
                & (z >= -threshold)
                & (z <= threshold)
            )
        elif standard == "around_90":
            return
        elif standard == "gsplat_culling":
            indices = threshold > 0
        elif standard == "small_scales":
            indices = (self.scales[:, :2] > threshold).all(dim=-1)
        else:
            raise ValueError(f"Unknown sorting order: {standard}")

        self.means = self.means[indices]
        self.opacities = self.opacities[indices]
        if self.phases is not None:
            self.phases = self.phases[indices]
        if self.colors is not None:
            self.colors = self.colors[indices]

    def remap_depth_range(self, depth_range, sigma=3, original_depth_range=None):
        logger.info(f"[Points] Remapping {sigma} * IQR depth range to {depth_range}")
        if depth_range is None:
            z_mid = self.means[..., 2].median()
            self.means[..., 2] -= z_mid
        else:
            z_min, z_max = depth_range
            depth_values = self.means[..., 2]
            logger.info(f"[Points] Depth range: {depth_values.min()} - {depth_values.max()}")

            if original_depth_range is None:
                Q1 = torch.quantile(depth_values, 0.25)
                Q3 = torch.quantile(depth_values, 0.75)
                IQR = Q3 - Q1
                logger.info(f"[Points] IQR: {IQR}")

                lower_bound = Q1 - 0.5 * sigma * IQR
                upper_bound = Q3 + 0.5 * sigma * IQR

                non_outliers = depth_values[
                    (depth_values >= lower_bound) & (depth_values <= upper_bound)
                ]

                min_val = torch.min(non_outliers)
                max_val = torch.max(non_outliers)
            else:
                min_val, max_val = original_depth_range

            eps = 1e-5
            normalized_depths = torch.zeros_like(depth_values)

            mask_middle = (depth_values > min_val) & (depth_values < max_val)
            mask_small = depth_values < min_val
            mask_large = depth_values > max_val

            normalized_depths[mask_middle] = utils.scale_to_range(
                depth_values[mask_middle], eps, 1 - eps
            )
            if depth_values.min() < min_val:
                normalized_depths[mask_small] = utils.scale_to_range(
                    depth_values[mask_small], 0, eps
                )
            if depth_values.max() > max_val:
                normalized_depths[mask_large] = utils.scale_to_range(
                    depth_values[mask_large], 1 - eps, 1
                )
            remapped_depths = normalized_depths * (z_max - z_min) + z_min
            self.means[..., 2] = remapped_depths

    def set_zero_phase(self, wavelength):
        # Set phases of points to be zero on the SLM plane
        z = self.means[:, -1]
        self.phases = -2 * math.pi * z / wavelength

    def transform_perspective(self, K, pixel_pitch):
        logger.info("[Points] Transforming gaussians to perspective pixel space")

        fx = K[0, 0] * pixel_pitch
        fy = K[1, 1] * pixel_pitch
        x, y, z = self.means[..., 0], self.means[..., 1], self.means[..., 2]
        z += 1e-8

        self.means = torch.stack([fx * x / z, fy * y / z, z], dim=-1)

    def flip_z(self):
        self.means[..., 2] *= -1

    def zero_z(self):
        self.means[..., 2] = 0

    @property
    def z(self):
        return self.means[..., 2]

    def add_points_batch(self, positions, amplitudes, phases):
        for pos, amp, phase in zip(positions, amplitudes, phases):
            self.primitives.append(Point(pos, amplitude=amp.item(), phase=phase.item()))

    def sample_points(self, num_points):
        if num_points is None or num_points >= self.means.shape[0]:
            logger.info(
                "[Gaussians] Number of points to sample is greater than the number of Gaussians. Skipping sampling."
            )
            return list(range(self.means.shape[0]))
        np.random.seed(0)
        sampled_idx = np.random.choice(self.means.shape[0], num_points, replace=False)
        self.means = self.means[sampled_idx]
        self.opacities = self.opacities[sampled_idx]
        if self.phases is not None:
            self.phases = self.phases[sampled_idx]
        if self.colors is not None:
            self.colors = self.colors[sampled_idx]
        if self.normals is not None:
            self.normals = self.normals[sampled_idx]
        return sampled_idx