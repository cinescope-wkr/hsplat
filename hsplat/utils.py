"""
Utility functions

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
import imageio

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import logging

logger = logging.getLogger(__name__)

BLENDER_SCENES = ["lego", "chair", "hotdog", "ficus", "mic", "materials", "ship", "drums"]
MIPNERF360_SCENES = ["bicycle", "counter", "garden", "room", "bonsai", "stump", "kitchen"]
NUM_TRIANGLES = {
    "chair": 107061,
    "hotdog": 111380,
    "drums": 192989,
    "ficus": 178844,
    "materials": 123464,
    "mic": 116698,
    "ship": 216055,
    "lego": 224486,
    "bicycle": 865949,
    "counter": 688451,
    "garden": 771393,
    "kitchen": 574235,
    "stump": 820855,
    "bonsai": 636871,
    "room": 546218
}

def pad_image(field, target_shape, pytorch=True, stacked_complex=False, padval=0, mode='constant', lf=False):
    """Pads a 2D complex field up to target_shape in size

    Padding is done such that when used with crop_image(), odd and even dimensions are
    handled correctly to properly undo the padding.

    field: the field to be padded. May have as many leading dimensions as necessary
        (e.g., batch or channel dimensions)
    target_shape: the 2D target output dimensions. If any dimensions are smaller
        than field, no padding is applied
    pytorch: if True, uses torch functions, if False, uses numpy
    stacked_complex: for pytorch=True, indicates that field has a final dimension
        representing real and imag
    padval: the real number value to pad by
    mode: padding mode for numpy or torch
    """
    if lf:
        size_diff = np.array(target_shape) - np.array(field.shape[-4:-2])
        odd_dim = np.array(field.shape[-4:-2]) % 2
    else:
        if pytorch:
            if stacked_complex:
                size_diff = np.array(target_shape) - np.array(field.shape[-3:-1])
                odd_dim = np.array(field.shape[-3:-1]) % 2
            else:
                size_diff = np.array(target_shape) - np.array(field.shape[-2:])
                odd_dim = np.array(field.shape[-2:]) % 2
        else:
            size_diff = np.array(target_shape) - np.array(field.shape[-2:])
            odd_dim = np.array(field.shape[-2:]) % 2

    # pad the dimensions that need to increase in size
    if (size_diff > 0).any():
        pad_total = np.maximum(size_diff, 0)
        pad_front = (pad_total + odd_dim) // 2
        pad_end = (pad_total + 1 - odd_dim) // 2

        if pytorch:
            pad_axes = [int(p)  # convert from np.int64
                        for tple in zip(pad_front[::-1], pad_end[::-1])
                        for p in tple]
            if lf:
                original_shape = field.shape
                return nn.functional.pad(field.permute(0, 1, 4, 5, 2, 3).view(-1, 1, *original_shape[-4:-2]), pad_axes, mode=mode, value=padval).reshape(*original_shape[:2], *original_shape[-2:], *target_shape).permute(0, 1, 4, 5, 2, 3)
            else:
                if stacked_complex:
                    return pad_stacked_complex(field, pad_axes, mode=mode, padval=padval)
                else:
                    return nn.functional.pad(field, pad_axes, mode=mode, value=padval)
        else:
            leading_dims = field.ndim - 2  # only pad the last two dims
            if leading_dims > 0:
                pad_front = np.concatenate(([0] * leading_dims, pad_front))
                pad_end = np.concatenate(([0] * leading_dims, pad_end))
            return np.pad(field, tuple(zip(pad_front, pad_end)), mode,
                          constant_values=padval)
    else:
        return field



def crop_image(field, target_shape, pytorch=True, stacked_complex=False, lf=False):
    """Crops a 2D field, see pad_image() for details
    No cropping is done if target_shape is already smaller than field
    """
    if target_shape is None:
        return field

    if lf:
        size_diff = np.array(field.shape[-4:-2]) - np.array(target_shape)
        odd_dim = np.array(field.shape[-4:-2]) % 2
    else:
        if pytorch:
            if stacked_complex:
                size_diff = np.array(field.shape[-3:-1]) - np.array(target_shape)
                odd_dim = np.array(field.shape[-3:-1]) % 2
            else:
                size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
                odd_dim = np.array(field.shape[-2:]) % 2
        else:
            size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
            odd_dim = np.array(field.shape[-2:]) % 2

    # crop dimensions that need to decrease in size
    if (size_diff > 0).any():
        crop_total = np.maximum(size_diff, 0)
        crop_front = (crop_total + 1 - odd_dim) // 2
        crop_end = (crop_total + odd_dim) // 2

        crop_slices = [slice(int(f), int(-e) if e else None)
                       for f, e in zip(crop_front, crop_end)]
        if lf:
            return field[(..., *crop_slices, slice(None), slice(None))]
        else:
            if pytorch and stacked_complex:
                return field[(..., *crop_slices, slice(None))]
            else:
                return field[(..., *crop_slices)]
    else:
        return field


def grid_sample_complex(input_field, grid, align_corners=False, mode='bilinear', coord='rect'):
    if coord == 'rect':
        return F.grid_sample(input_field.real, grid, align_corners=align_corners, mode=mode) + 1j * F.grid_sample(input_field.imag, grid, align_corners=align_corners, mode=mode)
    elif coord == 'polar':
        return F.grid_sample(input_field.abs(), grid, align_corners=align_corners, mode=mode) * torch.exp(1j * F.grid_sample(input_field.angle(), grid, align_corners=align_corners, mode=mode))


def coordinate_rotation_matrix(axis, angle, keep_channel_dim=False):
    """
    Compute rotation matrices for a given axis and angles.
    This rotates coordinates, not the object itself.

    axis: 'x', 'y', or 'z' (rotation axis)
    angle: A scalar tensor or a tensor with shape (N,) (rotation angles)

    Returns:
    A rotation matrix of shape (3, 3) if angle is scalar,
    or (N, 3, 3) if angle is a batch of angles.
    """
    if not torch.is_tensor(angle):
        angle = torch.tensor(angle, dtype=torch.float32)

    # Ensure angles are at least 1D for consistent batching
    angle = angle.unsqueeze(0) if angle.ndim == 0 else angle  # Shape: (N,)

    # Compute cos and sin of the angles
    cos = torch.cos(angle)  # Shape: (N,) or scalar
    sin = torch.sin(angle)  # Shape: (N,) or scalar

    if axis == 'x':
        rotation_matrices = torch.stack([
            torch.stack([torch.ones_like(cos), torch.zeros_like(cos), torch.zeros_like(cos)], dim=-1),
            torch.stack([torch.zeros_like(cos), cos, -sin], dim=-1),
            torch.stack([torch.zeros_like(cos), sin, cos], dim=-1)
        ], dim=-2)  # Shape: (N, 3, 3)
    elif axis == 'y':
        rotation_matrices = torch.stack([
            torch.stack([cos, torch.zeros_like(cos), sin], dim=-1),
            torch.stack([torch.zeros_like(cos), torch.ones_like(cos), torch.zeros_like(cos)], dim=-1),
            torch.stack([-sin, torch.zeros_like(cos), cos], dim=-1)
        ], dim=-2)  # Shape: (N, 3, 3)
    elif axis == 'z':
        rotation_matrices = torch.stack([
            torch.stack([cos, -sin, torch.zeros_like(cos)], dim=-1),
            torch.stack([sin, cos, torch.zeros_like(cos)], dim=-1),
            torch.stack([torch.zeros_like(cos), torch.zeros_like(cos), torch.ones_like(cos)], dim=-1)
        ], dim=-2)  # Shape: (N, 3, 3)
    else:
        raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")

    # If the input angle is a scalar, return a 3x3 matrix; otherwise, return (N, 3, 3)
    if keep_channel_dim:
        return rotation_matrices.to(torch.get_default_dtype())
    else:
        return rotation_matrices.squeeze(0).to(torch.get_default_dtype())



def normalized_quat_to_rotmat(quat: Tensor) -> Tensor:
    """Convert normalized quaternion to rotation matrix.

    Args:
        quat: Normalized quaternion in wxyz convension. (..., 4)

    Returns:
        Rotation matrix (..., 3, 3)
    """
    assert quat.shape[-1] == 4, quat.shape
    w, x, y, z = torch.unbind(quat, dim=-1)
    mat = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return mat.reshape(quat.shape[:-1] + (3, 3))


def quaternion_to_euler_angles(q):
    """
    Convert a quaternion to Euler angles (in radians) using the ZYX order.
    Returns angles (theta_z, theta_y, theta_x).
    """
    w, x, y, z = q

    # Rotate X, Y, Z order
    # Calculate the Euler angles
    theta_z = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    theta_y = torch.asin(torch.clamp(2 * (w * y - x * z), -1.0, 1.0))  # Clamp to avoid numerical errors
    theta_x = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    
    return theta_x, theta_y, theta_z


def quaternion_to_euler_angles_zyx(q, flip_angles_for_large_x=True):
    """
    Convert a quaternion to Euler angles (in radians) using the ZYX order.
    Returns angles (theta_z, theta_y, theta_x).
    """
    if len(q.shape) == 2:
        w = q[..., 0]
        x = q[..., 1]
        y = q[..., 2]
        z = q[..., 3]
    else:
        w, x, y, z = q

    # Rotate Z, Y, X order (Q = Rx @ Ry @ Rz)
    # Calculate the Euler angles
    eps = 3e-4
    theta_z = torch.atan2(- 2 * (x * y - w * z), 1 - 2 * (y**2 + z**2))
    theta_y = torch.asin(torch.clamp(2 * (w * y + x * z), -1.0+eps, 1.0-eps))  # Clamp to avoid numerical errors
    theta_x = torch.atan2(- 2 * (y * z - w * x), 1 - 2 * (x**2 + y**2))
        
    if flip_angles_for_large_x:
        # flip the angles for x over 90 degrees but make sure that this results in the same rotation ...
        if theta_x.ndim == 0:
            if theta_x > 90 * np.pi / 180:
                theta_x = theta_x - 180 * np.pi / 180
                theta_y = -theta_y
                theta_z = -theta_z
            elif theta_x < -90 * np.pi / 180:
                theta_x = theta_x + 180 * np.pi / 180
                theta_y = -theta_y
                theta_z = -theta_z
        else:
            flip_mask_1 = theta_x > 90 * np.pi / 180
            theta_x[flip_mask_1] = theta_x[flip_mask_1] - 180 * np.pi / 180
            theta_y[flip_mask_1] = -theta_y[flip_mask_1]
            theta_z[flip_mask_1] = -theta_z[flip_mask_1]

            flip_mask_2 = theta_x < -90 * np.pi / 180
            theta_x[flip_mask_2] = theta_x[flip_mask_2] + 180 * np.pi / 180
            theta_y[flip_mask_2] = -theta_y[flip_mask_2]
            theta_z[flip_mask_2] = -theta_z[flip_mask_2]

    return theta_x, theta_y, theta_z


def normalize_and_write(name, a, max_val=None, min_val=None):
    # normalize the absolute value of a and write to file
    b = a.abs()
    if max_val is None and min_val is None:
        imageio.imwrite(name, (((b - b.min()).squeeze().cpu().detach().numpy() / (b.max() - b.min()).squeeze().cpu().detach().numpy()) * 255).astype(np.uint8))
    else:
        imageio.imwrite(name, (((b - min_val).squeeze().cpu().detach().numpy() / (max_val - min_val)) * 255).astype(np.uint8))


def make_freq_grid(cfg):
    wavelength = cfg.wavelength
    pixel_pitch = cfg.pixel_pitch
    ny = cfg.pad_n * cfg.resolution_hologram[0]  # this was n_pad * out_resolution_hologram
    nx = cfg.pad_n * cfg.resolution_hologram[1]
    dfx = 1 / pixel_pitch / nx
    dfy = 1 / pixel_pitch / ny

    fx = torch.linspace(-(nx-1)/2, (nx-1)/2, nx, device=cfg.dev) * dfx
    fy = torch.linspace(-(ny-1)/2, (ny-1)/2, ny, device=cfg.dev) * dfy
    fx, fy = torch.meshgrid(fx, fy)
    fx = torch.transpose(fx, 0, 1)
    fy = torch.transpose(fy, 0, 1)  
    fz = torch.sqrt((1/wavelength)**2 - fx**2 - fy**2)
    fz = torch.where(torch.isnan(fz), torch.ones_like(fz)*1e-12, fz)  # replace nan with 0
    return fx, fy, fz

# Transform f -> f_local
def rotate_frequency_grid(R, fx, fy, fz=None, wavelength=None):
    # rotate the frequency grid
    if fz is None:
        # 2D case
        flx = R[0, 0] * fx + R[0, 1] * fy
        fly = R[1, 0] * fx + R[1, 1] * fy
        return flx, fly
    else:
        flx = R[0, 0] * fx + R[0, 1] * fy + R[0, 2] * fz
        fly = R[1, 0] * fx + R[1, 1] * fy + R[1, 2] * fz
        flz = torch.sqrt(((1 / wavelength) ** 2 - flx ** 2 - fly ** 2))
        flz = torch.where(torch.isnan(flz), torch.ones_like(flz)*1e-12, flz)  # replace nan with 0
        return flx, fly, flz
    

def get_rotation_matrix(n):
    # Compute rotation angles theta and phi from the normal vector
    if n[2] < 0: 
        n *= -1
        # logger.debug('flipping normal vector')
    else:
        # logger.debug('normal vector is already facing up')
        pass
    
    if n[2] == 0:
        theta = 0
    else:
        theta = torch.atan(n[0] / n[2])

    phi = torch.atan(n[1] / torch.sqrt(n[0] ** 2 + n[2] ** 2))

    # Rotation matrices (See utils.py)
    Rx = coordinate_rotation_matrix('x', phi).to(n.device)
    Ry = coordinate_rotation_matrix('y', -theta).to(n.device)

    # Rotate around y first, then x
    return Rx @ Ry



def ifft(a):
    return torch.fft.ifftshift(torch.fft.ifftn(torch.fft.fftshift(a, dim=(-2, -1)), dim=(-2, -1), norm='ortho'), dim=(-2, -1))

def fft(a, norm='ortho'):
    return torch.fft.ifftshift(torch.fft.fftn(torch.fft.fftshift(a, dim=(-2, -1)), dim=(-2, -1), norm=norm), dim=(-2, -1))  # TODO: check shifts

def conv(a, b, norm='ortho'):
    Fa = ifft(a)
    Fb = ifft(b)        
    return fft(Fa * Fb, norm=norm)

def factorial(n):
    """Compute the factorial of n."""
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

def legendre_polynomial(l, m, x):
    """
    Compute the associated Legendre polynomial P_l^m(x).
    Parameters:
    - l: Degree (non-negative integer).
    - m: Order (-l <= m <= l).
    - x: Input value (float or tensor in [-1, 1]).
    """
    pmm = 1.0
    if m > 0:
        pmm = (-1)**m * factorial(2*m) / (2**m * factorial(m)) * (1 - x**2)**(m / 2)

    if l == m:
        return pmm

    pmmp1 = x * (2 * m + 1) * pmm
    if l == m + 1:
        return pmmp1

    # Compute P_l^m(x) iteratively
    pll = pmmp1
    for ll in range(m + 2, l + 1):
        pll = ((2 * ll - 1) * x * pll - (ll + m - 1) * pmm) / (ll - m)

    return pll

def compute_sh_basis(L, direction):
    """
    Compute all spherical harmonics basis functions up to order L for a given direction.
    
    Parameters:
    - L: Maximum degree of SH.
    - direction: A 3D unit vector (torch tensor).

    Returns:
    - A tensor containing all SH basis values up to degree L.
    """
    x, y, z = direction
    r = torch.norm(direction)
    theta = torch.acos(z / r)  # Polar angle
    phi = torch.atan2(y, x)    # Azimuthal angle

    sh_basis = []
    for l in range(L + 1):
        for m in range(-l, l + 1):
            # Compute the normalization constant
            K_lm = math.sqrt((2 * l + 1) / (4 * math.pi) * factorial(l - abs(m)) / factorial(l + abs(m)))
            # Compute the associated Legendre polynomial
            P_lm = legendre_polynomial(l, abs(m), torch.cos(theta))
            # Apply the azimuthal factor
            if m < 0:
                Y_lm = math.sqrt(2) * K_lm * P_lm * torch.sin(-m * phi)
            elif m == 0:
                Y_lm = K_lm * P_lm
            else:
                Y_lm = math.sqrt(2) * K_lm * P_lm * torch.cos(m * phi)
            sh_basis.append(Y_lm)

    return torch.tensor(sh_basis).to(direction.device)


def rotation_matrix_to_quaternion(R):
    """
    Converts a rotation matrix to a quaternion.
    R: A 3x3 rotation matrix (torch.Tensor)
    Returns:
    A quaternion (w, x, y, z) as a torch.Tensor
    """
    trace = torch.trace(R)
    if trace > 0:
        w = torch.sqrt(1.0 + trace) / 2.0
        x = (R[2, 1] - R[1, 2]) / (4.0 * w)
        y = (R[0, 2] - R[2, 0]) / (4.0 * w)
        z = (R[1, 0] - R[0, 1]) / (4.0 * w)
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            x = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) / 2.0
            w = (R[2, 1] - R[1, 2]) / (4.0 * x)
            y = (R[0, 1] + R[1, 0]) / (4.0 * x)
            z = (R[0, 2] + R[2, 0]) / (4.0 * x)
        elif R[1, 1] > R[2, 2]:
            y = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) / 2.0
            w = (R[0, 2] - R[2, 0]) / (4.0 * y)
            x = (R[0, 1] + R[1, 0]) / (4.0 * y)
            z = (R[1, 2] + R[2, 1]) / (4.0 * y)
        else:
            z = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) / 2.0
            w = (R[1, 0] - R[0, 1]) / (4.0 * z)
            x = (R[0, 2] + R[2, 0]) / (4.0 * z)
            y = (R[1, 2] + R[2, 1]) / (4.0 * z)

    # Normalize the quaternion
    quaternion = torch.tensor([w, x, y, z])
    quaternion = quaternion / torch.norm(quaternion)

    return quaternion.to(R.device)


def quaternion_multiply(q1, q2):
    """
    Multiplies two quaternions q1 and q2.
    q1 and q2 are tensors of shape (4,), representing (w, x, y, z).
    Returns:
    A quaternion q3 = q1 * q2 as a tensor of shape (4,).
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w3 = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x3 = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y3 = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z3 = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.tensor([w3, x3, y3, z3]).to(q1.device)


def save_video(amps, out_folder):
    """Create and save video from amplitude images
    Args:
        amps (list): List of amplitude images
        out_folder (str): Output folder path
    """
    # Stack and normalize amplitudes
    amps, color = decode_dict_to_tensor(amps, order='bgr') # for cv2 saving order
    # amps = torch.stack(amps)
    logger.info(f"{amps.shape} amps.shape")
    
    # Convert grayscale to RGB by repeating channels
    if color is not None and color == 'color':
        video_data = (amps * 255).clamp(0, 255).squeeze().permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        logger.info(f"{video_data.shape} video_data.shape")
    else:
        video_data = (amps * 255).clamp(0, 255).squeeze().cpu().numpy().astype(np.uint8)
        video_data = np.repeat(video_data[..., np.newaxis], 3, axis=-1)
        logger.info(f"{video_data.shape} video_data.shape")
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f'{out_folder}/propagation_video.mp4', fourcc, 25,
                            (video_data.shape[2], video_data.shape[1]))
    
    # Write frames
    for frame in video_data:
        video.write(frame)
    video.release()

def save_focal_stack(amps, out_folder):
    """Save focal stack as indepdendent frames not video
    Args:
        amps (list): List of amplitude images
        out_folder (str): Output folder path
    """
    out_folder = os.path.join(out_folder, "focal_stack")
    os.makedirs(out_folder, exist_ok=True)

    # Stack and normalize amplitudes
    amps, color = decode_dict_to_tensor(amps)
    # amps = torch.stack(amps)
    logger.info(f"{amps.shape} amps.shape")
    
    # Convert grayscale to RGB by repeating channels
    if color is not None and color == 'color':
        fs_data = (amps * 255).clamp(0, 255).squeeze().permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        logger.info(f"{fs_data.shape} video_data.shape")
    else:
        fs_data = (amps * 255).clamp(0, 255).squeeze().cpu().numpy().astype(np.uint8)
        fs_data = np.repeat(fs_data[..., np.newaxis], 3, axis=-1)
        logger.info(f"{fs_data.shape} fs_data.shape")

    for fs_id, fs in enumerate(fs_data):
        fs_id = str(fs_id).zfill(4)
        imageio.imwrite(os.path.join(out_folder, f"{fs_id}.png"), fs)

def decode_dict_to_tensor(data, order='rgb'):
    if isinstance(data, dict):
        if order == 'bgr':
            keyss = reversed(data.keys())
        else:
            keyss = data.keys()
        return torch.cat([data[key] for key in keyss], dim=1), 'color' if len(data.keys()) == 3 else list(data.keys())[0]
    else:
        return data, None
    
def gsplat_projection_2dgs(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    viewmats: Tensor,  # [C, 4, 4]
):
    from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
    """PyTorch implementation of `gsplat.cuda._wrapper.fully_fused_projection_2dgs()`

    .. note::

        This is a minimal implementation of fully fused version, which has more
        arguments. Not all arguments are supported.
    """
    R_cw = viewmats[:, :3, :3]  # [C, 3, 3]
    t_cw = viewmats[:, :3, 3]  # [C, 3]
    means_c = torch.einsum("cij,nj->cni", R_cw, means) + t_cw[:, None, :]  # (C, N, 3)

    qR = quaternion_to_matrix(quats)  # [N, 3, 3]
    qR_c = torch.einsum("cij,njk->cnik", R_cw, qR)  # [C, N, 3, 3]
    quats_c = matrix_to_quaternion(qR_c)  # [C, N, 4]
    return means_c, quats_c


def get_gaussian_amplitude(fx, fy):
    sigma = max(fx.max(), fy.max()) / 2
    gau = torch.exp(-(fx**2 + fy**2) / 2 / sigma**2)
    return gau


def im2float(im, dtype=np.float32, im_max=None):
    """convert uint16 or uint8 image to float32, with range scaled to 0-1

    :param im: image
    :param dtype: default np.float32
    :return:
    """
    if issubclass(im.dtype.type, np.floating):
        return im.astype(dtype)
    elif issubclass(im.dtype.type, np.integer):
        if im_max is not None:
            return im / im_max
        else:
            return im / dtype(np.iinfo(im.dtype).max)
    else:
        raise ValueError(f'Unsupported data type {im.dtype}')
    
    
def compute_quaternions_from_triangles(triangles):
    """
    Compute quaternions for a list of triangles to align their normals to the z-axis.

    :param triangles: Tensor of shape [N, 3, 3] representing N triangles with 3 vertices in 3D.
    :return: Tensor of shape [N, 4] representing quaternions (w, x, y, z).
    """
    # Step 1: Compute the normals
    edges1 = triangles[:, 1] - triangles[:, 0]
    edges2 = triangles[:, 2] - triangles[:, 0]
    normals = torch.linalg.cross(edges1, edges2) # Shape [N, 3]

    # Step 2: Normalize the normals
    norms = torch.norm(normals, dim=1, keepdim=True)
    normals_unit = normals / (norms + 1e-8)  # Add epsilon to avoid division by zero

    # Step 3: Compute the rotation axis
    z_axis = torch.tensor([0, 0, 1], dtype=triangles.dtype, device=triangles.device).view(1, 3)
    rotation_axes = torch.cross(normals_unit, z_axis.expand_as(normals_unit), dim=1)

    # Step 4: Compute the rotation angle
    dot_products = torch.einsum('ij,j->i', normals_unit, z_axis[0])  # Dot product with z-axis
    angles = torch.acos(torch.clamp(dot_products, -1.0, 1.0))  # Ensure values are within valid range

    # Step 5: Normalize rotation axes
    axis_norms = torch.norm(rotation_axes, dim=1, keepdim=True)
    rotation_axes = rotation_axes / (axis_norms + 1e-8)

    # Step 6: Compute quaternions
    half_angles = angles / 2
    w = torch.cos(half_angles)
    xyz = rotation_axes * torch.sin(half_angles).unsqueeze(1)
    quaternions = torch.cat([w.unsqueeze(1), xyz], dim=1)  # Shape [N, 4]

    return quaternions

def compute_scales_from_triangles(triangles):
    """
    Compute the bounding box size of each triangle.

    :param triangles: Tensor of shape [N, 3, 3] representing N triangles with 3 vertices in 3D.
    :return: Tensor of shape [N, 3] representing the size of the bounding box for each triangle (width, height, depth).
    """
    # Compute the min and max coordinates along each axis for each triangle
    mins = torch.min(triangles, dim=1).values  # Shape [N, 3]
    maxs = torch.max(triangles, dim=1).values  # Shape [N, 3]

    # Compute the size of the bounding box
    bounding_box_sizes = maxs - mins  # Shape [N, 3]

    return bounding_box_sizes

def get_intrinsics_keep_fov(K, width, height):
    """
        Create intrinsic matrix such that the FOV matches that of input K.
        The principal point is kept at the center of the image.
    """
    assert K.shape == (3, 3), K.shape
    assert width > 0 and height > 0, (width, height)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    FoVx = 2 * math.atan(width / (2 * fx))
    FoVy = 2 * math.atan(height / (2 * fy))

    fx_new = width / (2 * math.tan(FoVx / 2))
    fy_new = height / (2 * math.tan(FoVy / 2))

    K_new = torch.tensor([[fx_new, 0.0, width / 2.0], [0.0, fy_new, height / 2.0], [0.0, 0.0, 1.0]]).to(K.device)
    return K_new

def get_intrinsics_resize_to_fit(K, width, height):
    """
    Resize the original image to fit within the specified width and height
    while maintaining the field of view (FoV) and aspect ratio.
    """
    assert K.shape == (3, 3), K.shape
    assert width > 0 and height > 0, (width, height)

    original_width = 2 * K[0, 2]
    original_height = 2 * K[1, 2]

    # Calculate original aspect ratio
    original_aspect = original_width / original_height
    new_aspect = width / height

    # Determine the scaling factor
    if new_aspect > original_aspect:
        # Width is the limiting dimension
        scale = width / original_width
    else:
        # Height is the limiting dimension
        scale = height / original_height

    # Scale the focal lengths and center
    fx_new = K[0, 0] * scale
    fy_new = K[1, 1] * scale
    cx_new = width / 2.0
    cy_new = height / 2.0

    K_new = torch.tensor([[fx_new, 0.0, cx_new], 
                          [0.0, fy_new, cy_new], 
                          [0.0, 0.0, 1.0]]).to(K.device)
    return K_new

def scale_to_range(x, a, b):
    return a + (b - a) * (x - x.min()) / (x.max() - x.min())

def compute_barycentric_coords(point, triangle):
    """
    Computes the barycentric coordinates of a point with respect to a triangle.

    Parameters:
    point (np.ndarray): The point of shape [3] (x, y, z).
    triangle (np.ndarray): The triangle vertices of shape [3, 3].

    Returns:
    np.ndarray: Barycentric coordinates (alpha, beta, gamma).
    """
    # Extract vertices
    V0, V1, V2 = triangle[0], triangle[1], triangle[2]
    
    # Compute vectors
    v0 = V1 - V0
    v1 = V2 - V0
    v2 = point - V0
    
    # Compute dot products
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    
    # Compute denominators
    denom = d00 * d11 - d01 * d01
    
    # Barycentric coordinates
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w
    
    return np.array([u, v, w])