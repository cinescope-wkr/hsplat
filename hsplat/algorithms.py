"""
A script to implement CGH algorithms using primitives.

Released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC):
  - License is for non-commercial use only (contact Stanford for commercial licensing).
  - Provided as-is, without warranties.
  - Please cite our work if you use any code, data, or publish research based on this.

Article:
S. Choi, B. Chao, J. Yang, M. Gopakumar, G. Wetzstein
"Gaussian Wave Splatting for Computer-generated Holography",
ACM Transactions on Graphics (Proc. SIGGRAPH 2025)
"""

import os
import math
import logging
import numpy as np

import tqdm
import torch
import torch.nn.functional as F
import imageio

import utils

from load_data import generate_target_grid
from primitives import Point, Polygon, Gaussian, Points, Polygons, Gaussians
from propagations import ASM_parallel
from cuda._wrapper import cgh_gaussians_naive

logger = logging.getLogger(__name__)

def get_cgh_method(cfg):
    """Return the selected CGH method based on the configuration."""
    if cfg.method in ["naive-slow", "naive_slow"]:
        logger.info('Using naive-slow method ...')
        return naive_cgh_from_primitives
    elif cfg.method in ["naive-fast", "naive_fast"]:
        logger.info('Using naive-fast method ...')
        return naive_cgh_from_primitives_fast
    elif cfg.method in ["silhouette", "silhouette-based", "silhouette_based"]:
        logger.info('Using silhouette-based method ...')
        return silhouette_based_method
    elif cfg.method in ["alpha-wave-blending", "alpha_wave_blending"]:
        logger.info('Using alpha-wave-blending method ...')
        return alpha_wave_blending
    else:
        raise ValueError(f"Unsupported CGH method: {cfg.method}")

def ifft(a):
    """Centered 2D IFFT with orthonormal normalization."""
    return torch.fft.ifftshift(torch.fft.ifftn(torch.fft.fftshift(a, dim=(-2, -1)), dim=(-2, -1), norm='ortho'), dim=(-2, -1))

def fft(a):
    """Centered 2D FFT with orthonormal normalization."""
    return torch.fft.ifftshift(torch.fft.fftn(torch.fft.fftshift(a, dim=(-2, -1)), dim=(-2, -1), norm='ortho'), dim=(-2, -1))

def lateral_shift(input_wavefront, shift, pixel_pitch, resolution_hologram):
    """
    Laterally shift the wavefront by the specified vector.
    """
    x_scale = resolution_hologram[-1] / input_wavefront.shape[-1]
    y_scale = resolution_hologram[-2] / input_wavefront.shape[-2]
    theta = torch.tensor([
        [x_scale, 0, -2 * shift[0] / pixel_pitch * x_scale / resolution_hologram[-1]],
        [0, y_scale, -2 * shift[1] / pixel_pitch * y_scale / resolution_hologram[-2]]
    ],
        device=input_wavefront.device,
        dtype=torch.get_default_dtype()
    ).unsqueeze(0)

    grid = F.affine_grid(theta, (1, 1, *resolution_hologram), align_corners=False)
    output_wavefront = utils.grid_sample_complex(input_wavefront, grid, align_corners=False, mode='nearest')
    return output_wavefront

def cgh_from_point(primitive, cfg):
    """
    Generate hologram for a point primitive via slow FFT-based method.
    """
    z = primitive.position[-1]
    try:
        subhologram_size = abs(z) * math.tan(math.asin(cfg.wavelength / cfg.pixel_pitch))
        subhologram_size = math.ceil(subhologram_size.item() / cfg.pixel_pitch)
        subhologram_size = 2 * subhologram_size + 1
    except Exception:
        subhologram_size = 3

    input_wavefront = torch.zeros(
        1, 1, subhologram_size, subhologram_size, dtype=torch.complex64, device=cfg.dev)
    input_wavefront[..., subhologram_size // 2, subhologram_size // 2] = (
        primitive.amplitude * torch.exp(1j * primitive.phase)
    )

    asm = ASM_parallel(cfg)
    propagated_wavefront = asm(input_wavefront, z)
    output_wavefront = lateral_shift(propagated_wavefront, primitive.position[:2],
                                     cfg.pixel_pitch, cfg.resolution_hologram)
    return output_wavefront

def cgh_from_triangle(primitive, cfg, prev_wavefront=None):
    """
    Compute the wavefront from a triangle primitive.
    """
    wavefront = fully_analytic_cgh_basic(
        primitive, cfg, shape='triangle', prev_wavefront=prev_wavefront)

    if torch.is_tensor(wavefront) and torch.isnan(wavefront).any():
        raise ValueError('wavefront is nan')

    return wavefront

def cgh_from_gaussian(primitive, cfg, prev_wavefront=None, frequency_grid=None):
    """
    Compute the wavefront from a single Gaussian primitive.
    """
    wavefront = fully_analytic_cgh_gaussian(
        primitive,
        cfg,
        shape='gaussian',
        prev_wavefront=prev_wavefront,
        frequency_grid=frequency_grid
    )

    if torch.is_tensor(wavefront) and torch.isnan(wavefront).any():
        raise ValueError('wavefront is nan')

    return wavefront

def cgh_from_gaussians_fast(primitives, cfg, prev_wavefront=None):
    """
    Compute the wavefront from Gaussian primitives (batched CUDA implementation).
    """
    wavefront = fully_analytic_cgh_gaussians_fast(
        primitives, cfg, shape='gaussian', prev_wavefront=prev_wavefront)

    if torch.isnan(wavefront).any():
        raise ValueError('wavefront is nan')

    return wavefront

def cgh_from_primitive(primitive, cfg, prev_wavefront=None, frequency_grid=None):
    """
    Dispatch wavefront computation for an individual primitive type.
    """
    if isinstance(primitive, Point):
        return cgh_from_point(primitive, cfg)
    elif isinstance(primitive, Polygon):
        return cgh_from_triangle(primitive, cfg, prev_wavefront=prev_wavefront)
    elif isinstance(primitive, Gaussian):
        return cgh_from_gaussian(primitive, cfg, prev_wavefront=prev_wavefront, frequency_grid=None)
    else:
        raise TypeError('Unsupported primitive type: {}'.format(type(primitive)))

def cgh_from_primitives_fast(primitives, cfg, prev_wavefront=None):
    """
    Batched fast wavefront computation for primitives collection.
    """
    if isinstance(primitives, Gaussians):
        return cgh_from_gaussians_fast(primitives, cfg, prev_wavefront=prev_wavefront)
    else:
        raise TypeError('Unsupported primitives container: {}'.format(type(primitives)))

def naive_cgh_from_primitives(primitives, cfg):
    """
    Generate a mid-plane hologram (Wavefront Recording Plane, WRP) from primitives using PyTorch (single-threaded).
    """
    # Setup target grid (not actually used in this basic implementation, but preserved for future compatibility)
    _ = generate_target_grid(cfg.resolution_hologram, cfg.dev, cfg.pixel_pitch)

    # Initialize the wavefront as zeros
    wavefront = torch.zeros(
        1, 1, *cfg.resolution_hologram, dtype=torch.complex64, device=cfg.dev)
    T = torch.ones_like(wavefront)

    # Main primitive rendering loop
    for primitive in tqdm.tqdm(primitives):
        occluded_wavefront = cgh_from_primitive(
            primitive, cfg, prev_wavefront=wavefront)
        occluded_wavefront = (
            occluded_wavefront * (occluded_wavefront.abs() > cfg.threshold_epsilon)
            + torch.zeros_like(occluded_wavefront) * torch.exp(1j * torch.zeros_like(occluded_wavefront))
        )
        c = primitive.color
        wavefront = wavefront + T * primitive.opacity * occluded_wavefront * c

    return wavefront

def naive_cgh_from_primitives_fast(primitives, cfg):
    """
    Generate a mid-plane hologram using fast CUDA kernel implementation.
    """
    logger.info("Using naive fast CGH method")
    wavefront = torch.zeros(
        1, 1, *cfg.resolution_hologram, dtype=torch.complex64, device=cfg.dev)

    # Batched, CUDA version of cgh_from_primitive:
    wavefront = cgh_from_primitives_fast(primitives, cfg)
    return wavefront

def silhouette_based_method(primitives, cfg):
    """
    Generates a mid-plane hologram using silhouette-based occlusion handling.
    """
    # Setup grid and frequency grid for mask operations as needed
    target_grid_x, target_grid_y = generate_target_grid(cfg.resolution_hologram, cfg.dev, cfg.pixel_pitch)
    wavefront = torch.zeros(
        1, 1, *cfg.resolution_hologram, dtype=torch.complex64, device=cfg.dev)
    frequency_grid = utils.make_freq_grid(cfg)

    primitives.sort('back2front')
    asm_prop = ASM_parallel(cfg)
    prev_z = None
    cfg.return_at_object_depth = True

    for primitive in tqdm.tqdm(primitives):
        if prev_z is not None:
            propagated_wavefront = asm_prop(wavefront, primitive.z())
        else:
            propagated_wavefront = wavefront

        prev_z = primitive.z()
        object_wavefront, phase_compensation = cgh_from_primitive(primitive, cfg)
        object_wavefront = asm_prop(object_wavefront, primitive.z(), linear_conv=False, phase_compensation=phase_compensation)
        object_wavefront = (
            object_wavefront * (object_wavefront.abs() > cfg.threshold_epsilon)
            + torch.zeros_like(object_wavefront) * torch.exp(1j * torch.zeros_like(object_wavefront))
        )
        c = primitive.color
        wavefront = (
            (1 - object_wavefront.abs() * primitive.opacity) * propagated_wavefront
            + object_wavefront * c * primitive.opacity
        )

        wavefront = asm_prop(wavefront, -primitive.z())

    return wavefront

def alpha_wave_blending(primitives, cfg):
    """
    Alpha Wave Blending (AWB) method for occlusion-aware rendering of primitives.
    """
    assert cfg.return_at_object_depth, 'return_at_object_depth must be True for alpha wave blending'
    frequency_grid = utils.make_freq_grid(cfg)
    _ = generate_target_grid(cfg.resolution_hologram, cfg.dev, cfg.pixel_pitch)
    wavefront = torch.zeros(
        1, 1, *cfg.resolution_hologram, dtype=torch.complex64, device=cfg.dev
    )
    primitives.sort('front2back' if cfg.order_front2back else 'back2front')
    asm_prop = ASM_parallel(cfg)
    T = torch.ones_like(wavefront)

    for primitive in tqdm.tqdm(primitives):
        object_wavefront, phase_compensation = cgh_from_primitive(
            primitive, cfg, frequency_grid=frequency_grid
        )

        # Optionally add Gaussian-distributed random phase
        if cfg.random_sigma_gaussian > 0.0:
            object_wavefront = object_wavefront * torch.exp(
                1j * torch.randn_like(object_wavefront) * cfg.random_sigma_gaussian
            )

        c = primitive.color

        # Compute the splatted wavefront at the SLM plane
        splatted_object_wavefront = asm_prop(
            T * primitive.opacity * c * object_wavefront,
            0.0,
            linear_conv=False,
            phase_compensation=phase_compensation,
        )

        if cfg.threshold_epsilon > 0.0:
            splatted_object_wavefront = splatted_object_wavefront * (
                splatted_object_wavefront.abs() > cfg.threshold_epsilon
            )

        if cfg.order_front2back:
            wavefront = wavefront + splatted_object_wavefront

        # Occlusion mask propagation for alpha wave blending
        if cfg.alpha_wave_blending:
            alpha_u = primitive.opacity * object_wavefront.abs()
            alpha_u = alpha_u * (alpha_u > cfg.threshold_epsilon)
            if cfg.threshold_binary_gaussian > 0.0:
                # Threshold makes alpha_u binary; activations below threshold suppressed to 0.
                alpha_u = (alpha_u > cfg.threshold_binary_gaussian).float()

            # T accumulates occlusion masks, i.e., what gets masked by front objects
            T = T * (1 - alpha_u)

    return wavefront

def angular_spectrum_reference(fx, fy, shape='triangle'):
    """
    Generate reference angular spectrum (AS) for a few primitive shapes.
    """
    if shape == 'triangle':
        # Reference: Zhang et al., Polygon CGH review paper; Applied Optics, 2022
        G0 = ((torch.exp(-1j * 2 * torch.pi * fx) - 1) / ((2 * torch.pi) ** 2 * fx * fy)) + \
             ((1 - torch.exp(-1j * 2 * torch.pi * (fx + fy))) / ((2 * torch.pi) ** 2 * fy * (fx + fy))
        )

        G0[(fx == 0) & (fy != 0)] = ((1 - torch.exp(-1j * 2 * torch.pi * fy[(fx == 0) & (fy != 0)])) /
                                     (2 * torch.pi * fy[(fx == 0) & (fy != 0)]) ** 2) - \
                                    (1j / (2 * torch.pi * fy[(fx == 0) & (fy != 0)]))

        G0[(fx != 0) & (fy == 0)] = ((torch.exp(-1j * 2 * torch.pi * fx[(fx != 0) & (fy == 0)]) - 1) /
                                     (2 * torch.pi * fx[(fx != 0) & (fy == 0)]) ** 2) + \
                                   (1j * torch.exp(-1j * 2 * torch.pi * fx[(fx != 0) & (fy == 0)])) / \
                                   (2 * torch.pi * fx[(fx != 0) & (fy == 0)])

        G0[(fx == fy) & (fx == 0)] = 0.5
        G0[(fx == -fy) & (fx != 0)] = ((1 - torch.exp(1j * 2 * torch.pi * fy[(fx == -fy) & (fx != 0)])) /
                                       (2 * torch.pi * fy[(fx == -fy) & (fx != 0)]) ** 2) + \
                                      (1j / (2 * torch.pi * fy[(fx == -fy) & (fx != 0)]))
    elif shape == 'rectangle':
        raise NotImplementedError("Rectangle angular spectrum not implemented")
    elif shape == 'circle':
        # Reference angular spectrum for circles
        G0 = torch.special.bessel_j1(2 * torch.pi * torch.sqrt(fx**2 + fy**2)) / torch.sqrt(fx**2 + fy**2)
    elif shape == 'gaussian':
        # Reference angular spectrum of a gaussian is another gaussian
        G0 = 2 * torch.pi * torch.exp(-(2 * torch.pi)**2 * (fx**2 + fy**2) / 2)
    else:
        raise ValueError("Unknown shape for angular_spectrum_reference: {}".format(shape))
    return G0

def fully_analytic_cgh_basic(
    primitive, cfg, shape='triangle', return_angular_spectrum=False, *args, **kwargs
):
    """
    Analytic angular spectrum computation for triangle elements.
    Returns (optionally) phase compensation for occlusion handling. Output is always (x, y, z) ordering.
    """
    try:
        sgn = primitive.normal[2] / primitive.normal[2].abs()
        X = torch.cat([primitive.v1, primitive.v2, primitive.v3], dim=1)  # (3,3)
        fx, fy, fz = utils.make_freq_grid(cfg)
        n = primitive.normal
        R = utils.get_rotation_matrix(n)
        coord = X[:, 0]
        Xl = R @ X
        c = -Xl[:, 0]
        Xl = Xl - Xl[:, 0:1]

        detXl = Xl[0, 2] * Xl[1, 1] - Xl[1, 2] * Xl[0, 1]
        if detXl == 0:
            raise ValueError('Degenerate triangle: vertices are collinear.')

        A = torch.tensor(
            [
                [Xl[1, 1] - Xl[1, 2], -Xl[0, 1] + Xl[0, 2]],
                [-Xl[1, 2], Xl[0, 2]],
            ],
            device=X.device,
        ) / detXl

        flx, fly, flz = utils.rotate_frequency_grid(R, fx, fy, fz, cfg.wavelength)
        uc = torch.tensor([0.0, 0.0, 1.0], device=X.device)
        du = (R @ uc) / cfg.wavelength
        fx_l_offset = flx - du[0]
        fy_l_offset = fly - du[1]
        fx_ref, fy_ref = utils.rotate_frequency_grid(torch.inverse(A.t()), fx_l_offset, fy_l_offset)

        G0 = angular_spectrum_reference(fx_ref, fy_ref, shape=shape)
        Gl = G0 / torch.det(A)

        if cfg.phase_matching and not cfg.return_at_object_depth:
            Gl = Gl * torch.exp(1j * 2 * torch.pi / cfg.wavelength * coord[2])

        # Triangle's AS in world space
        G = Gl * flz / fz * torch.exp(-1j * 2 * torch.pi * (fx * coord[0] + fy * coord[1]))
        if not cfg.return_at_object_depth:
            G = G * torch.exp(-1j * 2 * torch.pi * fz * coord[2])

        G = G.unsqueeze(0).unsqueeze(0)
        wavefront = torch.fft.fftshift(
            torch.fft.ifftn(
                torch.fft.ifftshift(G, dim=(-2, -1)),
                dim=(-2, -1),
                norm='backward'
            ),
            dim=(-2, -1)
        )
        wavefront = utils.crop_image(wavefront, cfg.resolution_hologram)
        normalization_factor = 1 / (cfg.pixel_pitch**2)
        wavefront = wavefront * normalization_factor

        if cfg.return_at_object_depth:
            phase_comp = 2 * torch.pi / cfg.wavelength * coord[2] - 2 * torch.pi * fz * coord[2]
            return sgn * wavefront, phase_comp
        else:
            return sgn * wavefront, None
    except Exception as e:
        logger.info(f'Error in fully_analytic_cgh_basic: {e}')
        zeros = torch.zeros(cfg.resolution_hologram, device=cfg.dev)
        if cfg.return_at_object_depth:
            return zeros, zeros
        else:
            return zeros, None

def fully_analytic_cgh_gaussian(
    primitive, cfg, shape='gaussian', return_angular_spectrum=False, frequency_grid=None, *args, **kwargs
):
    """
    Compute analytically the CGH of an individual Gaussian primitive.
    """
    return_at_object_depth = cfg.return_at_object_depth

    if frequency_grid is None:
        fx, fy, fz = utils.make_freq_grid(cfg)
    else:
        fx, fy, fz = frequency_grid

    # Euler angles from quaternion
    theta_x, theta_y, theta_z = utils.quaternion_to_euler_angles_zyx(primitive.quat)

    Rx = utils.coordinate_rotation_matrix('x', -theta_x).to(primitive.device)
    Ry = utils.coordinate_rotation_matrix('y', -theta_y).to(primitive.device)
    Sinv = torch.diag(1 / primitive.scale[:2])
    R = (Ry @ Rx)
    coord = primitive.mean
    Rz = utils.coordinate_rotation_matrix('z', -theta_z).to(primitive.device)
    A = Sinv @ Rz[:2, :2]

    flx, fly, flz = utils.rotate_frequency_grid(R, fx, fy, fz, cfg.wavelength)
    uc = torch.tensor([0.0, 0.0, 1.0], device=primitive.device)
    du = (R @ uc) / cfg.wavelength
    fx_l_offset = flx - du[0]
    fy_l_offset = fly - du[1]
    fx_ref, fy_ref = utils.rotate_frequency_grid(torch.inverse(A.t()), fx_l_offset, fy_l_offset)

    G0 = angular_spectrum_reference(fx_ref, fy_ref, shape='gaussian')
    Gl = G0 / torch.det(A)
    if cfg.phase_matching and not return_at_object_depth:
        Gl = Gl * torch.exp(1j * 2 * torch.pi / cfg.wavelength * coord[2])

    G = Gl * flz / fz * torch.exp(-1j * 2 * torch.pi * (fx * coord[0] + fy * coord[1]))
    if not return_at_object_depth:
        G = G * torch.exp(-1j * 2 * torch.pi * fz * coord[2])

    G = G.unsqueeze(0).unsqueeze(0)
    wavefront = torch.fft.fftshift(
        torch.fft.ifftn(
            torch.fft.ifftshift(G, dim=(-2, -1)),
            dim=(-2, -1),
            norm='backward'
        ),
        dim=(-2, -1)
    )
    wavefront = utils.crop_image(wavefront, cfg.resolution_hologram)
    normalization_factor = 1 / (cfg.pixel_pitch**2)
    wavefront = wavefront * normalization_factor

    if return_at_object_depth:
        phase_compensation = 2 * torch.pi / cfg.wavelength * coord[2] - 2 * torch.pi * fz * coord[2]
        return wavefront, phase_compensation
    else:
        if return_angular_spectrum:
            return wavefront, G
        else:
            return wavefront

def fully_analytic_cgh_gaussians_fast(
    primitives, cfg, shape='gaussian', return_angular_spectrum=False, *args, **kwargs
):
    """
    Batched, CUDA implementation of analytic Gaussian CGH. Accepts a Primitives container (with .means, .quats, etc).
    """
    device = primitives.means.device
    colors = primitives.colors

    # Compute frequency grid
    fx, fy, fz = utils.make_freq_grid(cfg)
    theta_x, theta_y, theta_z = utils.quaternion_to_euler_angles_zyx(primitives.quats)

    Rx = utils.coordinate_rotation_matrix('x', -theta_x, keep_channel_dim=True).to(device)
    Ry = utils.coordinate_rotation_matrix('y', -theta_y, keep_channel_dim=True).to(device)
    Sinv = torch.diag_embed(1 / primitives.scales[..., :2])
    R = torch.bmm(Ry, Rx)
    c = torch.bmm(-R, primitives.means.unsqueeze(-1)).squeeze(-1)
    Rz = utils.coordinate_rotation_matrix('z', -theta_z, keep_channel_dim=True).to(device)
    A = torch.bmm(Sinv, Rz[..., :2, :2])
    A_inv_T = torch.inverse(A.transpose(-1, -2))
    A_det = torch.det(A)
    uc = torch.tensor([0.0, 0.0, 1.0], device=device).unsqueeze(0).expand(R.shape[0], -1)
    R_T = R.transpose(-1, -2)
    local_AS_shift = torch.sum(uc * torch.bmm(R_T, c.unsqueeze(-1)).squeeze(-1), dim=-1)
    du = torch.bmm(R, uc.unsqueeze(-1)).squeeze(-1) / cfg.wavelength

    # CUDA kernel for summing up all Gaussians in a single pass in Fourier domain:
    G_real, G_imag = cgh_gaussians_naive(
        fx.contiguous(),
        fy.contiguous(),
        fz.contiguous(),
        cfg.wavelength,
        R.contiguous(),
        A_inv_T.contiguous(),
        A_det.contiguous(),
        c.contiguous(),
        du.contiguous(),
        local_AS_shift.contiguous(),
        primitives.opacities.contiguous(),
        colors.contiguous(),
    )
    G = G_real + 1j * G_imag
    G = G.unsqueeze(0).unsqueeze(0)
    wavefront = torch.fft.fftshift(
        torch.fft.ifftn(
            torch.fft.ifftshift(G, dim=(-2, -1)),
            dim=(-2, -1),
            norm='backward'
        ),
        dim=(-2, -1)
    )
    wavefront = utils.crop_image(wavefront, cfg.resolution_hologram)
    normalization_factor = 1 / (cfg.pixel_pitch**2)
    wavefront = wavefront * normalization_factor
    wavefront = (
        wavefront * (wavefront.abs() > cfg.threshold_epsilon)
        + torch.zeros_like(wavefront) * torch.exp(1j * torch.zeros_like(wavefront))
    )

    if return_angular_spectrum:
        return wavefront, G
    else:
        return wavefront