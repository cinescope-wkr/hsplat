
"""
Wave propagation models (ASM, tilted ASM, ...)

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

import torch
import torch.nn as nn
import torch.fft as tfft

import utils


class ASM_rotation(nn.Module):
    """
    ASM with rotation (tilted ASM)

    The rotation order is: x and y.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.prop_dist = None  # TODO: assign z, and check if z is different from the previous one
        self._rotation = None
        self.src_pixel_pitch = cfg.src_pixel_pitch  # pixel pitch of tilted plane
        self.dst_pixel_pitch = cfg.pixel_pitch  # hologram pixel pitch
        self.dst_resolution = cfg.resolution_hologram  # hologram resolution
        self.n_pad_src = cfg.n_pad_src  # number of padding pixels
        self.n_pad_dst = cfg.n_pad_dst  # number of padding pixels

    def forward(self, wavefront, rotation=None, return_angular_spectrum=False):
        # Pad image first
        wavefront = utils.pad_image(
            wavefront,
            [self.n_pad_src * r for r in wavefront.shape[-2:]],
            padval=0,
            stacked_complex=False,
        )

        # Take Fourier transform
        wavefront = tfft.ifftshift(wavefront, (-2, -1))
        angular_spectrum = tfft.fftshift(
            tfft.fftn(wavefront, dim=(-2, -1), norm='ortho'), (-2, -1)
        )

        remapped_angular_spectrum = self.remap_angular_spectrum(
            angular_spectrum, rotation
        )

        # Take inverse Fourier transform
        wavefront = tfft.ifftn(
            tfft.ifftshift(remapped_angular_spectrum, dim=(-2, -1)),
            dim=(-2, -1),
            norm='ortho',
        )
        wavefront = tfft.ifftshift(wavefront, dim=(-2, -1))

        wavefront = utils.crop_image(
            wavefront, self.dst_resolution, pytorch=True, stacked_complex=False
        )

        if return_angular_spectrum:
            return wavefront, angular_spectrum, remapped_angular_spectrum
        return wavefront

    def w_uv(self, uu, vv, wavelength):
        return ((1 / wavelength) ** 2 - (uu ** 2 + vv ** 2)).sqrt()

    def remap_angular_spectrum(self, angular_spectrum, rotation=None):
        sample_resolution = tuple(
            [s * self.n_pad_dst for s in self.dst_resolution]
        )
        u = (
            torch.linspace(
                -(sample_resolution[0] - 1) / 2,
                (sample_resolution[0] - 1) / 2,
                sample_resolution[0],
                device=angular_spectrum.device,
            )
            / sample_resolution[0]
            / self.dst_pixel_pitch
        )
        v = (
            torch.linspace(
                -(sample_resolution[1] - 1) / 2,
                (sample_resolution[1] - 1) / 2,
                sample_resolution[1],
                device=angular_spectrum.device,
            )
            / sample_resolution[1]
            / self.dst_pixel_pitch
        )

        # -1/2p ~ 1/2p
        u, v = torch.meshgrid(u, v, indexing='ij')
        u = torch.transpose(u, 0, 1)
        v = torch.transpose(v, 0, 1)

        u_s, v_s = self.fourier_coordinate_transform(
            u, v, rotation, self.cfg.wavelength
        )

        grid = torch.stack((u_s, v_s), dim=-1).unsqueeze(0).to(angular_spectrum.device)
        bw = 1 / (2 * self.src_pixel_pitch)
        grid = grid / bw
        remapped_angular_spectrum = utils.grid_sample_complex(
            angular_spectrum,
            grid,
            align_corners=False,
            mode='bilinear',
            coord='rect'
        )

        return remapped_angular_spectrum

    def fourier_coordinate_transform(
        self, u, v, T, wavelength, direction='ref2src'
    ):
        """Assuming it is all in spatial frequency domain (1/x unit)."""
        u_c = T[..., 0, 2] / wavelength
        v_c = T[..., 1, 2] / wavelength

        wu = self.w_uv(u, v, wavelength)
        u_s = T[..., 0, 0] * u + T[..., 0, 1] * v + T[..., 0, 2] * wu
        v_s = T[..., 1, 0] * u + T[..., 1, 1] * v + T[..., 1, 2] * wu

        return u_s - u_c, v_s - v_c

    @property
    def rotation(self):
        """rotation is a 3x3 rotation matrix"""
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        if value is not None:
            if torch.is_tensor(value):
                self._rotation = value
            else:
                self._rotation = torch.tensor(value, dtype=torch.float32)
        else:
            self._rotation = None

    @property
    def H(self):
        """H is the transfer function in the Fourier space"""
        return self._H

    @H.setter
    def H(self, value):
        if value is not None:
            if torch.is_tensor(value):
                self._H = value
            else:
                self._H = torch.tensor(value, dtype=torch.float32)
        else:
            self._H = None

    def compute_H(self, wavefront, prop_dist, wavelength, pixel_pitch):
        pass


class ASM_parallel(nn.Module):
    """
    Normal ASM we have used everyday.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.prop_dist = None  # TODO: assign z, and check if z is different from the previous one

    def forward(self, wavefront, z, linear_conv=True, phase_compensation=None):
        # TODO:
        if phase_compensation is None and abs(z) < 1e-6:
            return wavefront
        H = self.compute_H(
            wavefront, z, self.cfg.wavelength, self.cfg.pixel_pitch, lin_conv=linear_conv
        )
        wavefront = self.prop(
            wavefront, H, linear_conv=linear_conv, phase_compensation=phase_compensation
        )
        return wavefront

    def propagate(self, wavefront):
        return wavefront

    def compute_H(
        self,
        input_field,
        prop_dist,
        wavelength,
        pixel_pitch,
        lin_conv=True,
        ho=(1, 1)
    ):
        dev = input_field.device
        output_res = [o * i for o, i in zip(ho, input_field.size()[-2:])]
        res_mul = 2 if lin_conv else 1
        odd_pad_y = 1 if input_field.shape[-2] % 2 == 1 else 0
        odd_pad_x = 1 if input_field.shape[-1] % 2 == 1 else 0
        num_y = res_mul * output_res[-2] + odd_pad_y
        num_x = res_mul * output_res[-1] + odd_pad_x
        dy, dx = pixel_pitch, pixel_pitch

        # Frequency coordinates sampling
        fy = torch.linspace(
            -(1 / (2 * dy) - 1 / (2 * dy * num_y)),
            1 / (2 * dy) - 1 / (2 * dy * num_y),
            num_y,
            device=dev
        )
        fx = torch.linspace(
            -(1 / (2 * dx) - 1 / (2 * dx * num_x)),
            1 / (2 * dx) - 1 / (2 * dx * num_x),
            num_x,
            device=dev
        )

        # Momentum/reciprocal space
        FX, FY = torch.meshgrid(fx, fy, indexing='ij')
        FX = torch.transpose(FX, 0, 1)
        FY = torch.transpose(FY, 0, 1)

        max_sq = torch.abs(FX ** 2 + FY ** 2).max()
        H_filter = (
            torch.abs(FX ** 2 + FY ** 2)
            <= (self.cfg.F_aperture ** 2) * max_sq
        ).float()

        if abs(prop_dist) < 1e-6:
            return H_filter
        G = 2 * math.pi * (
            (1 / wavelength) ** 2 - (FX ** 2 + FY ** 2)
        ).sqrt()
        H_exp = G.unsqueeze(0).unsqueeze(0).to(dev)
        return torch.exp(1j * (H_exp * prop_dist)) * H_filter

    def prop(
        self,
        u_in,
        H,
        linear_conv=True,
        padtype='zero',
        ho=(1, 1),
        phase_compensation=None
    ):
        input_resolution = u_in.size()[-2:]
        if linear_conv:
            # Preprocess with padding for linear conv.
            conv_size = [
                i * 2 if i % 2 == 0 else i * 2 + 1
                for i in input_resolution
            ]
            u_in = utils.pad_image(
                u_in, conv_size, padval=0, stacked_complex=False
            )

        if phase_compensation is not None:
            u_in = tfft.ifftshift(u_in, dim=(-2, -1))
        U1 = tfft.fftshift(
            tfft.fftn(u_in, dim=(-2, -1), norm='ortho'), (-2, -1)
        )

        if H is not None:
            U2 = U1 * H
        else:
            U2 = U1

        if phase_compensation is not None:
            U2 = U2 * torch.exp(1j * phase_compensation)

        u_out = tfft.ifftn(
            tfft.ifftshift(U2, (-2, -1)), dim=(-2, -1), norm='ortho'
        )

        if phase_compensation is not None:
            u_out = tfft.fftshift(u_out, dim=(-2, -1))

        output_res = [o * i for o, i in zip(ho, input_resolution)]

        if linear_conv:
            u_out = utils.crop_image(u_out, output_res, pytorch=True, stacked_complex=False)

        return u_out
