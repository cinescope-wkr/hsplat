import sys
import os
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import tyro
import imageio
import numpy as np  # needed for .astype(np.uint8)
import utils
from main import Config


def naive_lut_phase_encoding(phase, max_phase=2 * math.pi):
    """Encode phase to 8-bit (uint8) using LUT for a 2D phase image.

    Args:
        phase (torch.Tensor): 2D phase map.
        max_phase (float): Phase wrapping value (default: 2*pi).

    Returns:
        torch.Tensor: uint8 phase map ranging [0,255].
    """
    phase = ((phase + max_phase / 2) % max_phase) / max_phase
    return ((1 - phase) * 255).to(torch.uint8)


def double_phase_encoding_multi_level(
    field,
    lut_perpixel=None,
    offset=0.0,  # Unused
    return_pa_pdiff=False,  # Unused
    offset_and_wrapping=True,
    max_phase=3 * math.pi,
    ref_levels=None,  # Unused
):
    """Convert a complex field to double phase encoded output for multi-level pixel LUT.

    Args:
        field (torch.Tensor): Input complex field.
        lut_perpixel (torch.Tensor, optional): Pixelwise LUT tensor.
        offset: unused.
        return_pa_pdiff: unused.
        offset_and_wrapping (bool): Whether to center and wrap phase.
        max_phase (float): Maximum phase value.
        ref_levels: unused.

    Returns:
        torch.Tensor: Encoded double phase output.
    """
    # Convert field to amplitude and phase representation
    amplitudes = field.abs()
    phases = field.angle()

    # Normalize amplitudes to avoid issues with arccos
    amplitudes = torch.clip(amplitudes / (amplitudes.max() + 1e-6), 1e-6, 1.0 - 1e-6)

    def get_phase_diff(amplitudes_local, perpixellut=None, max_phase_val=3 * math.pi):
        """Calculate pixelwise phase difference for double phase encoding.
        Uses LUT if provided, otherwise analytic formula."""
        phase_diff = 2 * torch.acos(amplitudes_local)  # Range: 0 ~ pi
        if perpixellut is not None:
            lut = perpixellut
            # Find the LUT index that best matches phase_diff
            phase_idx = torch.argmin(torch.abs(lut - phase_diff), dim=0, keepdim=True)
            phase_idx = phase_idx.float() * max_phase_val / 255
            return phase_idx
        else:
            return phase_diff

    phase_diff = get_phase_diff(amplitudes, lut_perpixel, max_phase_val=max_phase)

    phases_a = phases - phase_diff / 2
    phases_b = phases + phase_diff / 2

    # Double phase encoding: tile phases in checkerboard arrangement
    phases_out = phases_a.clone()
    phases_out[..., ::2, 1::2] = phases_b[..., ::2, 1::2]
    phases_out[..., 1::2, ::2] = phases_b[..., 1::2, ::2]

    if offset_and_wrapping:
        # Center phase about zero
        phases_out = phases_out - 0.5 * (phases_out.max() + phases_out.min())

        # Wrap phase so all values are within [-max_phase/2, max_phase/2]
        while phases_out.max() > max_phase / 2:
            phases_out = torch.where(phases_out > max_phase / 2, phases_out - 2.0 * math.pi, phases_out)
        while phases_out.min() < -max_phase / 2:
            phases_out = torch.where(phases_out < -max_phase / 2, phases_out + 2.0 * math.pi, phases_out)

        assert phases_out.max() <= max_phase / 2
        assert phases_out.min() >= -max_phase / 2

    return phases_out


def load_luts(lut_path, ref_levels=None):
    """Load and stack LUTs for specified reference levels, resized to 1024x1024.

    Args:
        lut_path (str): Directory containing phi_all_XXX.pt files.
        ref_levels (list of int): Reference intensity levels.

    Returns:
        torch.Tensor: Stacked LUT tensor of shape [K, 256, 1024, 1024].
    """
    if ref_levels is None:
        ref_levels = [127]
    luts = []
    for ref_level in ref_levels:
        lut_file = os.path.join(lut_path, f'phi_all_{int(ref_level)}.pt')
        lut = torch.load(lut_file).squeeze()
        luts.append(lut)
    luts = torch.stack(luts, 0)
    luts = nn.functional.interpolate(luts, size=(1024, 1024), mode='nearest')
    return luts


def get_per_pixel_lut_per_level(
    phase,
    lut_folder_path,
    calibrated_ref_levels,
    max_phase=3 * math.pi,
):
    """For each pixel, select the best LUT corresponding to the reference level which minimizes the
    difference between the pixel phase and LUT center candidate.

    Args:
        phase (torch.Tensor): Input phase map [M, N].
        lut_folder_path (str): Path to LUT files.
        calibrated_ref_levels (torch.Tensor): (K,) reference levels.
        max_phase (float): Max phase.

    Returns:
        torch.Tensor: Assembled per-pixel LUT [256, 1, M, N].
    """
    center_candidates = (255 - calibrated_ref_levels) / 255 * max_phase - max_phase / 2
    center_candidates = center_candidates.reshape(center_candidates.shape[0], 1, 1, 1)
    center_candidates = center_candidates.repeat(1, 1, *phase.shape[-2:])

    lut_ref_idx = torch.argmin(torch.abs(center_candidates - phase), dim=0, keepdim=True)
    lut_canvas = torch.zeros(256, 1, *phase.shape[-2:], device=phase.device)
    mask_total = torch.zeros_like(phase)

    for i, ref_level in enumerate(calibrated_ref_levels):
        this_lut_path = os.path.join(lut_folder_path, f'phi_all_{int(ref_level)}.pt')
        lut = torch.load(this_lut_path).squeeze().to(phase.device).unsqueeze(1)
        lut = nn.functional.interpolate(lut, size=phase.shape[-2:], mode='nearest')
        mask = (lut_ref_idx == i).squeeze()
        lut_canvas[mask.expand(256, 1, *phase.shape[-2:])] = lut[mask.expand(256, 1, *phase.shape[-2:])]
        mask_total += mask.float()
    return lut_canvas


@dataclass
class ConfigDPAC(Config):
    """Configuration class for DPAC encoding."""
    lut_folder_path: str = ''
    lut_path: str = None
    laser_amp_path: str = ''
    out_path: str = ''
    wavefront_path: str = ''
    max_phase: str = '3pi'
    slm_res: tuple = (1080, 1920)


def main(cfg: ConfigDPAC):
    """Main routine to apply DPAC encoding given user config.
    Loads wavefront/amp/LUT, applies per-pixel encoding, writes result image.
    """
    # Load and prepare data
    channel = cfg.channel
    wavefront = torch.load(cfg.wavefront_path)
    out_dir = os.path.dirname(cfg.out_path)
    os.makedirs(out_dir, exist_ok=True)
    lut_folder_path = cfg.lut_folder_path
    max_phase = 3 * math.pi if cfg.max_phase == '3pi' else 2 * math.pi
    wavefront = utils.crop_image(wavefront, cfg.slm_res)
    wavefront = utils.pad_image(wavefront, cfg.slm_res)

    lut = torch.load(cfg.lut_path) if cfg.lut_path is not None else None
    laser_amp = torch.load(cfg.laser_amp_path) if cfg.laser_amp_path is not None else None

    # Compensate wavefront by amplitude if LUT is used
    if lut is not None and laser_amp is not None:
        compensated_wavefront = wavefront / laser_amp
    else:
        compensated_wavefront = wavefront

    # Per-pixel LUT assignment
    calibrated_ref_levels = torch.linspace(104, 150, 47).to(wavefront.device)
    lut_perpixel = get_per_pixel_lut_per_level(
        compensated_wavefront, lut_folder_path, calibrated_ref_levels, max_phase=max_phase
    )

    dpac_phase = double_phase_encoding_multi_level(
        compensated_wavefront,
        lut_perpixel=lut_perpixel,
        max_phase=max_phase,
        offset_and_wrapping=False,
    )

    phase_8bit = naive_lut_phase_encoding(dpac_phase, max_phase=3 * math.pi)
    imageio.imwrite(cfg.out_path, phase_8bit.squeeze().cpu().numpy().astype(np.uint8))


if __name__ == '__main__':
    with torch.no_grad():
        cfg = tyro.cli(ConfigDPAC)
        main(cfg)
