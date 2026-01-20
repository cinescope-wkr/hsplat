"""
Main execution script

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

Article: 
S. Choi, B. Chao, J. Yang, M. Gopakumar, G. Wetzstein
"Gaussian Wave Splatting for Computer-generated Holography",
ACM Transactions on Graphics (Proc. SIGGRAPH 2025)
"""

import os
import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import imageio
import tyro
import matplotlib.pyplot as plt

from load_data import load_primitives
from algorithms import get_cgh_method
from propagations import ASM_parallel
import time

import viz_utils
from viz_utils import parser
from viz_utils import visualization
import json

import utils

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class Config:
    # Unit definitions
    cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
    dev: torch.device = torch.device("cuda")
    pixel_pitch: float = 8*um
    src_pixel_pitch: float = 8*um
    wavelengths: Tuple[float, float, float] = (638*nm, 520*nm, 488*nm)
    ch_str: Tuple[str] = ('red', 'green', 'blue')
    channel: int = 1  # 3 is RGB

    out_resolution_hologram: Tuple[int, int] = (1024, 1536)
    resolution_scale_factor: float = 2
    culling_scale_factor: float = 2

    pad_n: int = 1  # for analytical method
    n_pad_src: int = 4  # for tilted plane
    n_pad_dst: int = 4  # for tilted plane

    target_asset: Literal[
        "gaussians", "textured_mesh", "points_from_gaussians",
        "points_from_mesh", "points_from_sfm", "rgbd",
    ] = "gaussians"

    num_points: Optional[int] = None
    target_size_scale: float = 300
    num_frames: int = 1

    data_path: Optional[str] = None
    out_path: Optional[str] = None
    gs_model_path: Optional[str] = None

    illum: Optional[torch.Tensor] = None

    alpha_wave_blending: bool = True
    phase_matching: bool = True
    order_front2back: bool = True

    method: Literal[
        "naive_slow", "naive_fast",
        "silhouette", "alpha-wave-blending", "alpha_wave_blending",
    ] = "naive_slow"
    batch_size: int = None

    scene_dir: Optional[str] = None
    num_trajectory_frames: Optional[int] = None
    camtoworld: Optional[np.ndarray] = None
    K: Optional[np.ndarray] = None
    frame_start_idx: int = 0
    frame_end_idx: int = 0

    num_focal_slices: int = 30
    z_min: float = -0.005
    z_max: float = 0.005
    remap_sigma: float = 1.0

    F_aperture: float = 1.0
    return_at_object_depth: bool = True
    threshold_epsilon: float = 1/255

    pc_scale_multiplier: float = 0.5
    random_sigma_gaussian: float = 0.0
    threshold_binary_gaussian: float = -1.0
    load_points_as_pixels: bool = False
    profiling: bool = False

def save_results(wavefront, target_amp, cfg, name=None, out_folder=None, no_save=False):
    """Save the results to disk."""
    wavefront, _ = utils.decode_dict_to_tensor(wavefront)
    target_amp, _ = utils.decode_dict_to_tensor(target_amp)
    if len(target_amp.squeeze().shape) == 4:
        target_amp = target_amp[0:1]  # handle multiple frame cases

    wavefront = utils.crop_image(wavefront, cfg.out_resolution_hologram)
    wavefront = torch.nan_to_num(wavefront, nan=0.0, posinf=0.0, neginf=0.0)
    amp = (wavefront.abs() ** 2).mean(dim=0, keepdim=True).sqrt()

    angle = None
    if wavefront.dtype == torch.complex64:
        angle = (wavefront.angle() + math.pi) / (2 * math.pi)

    # Amplitude adjustment
    if target_amp is not None and "points" not in cfg.target_asset:
        if target_amp.shape[1] == 3:
            amp_chs = []
            for ch in range(3):
                target_amp_ch = target_amp[:, ch:ch+1]
                amp_ch = amp[:, ch:ch+1]
                target_amp_ch = (target_amp_ch.abs() ** 2).mean(dim=0, keepdim=True).sqrt()
                amp_crop_ch = utils.crop_image(amp_ch, target_amp_ch.shape[-2:])
                target_amp_ch = utils.crop_image(target_amp_ch, amp_crop_ch.shape[-2:])
                s_ch = (amp_crop_ch * target_amp_ch.squeeze()).mean() / (amp_crop_ch ** 2).mean()
                amp_ch = (amp_ch * s_ch).clamp(0, 1)
                amp_chs.append(amp_ch)
            amp = torch.cat(amp_chs, dim=1)
        else:
            target_amp = (target_amp.abs() ** 2).mean(dim=0, keepdim=True).sqrt()
            amp_crop = utils.crop_image(amp, target_amp.shape[-2:])
            target_amp = utils.crop_image(target_amp, amp_crop.shape[-2:])
            s = (amp_crop * target_amp.squeeze()).mean() / (amp_crop ** 2).mean()
            amp = (amp * s).clamp(0, 1)

    def save_image(data, name, cmap=None):
        if data is None:
            return
        path_base = f'{out_folder}/{cfg.target_asset}_{name}_{cfg.pixel_pitch*1e6:.0f}um_{cfg.num_frames}frames'
        data = data.detach().cpu()
        if data.dim() >= 3 and data.shape[1] == 3:
            if data.dim() == 3:
                data = data.unsqueeze(0)
            for i, data_i in enumerate(data):
                out_path = f'{path_base}_color_{i}.png'
                imageio.imwrite(out_path, (data_i.permute(1, 2, 0) * 255).clamp(0, 255).numpy().astype(np.uint8))
        else:
            data = data.squeeze()
            if data.ndim == 3:
                for i in range(data.shape[0]):
                    out_path = f'{path_base}_{i}.png'
                    imageio.imwrite(out_path, (data[i, ...] * 255).clamp(0, 255).numpy().astype(np.uint8))
            else:
                out_path = f'{path_base}.png'
                if cmap is not None:
                    plt.imsave(out_path, data.numpy(), cmap=cmap)
                else:
                    imageio.imwrite(out_path, (data * 255).clamp(0, 255).numpy().astype(np.uint8))

    if not no_save:
        torch.save(wavefront, f'{out_folder}/{cfg.target_asset}_{name}_{cfg.pixel_pitch*1e6:.0f}um_{cfg.num_frames}frames_wavefront.pt')
        save_image(amp, f'amp{ "_" + str(name) if name is not None else ""}')
        save_image(angle, f'angle{ "_" + str(name) if name is not None else ""}', cmap='plasma')
        if target_amp is not None:
            if name is not None and "c_" in name:
                channel = name.split("c_")[-1]
                save_image(target_amp, f'target_c_{channel}')
            else:
                save_image(target_amp, 'target')

    return amp, angle

def get_camera_params(cfg):
    """Load camera intrinsics and extrinsics for CGH rendering."""
    if cfg.scene_dir is not None and "360" in cfg.scene_dir:
        test_idx_dict = {
            "kitchen": 7, "bicycle": 12, "stump": 10, "counter": 17,
            "room": 21, "bonsai": 36, "garden": 16
        }
        parser = viz_utils.parser.COLMAPParser(data_dir=cfg.scene_dir, normalize=True)
        scene_name = cfg.scene_dir.split("/")[-1]
        if cfg.num_trajectory_frames is None:
            test_frame_idx_list = [idx for idx in range(len(parser.camtoworlds)) if idx % parser.test_every == 0]
            if scene_name not in test_idx_dict:
                raise NotImplementedError(f"Scene {scene_name} is not supported.")
            test_idx = test_idx_dict[scene_name]
            test_frame_idx = test_frame_idx_list[test_idx]
            trajectory_camtoworlds = parser.camtoworlds[test_frame_idx:test_frame_idx + 1]
        else:
            trajectory_camtoworlds = viz_utils.visualization.get_ellipse_path(parser.camtoworlds, num_frames=cfg.num_trajectory_frames)
            trajectory_camtoworlds = trajectory_camtoworlds[cfg.frame_start_idx:cfg.frame_end_idx + 1]
        K = list(parser.Ks_dict.values())[0]
    elif cfg.scene_dir is not None and "nerf_synthetic" in cfg.scene_dir:
        test_idx_dict = {
            "chair": 35, "drums": 5, "ficus": 65, "hotdog": 51, "lego": 91,
            "materials": 18, "mic": 58, "ship": 85
        }
        scene_name = cfg.scene_dir.split("/")[-1]
        if cfg.num_trajectory_frames is None:
            parser = viz_utils.parser.BlenderParser(data_dir=cfg.scene_dir, split='val')
            if scene_name not in test_idx_dict:
                raise NotImplementedError(f"Scene {scene_name} is not supported.")
            frame_idx = test_idx_dict[scene_name]
            trajectory_camtoworlds = parser.camtoworlds[frame_idx:frame_idx+1]
        else:
            parser = viz_utils.parser.BlenderParser(data_dir=cfg.scene_dir, split='test')
            trajectory_camtoworlds = parser.camtoworlds[::4]
            cfg.num_trajectory_frames = len(trajectory_camtoworlds)
            trajectory_camtoworlds = trajectory_camtoworlds[cfg.frame_start_idx:cfg.frame_end_idx + 1]
        K = parser.K
    else:
        raise NotImplementedError(f"Scene directory {cfg.scene_dir} is not supported.")
    return trajectory_camtoworlds, K

def process_cfg(cfg):
    if cfg.gs_model_path is None:
        raise ValueError("gs_model_path must be provided for all target assets for depth range remapping.")
    cfg.resolution_hologram = (
        int(cfg.out_resolution_hologram[0] * cfg.resolution_scale_factor),
        int(cfg.out_resolution_hologram[1] * cfg.resolution_scale_factor)
    )
    if "naive" in cfg.method:
        cfg.alpha_wave_blending = False
    if "points_from" in cfg.target_asset:
        if cfg.num_points is None:
            scene_name = cfg.scene_dir.split("/")[-1]
            if scene_name not in utils.NUM_TRIANGLES:
                raise NotImplementedError(f"Scene {scene_name} is not supported.")
            logger.info(f"[{cfg.target_asset}]: Sampling {utils.NUM_TRIANGLES[scene_name]} points.")
            cfg.num_points = utils.NUM_TRIANGLES[scene_name]
        else:
            logger.info(f"[{cfg.target_asset}]: Sampling {cfg.num_points} points.")
    if cfg.method == "naive_fast" and not cfg.profiling:
        logger.info("Increasing padding to 2 for numerical stability.")
        cfg.pad_n = 2
    if cfg.num_trajectory_frames is not None:
        if cfg.frame_start_idx is None:
            cfg.frame_start_idx = 0
        else:
            assert cfg.frame_start_idx >= 0
        if cfg.frame_end_idx is None:
            cfg.frame_end_idx = cfg.num_trajectory_frames - 1
        else:
            assert cfg.frame_end_idx < cfg.num_trajectory_frames
    return cfg

def main(cfg: Config):
    """Main function to compute wavefront and amplitude."""
    target_primitives, target_amp = load_primitives(cfg)
    logger.info("Loaded primitives. Starting CGH algorithm...")
    start_t = time.time()
    wavefront = get_cgh_method(cfg)(target_primitives, cfg)
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_t
    info = {"elapsed_time": elapsed_time}
    logger.info(f"Elapsed time: {elapsed_time:.2f} sec")
    return wavefront, target_amp, info

if __name__ == "__main__":
    with torch.no_grad():
        cfg = tyro.cli(Config)
        cfg = process_cfg(cfg)
        camtoworlds, K = get_camera_params(cfg)

        out_folder = cfg.out_path if cfg.out_path is not None else \
            f'results/{cfg.data_path.split("/")[-1][:-4] if cfg.data_path is not None else ""}_'
        if not cfg.alpha_wave_blending:
            out_folder += "_no_ab"
        if not cfg.phase_matching:
            out_folder += "_no_pm"
        if not cfg.order_front2back:
            out_folder += "_silhouette"
        logger.info('out_folder %s', out_folder)
        os.makedirs(out_folder, exist_ok=True)

        is_rgb = (cfg.channel == 3)
        for frame_id in range(camtoworlds.shape[0]):
            frame_folder = os.path.join(out_folder, f"frame_{frame_id + cfg.frame_start_idx}")
            os.makedirs(frame_folder, exist_ok=True)
            cfg.camtoworld = camtoworlds[frame_id]
            cfg.K = torch.tensor(K)

            video_amps = {}
            wfs_color = {}
            target_amps_color = {}

            channels = range(3) if is_rgb else [cfg.channel]
            for channel in channels:
                cfg.channel = channel
                cfg.wavelength = cfg.wavelengths[cfg.channel]
                cfg.illum = torch.tensor([0., 0., 1.], device=cfg.dev)

                wfs = []
                target_amps = []

                for _ in range(cfg.num_frames):
                    wavefront, target_amp, info = main(cfg)
                    wfs.append(wavefront.reshape(1, 1, *wavefront.shape[-2:]))
                    target_amps.append(target_amp.reshape(1, 1, *target_amp.shape[-2:]))

                wfs = torch.cat(wfs, dim=0)
                wfs = torch.nan_to_num(wfs, nan=0.0, posinf=0.0, neginf=0.0)
                target_amps = torch.cat(target_amps, dim=0)

                amp, phase = save_results(wfs, target_amps, cfg,
                                          name=f"z0_c_{channel}",
                                          out_folder=frame_folder)

                torch.cuda.empty_cache()
                amps = []
                prop_dists = np.concatenate([
                    np.linspace(cfg.z_min, cfg.z_max, cfg.num_focal_slices),
                    np.linspace(cfg.z_max, cfg.z_min, cfg.num_focal_slices)
                ])
                for prop_dist in prop_dists:
                    asm = ASM_parallel(cfg)
                    wfs2 = asm(wfs, prop_dist)
                    amp, _ = save_results(wfs2, target_amps, cfg, no_save=True)
                    amps.append(amp.squeeze(-3))

                fs_amps = torch.stack(amps)
                key = cfg.ch_str[channel]
                video_amps[key] = fs_amps
                wfs_color[key] = wfs
                target_amps_color[key] = target_amps

            save_results(wfs_color, target_amps_color,
                         cfg, out_folder=frame_folder)
            utils.save_video(video_amps, frame_folder)
            utils.save_focal_stack(video_amps, frame_folder)
            with open(os.path.join(frame_folder, "info.json"), "w") as f:
                json.dump(info, f)