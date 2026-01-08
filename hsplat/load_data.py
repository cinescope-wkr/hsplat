"""
A script to load data for hsplat.

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
import logging
import numpy as np
import tqdm
import torch
import torch.nn.functional as F

# Third-party imports
try:
    import trimesh
except ImportError:
    trimesh = None # Handled in functions that require it

# Project specific imports
import utils
from primitives import Point, Polygon, Gaussians, Polygons, Points, Gaussian
from pytorch3d.transforms import (
    matrix_to_quaternion, 
    quaternion_multiply,
    quaternion_to_matrix,
)

# gsplat imports
try:
    from gsplat.cuda._wrapper import spherical_harmonics, fully_fused_projection_2dgs
    from gsplat.rendering import rasterization_2dgs
except ImportError:
    print("Warning: gsplat not installed. 2DGS features will fail.")

logger = logging.getLogger(__name__)
# Ensure logging is configured if not already
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

DEFAULT_GS_MODEL_PATH = "models/mipnerf360_default/garden/10000/ckpts/ckpt_29999.pt"

def _get_device(dev):
    """Helper to handle mutable default arguments for device."""
    if dev is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return dev


def load_primitives(cfg):
    """
    Main entry point to load primitives based on configuration.
    """
    # Load an optimized Gaussian model to get the depth range of primitives.
    _, _, original_depth_range = load_gaussians(
        cfg.gs_model_path,
        pixel_pitch=cfg.pixel_pitch,
        wavelength=cfg.wavelength,
        target_resolution=cfg.out_resolution_hologram,
        resolution_scale_factor=cfg.culling_scale_factor,
        num_points=None,
        ch=cfg.channel,
        alpha_blending=cfg.alpha_wave_blending,
        camtoworld=cfg.camtoworld,
        K=cfg.K,
        z_range=(cfg.z_min, cfg.z_max),
        remap_sigma=cfg.remap_sigma,
        # Do not clamp if using alpha-wave-blending
        clamp_normalized_depth=not (cfg.method in ["alpha-wave-blending", "alpha_wave_blending"])
    )
    logger.info("[Original Depth Range]: %s", original_depth_range)

    # Common args to avoid repetition
    common_args = {
        "pixel_pitch": cfg.pixel_pitch,
        "wavelength": cfg.wavelength,
        "target_resolution": cfg.out_resolution_hologram,
        "ch": cfg.channel,
        "camtoworld": cfg.camtoworld,
        "K": cfg.K,
        "z_range": (cfg.z_min, cfg.z_max),
        "remap_sigma": cfg.remap_sigma,
        "original_depth_range": original_depth_range
    }

    if cfg.target_asset == "gaussians":
        path = cfg.data_path if cfg.data_path is not None else DEFAULT_GS_MODEL_PATH
        target_primitives, target_amp, _ = load_gaussians(
            path,
            resolution_scale_factor=cfg.culling_scale_factor,
            num_points=cfg.num_points,
            alpha_blending=cfg.alpha_wave_blending,
            clamp_normalized_depth=not (cfg.method in ["alpha-wave-blending", "alpha_wave_blending"]),
            **common_args
        )
        
    elif cfg.target_asset == "textured_mesh":
        # Remove wavelength from common_args for mesh as it might not be used or signature differs
        mesh_args = common_args.copy()
        del mesh_args['wavelength']
        
        target_primitives, target_amp = load_textured_mesh(
            cfg.data_path,
            **mesh_args
        )

    elif cfg.target_asset == "points_from_mesh":
        target_primitives, target_amp = load_points_from_mesh(
            cfg.data_path,
            pc_scale_multiplier=cfg.pc_scale_multiplier,
            num_points=cfg.num_points,
            load_points_as_pixels=cfg.load_points_as_pixels,
            **common_args
        )
        
    elif cfg.target_asset == "points_from_sfm":
        target_primitives, target_amp = load_points_from_sfm(
            cfg.data_path,
            pc_scale_multiplier=cfg.pc_scale_multiplier,
            num_points=cfg.num_points,
            load_points_as_pixels=cfg.load_points_as_pixels,
            **common_args
        )
        
    elif cfg.target_asset == "points_from_gaussians":
        path = cfg.data_path if cfg.data_path is not None else DEFAULT_GS_MODEL_PATH
        target_primitives, target_amp = load_points_from_gaussians(
            path,
            num_points=cfg.num_points,
            alpha_blending=cfg.alpha_wave_blending,
            pc_scale_multiplier=cfg.pc_scale_multiplier,
            **common_args
        )
    else:
        raise NotImplementedError(f"Unknown target asset: {cfg.target_asset}")
        
    return target_primitives, target_amp


def load_2dgs_ckpt(pt_path, ch, viewmat, K, width, height, num_points=None, dev=None):
    dev = _get_device(dev)
    
    # Load the Gaussians
    gaussians = torch.load(pt_path, map_location=dev)

    # Ensure all tensors are on device
    for key in gaussians.keys():
        if isinstance(gaussians[key], torch.Tensor):
            gaussians[key] = gaussians[key].to(dev)

    samples = gaussians['splats']
    logger.info("Number of Gaussians before preprocessing: %d", len(samples['means']))
    
    # Preprocessing parameters
    samples["scales"] = torch.exp(samples["scales"])  # [N, 3]
    if samples["scales"].size(-1) == 1: # handle isotropic gaussians
        samples["scales"] = samples["scales"].expand(-1, 3)
    
    samples['quats'] = F.normalize(samples['quats'], dim=-1)  # [N, 4]
    samples['sh0'] = samples['sh0'][..., ch] # [N, 1, 1]
    samples['shN'] = samples['shN'][..., ch] # [N, K, 1]

    # Compute projection to get culling mask
    radii, _, _, _, _ = fully_fused_projection_2dgs(
        samples["means"],
        samples["quats"],
        samples["scales"],
        viewmat[None],
        K[None],
        width,
        height,
        radius_clip=0.0,
        near_plane=0.2,
        far_plane=200,
    )

    camtoworlds = torch.inverse(viewmat[None])
    dirs = samples["means"][None, :, :] - camtoworlds[:, None, :3, 3] # Gaussian ray directions
    
    # Compute opacities
    if "sh0_opacities" in samples.keys():
        logger.info("[opacity]: SH")
        sh_opacities = torch.cat([samples["sh0_opacities"], samples["shN_opacities"]], dim=-2)
        sh_opacities = sh_opacities[None].expand(-1, -1, -1, 3) # [1, N, K, 3], dummy dimension for SH
        sh_opacity_degree = int(math.sqrt(sh_opacities.shape[-2]) - 1)
        
        opacities = spherical_harmonics(
            sh_opacity_degree, dirs, sh_opacities, masks=radii > 0
        )  
        opacities = opacities[..., 0] # [C, N]
        
        # Replace key and delete old ones to save memory
        samples["opacities"] = opacities.squeeze(0) # [N, ]
        del samples["sh0_opacities"]
        del samples["shN_opacities"]
    else:
        logger.info("[opacity]: sigmoid")
        samples["opacities"] = torch.sigmoid(samples["opacities"])  # [N,]

    # SH coefficients
    if samples["shN"].size(-1) > 0:
        sh = torch.cat([samples["sh0"], samples["shN"]], dim=-1)
    else:
        sh = samples["sh0"]
    sh = sh[..., None].expand(-1, -1, 3) # [N, K, 3]

    colors = spherical_harmonics(
        0, dirs, sh.unsqueeze(0), masks=radii > 0
    )[..., 0].squeeze() # [N]
    colors = torch.clamp_min(colors + 0.5, 0.0)

    return samples, radii, sh, colors


def rasterization_2dgs_wrapper(samples, sh, viewmat, K, width, height, mode="RGB"):
    # Common args
    args = (
        samples["means"], samples["quats"], samples["scales"], samples["opacities"],
        sh, viewmat[None], K[None], width, height
    )
    kwargs = {
        "sh_degree": 0,
        "near_plane": 0.2,
        "far_plane": 200,
        "render_mode": mode
    }

    try:
        # Try unpacking 7 returns (current gsplat versions)
        res = rasterization_2dgs(*args, **kwargs)
        rendered_gt, rendered_alpha = res[0], res[1]
    except ValueError:
        try:
            # Fallback for older/newer versions with 9 returns
            res = rasterization_2dgs(*args, **kwargs)
            rendered_gt, rendered_alpha = res[0], res[1]
        except Exception as e:
            logger.error(f"Rasterization failed: {e}")
            raise e

    return rendered_gt, rendered_alpha


def load_gaussians(
    pt_path, 
    pixel_pitch=1e-6, 
    wavelength=520e-9, 
    dev=None, 
    target_resolution=(1024, 1024),
    resolution_scale_factor=2.0,
    num_points=None,
    illum=None,
    ch=1,
    alpha_blending=False,
    camtoworld=None,
    K=None,
    z_range=None,
    remap_sigma=3.0,
    clamp_normalized_depth=True,
    original_depth_range=None
):
    dev = _get_device(dev)

    # Camera extrinsics and intrinsics
    height, width = target_resolution
    viewmat = torch.linalg.inv(torch.tensor(camtoworld, device=dev)).float()
    K = torch.tensor(utils.get_intrinsics_resize_to_fit(K, width, height), device=dev).float()

    # Load Gaussians from checkpoint
    samples, radii, sh, colors = load_2dgs_ckpt(pt_path, ch, viewmat, K, width, height, num_points, dev)

    # --- Use gsplat.rendering to rasterize the ground truth image ---
    render_gsplat_success = True
    logger.info(f"[alpha blending]: {alpha_blending}")
    
    if alpha_blending:
        rendered_gt, _ = rasterization_2dgs_wrapper(samples, sh, viewmat, K, width, height)
    else:
        try:
            from gsplat.rendering import rasterization_2dgs_wsr
            # Create dummy cfg variable to be portable with gsplat.rasterization_2dgs_wsr
            cfg_dummy = type('render_cfg', (object,), {'use_sh_opacity': False})()
            
            res = rasterization_2dgs_wsr(
                samples["means"], samples["quats"], samples["scales"], samples["opacities"],
                sh, viewmat[None], K[None], width, height,
                sh_degree=0, render_mode="RGB", near_plane=0.2, far_plane=200, cfg=cfg_dummy
            )
            # Handle variable return elements safely
            rendered_gt = res[0]
            rendered_gt = torch.clamp(rendered_gt, 0.0, 1.0)
        except Exception as e:
            logger.info(f"Failed to use gsplat.rasterization_2dgs_wsr: {e}")
            render_gsplat_success = False
            rendered_gt = None # Placeholder

    if render_gsplat_success and rendered_gt is not None:
        rendered_gt = rendered_gt.squeeze(0)[..., 0] # extract single channel

    # --- Transforming Gaussians to View Space ---
    means_homogeneous = torch.cat([
        samples["means"],
        torch.ones_like(samples["means"][:, :1])], 
    dim=-1)
    
    # View transform
    means_homogeneous = (viewmat @ means_homogeneous.T).T
    means = means_homogeneous[:, :3]
    samples["means"] = means.to(dev)

    # Rotation transform
    R_mat = viewmat[:3, :3]
    q_mat = matrix_to_quaternion(R_mat)
    quats = quaternion_multiply(q_mat, samples["quats"])
    quats = F.normalize(quats, dim=-1) 
    
    samples["quats"] = quats.to(dev)
    samples["scales"] = samples["scales"].to(dev)
    samples["sh0"] = samples["sh0"].to(dev)
    samples["colors"] = colors.to(dev)

    # Create collection of Gaussian primitives
    gaussian_primitives = Gaussians(**samples)
    logger.info("Number of Gaussians after preprocessing: %d", len(gaussian_primitives))
    
    gaussian_primitives.transform_perspective(K, pixel_pitch)
    sampled_idx = gaussian_primitives.sample_points(num_points)

    # --- Culling ---
    gaussian_primitives.cull_elements("gsplat_culling", radii.squeeze()[sampled_idx])
    logger.info(f"Number of Gaussians after culling: {len(gaussian_primitives)}")
    
    # Prune Gaussians with Small Scales
    gaussian_primitives.cull_elements("small_scales", 0)
    
    # Pruning Heuristics
    if True: # cull_near_90
        gaussian_primitives.cull_elements('around_90', 5 * math.pi / 180)
    
    if True: # cull_outside_canvas
        gaussian_primitives.cull_elements('outside_canvas', 
            [resolution_scale_factor * pixel_pitch * target_resolution[0] / 2, 
             resolution_scale_factor * pixel_pitch * target_resolution[1] / 2]
        )

    # --- Depth Remapping ---
    gaussian_primitives.remap_depth_range(
        z_range, remap_sigma, 
        clamp_normalized_depth=clamp_normalized_depth,
        original_depth_range=original_depth_range
    )
    gaussian_primitives.flip_z() 

    # --- Final Amplitude Generation ---
    if render_gsplat_success and rendered_gt is not None:
        logger.info("Using gsplat rendered image")
        target_amp = rendered_gt 
    else:
        logger.info("Using our own projection function")
        target_amp = orthographic_projection_2d(
            gaussian_primitives, 
            target_resolution, 
            pixel_pitch, 
            dev=dev, 
            alpha_blending=alpha_blending, 
            illum=illum
        )
        
    target_amp = torch.clamp(target_amp, 0.0, 1.0)
    return gaussian_primitives, target_amp, (gaussian_primitives.min_depth, gaussian_primitives.max_depth)


def load_textured_mesh(
    mesh_path, 
    pixel_pitch=1e-6, 
    dev=None, 
    target_resolution=(1024, 1024),
    num_points=None,
    illum=None,
    ch=1,
    camtoworld=None,
    K=None,
    z_range=None,
    remap_sigma=3.0,
    original_depth_range=None
):
    dev = _get_device(dev)

    # Camera extrinsics and intrinsics
    height, width = target_resolution
    viewmat = torch.linalg.inv(torch.tensor(camtoworld, device=dev)).float()
    K = torch.tensor(utils.get_intrinsics_resize_to_fit(K, width, height), device=dev).float()

    # Optionally load blender 2dgs models for alpha mask
    try:
        logger.info("Using gsplat rendered alpha mask for mesh rendering cleanup")
        scene_name = os.path.basename(mesh_path).split(".")[0]
        pt_path = f"models/blender_default/{scene_name}/10000/ckpts/ckpt_29999.pt"
        samples, _, sh, _ = load_2dgs_ckpt(pt_path, ch, viewmat, K, width, height, num_points, dev)
        _, rendered_alpha = rasterization_2dgs_wrapper(samples, sh, viewmat, K, width, height)
        mask = rendered_alpha.squeeze(0)[..., 0] 
        mask = torch.clamp(mask, 0.0, 1.0) > 0.5 
    except Exception as e:
        logger.info(f"Could not load alpha mask (using ones): {e}")
        mask = torch.ones(target_resolution, device=dev).float()

    loaded = np.load(mesh_path)
    textured_mesh = {key: loaded[key] for key in loaded.files}

    vertices = textured_mesh['vertices'] 
    faces = textured_mesh['faces_idx'].astype(int) 
    faces_colors = textured_mesh['faces_colors'][..., ch] 
    logger.info(f"[Loaded textured mesh]: {vertices.shape[0]} vertices, {faces.shape[0]} triangles")

    vertices = torch.tensor(vertices).float().to(dev)
    faces = torch.tensor(faces, dtype=torch.long).to(dev)
    faces_colors = torch.tensor(faces_colors).float().to(dev)

    # Reuse gsplat.fully_fused_projection_2dgs to cull triangles
    means = vertices[faces].mean(axis=1) # centroid of triangle
    quats = torch.zeros((len(means), 4), device=dev).float()
    quats[:, 0] = 1.0
    scales = torch.ones_like(means).float() * 1e-8 
    
    radii, _, _, _, _ = fully_fused_projection_2dgs(
        means, quats, scales, viewmat[None], K[None], width, height, radius_clip=0.0
    )
    
    # Transforming Triangles to View Space
    vertices_homogeneous = torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=-1)
    vertices_homogeneous = (viewmat @ vertices_homogeneous.T).T
    vertices = vertices_homogeneous[:, :3]

    R_mat = viewmat[:3, :3]
    q_mat = matrix_to_quaternion(R_mat)
    quats = quaternion_multiply(q_mat, quats)
    quats = F.normalize(quats, dim=-1)

    # Reorder faces (0, 2, 1)
    faces = torch.stack([faces[:, 0], faces[:, 2], faces[:, 1]], dim=1)

    mesh_primitives = Polygons(
        vertices[faces].to(dev), 
        opacities=torch.ones(len(faces)).to(dev),
        amplitudes=torch.ones(len(faces)).to(dev), 
        quats=quats.to(dev), 
        colors=faces_colors.to(dev)
    )
    mesh_primitives.transform_perspective(K, pixel_pitch)

    # Culling
    mesh_primitives.cull_elements("gsplat_culling", radii.squeeze())
    logger.info(f"Number of triangles after culling: {len(mesh_primitives)}")

    if True: # cull_outside_canvas
        mesh_primitives.cull_elements('outside_canvas', 
            [pixel_pitch * target_resolution[0], pixel_pitch * target_resolution[1]]
        )

    mesh_primitives.remap_depth_range(z_range, remap_sigma, original_depth_range=original_depth_range)
    mesh_primitives.flip_z()

    target_amp = orthographic_projection_2d(
        mesh_primitives, target_resolution, pixel_pitch, dev=dev, 
        alpha_blending=True, illum=illum).float()
    
    target_amp = torch.clamp(target_amp, 0.0, 1.0) * mask
    return mesh_primitives.to(dev), target_amp


def load_points_from_mesh(
    mesh_path, 
    pixel_pitch=1e-6, 
    wavelength=520e-9,
    dev=None, 
    target_resolution=(1024, 1024),
    num_points=None,
    illum=None,
    ch=1,
    camtoworld=None,
    K=None,
    z_range=None,
    remap_sigma=3.0,
    pc_scale_multiplier=1.0,
    original_depth_range=None,
    load_points_as_pixels=False
):
    dev = _get_device(dev)

    height, width = target_resolution
    viewmat = torch.linalg.inv(torch.tensor(camtoworld, device=dev)).float()
    K = torch.tensor(utils.get_intrinsics_resize_to_fit(K, width, height), device=dev).float()

    loaded = np.load(mesh_path)
    textured_mesh = {key: loaded[key] for key in loaded.files}

    vertices = textured_mesh['vertices'] 
    faces = textured_mesh['faces_idx'].astype(int)
    
    if trimesh is None:
        raise ImportError("Trimesh is required for sampling from mesh")

    points, colors = sample_from_mesh(
        vertices, faces, textured_mesh["texture_map_idx"], 
        textured_mesh["uvs"], textured_mesh["tmaps"], num_points
    )

    means = torch.tensor(points).float().to(dev)
    quats = torch.zeros((len(means), 4), device=dev).float()
    quats[:, 0] = 1.0
    scales = torch.ones_like(means).float() * 1e-8 
    
    radii, _, _, _, _ = fully_fused_projection_2dgs(
        means, quats, scales, viewmat[None], K[None], width, height, radius_clip=0.0
    )
    
    means_homogeneous = torch.cat([means, torch.ones_like(means[:, :1])], dim=-1)
    means_homogeneous = (viewmat @ means_homogeneous.T).T
    means = means_homogeneous[:, :3]
 
    scales = torch.ones_like(scales)
    opacities = torch.ones(len(means), device=dev).float()
    colors = torch.tensor(colors).float().to(dev)[..., ch]

    if load_points_as_pixels:
        points_primitives = Points(
            means=means, colors=colors, opacities=torch.ones(len(means)).to(dev)
        )
        points_primitives.transform_perspective(K, pixel_pitch)
        sampled_idx = points_primitives.sample_points(num_points)
        primitives = points_primitives
    else:
        gaussian_primitives = Gaussians(
            means=means, scales=scales, quats=quats, opacities=opacities, colors=colors
        )
        gaussian_primitives.transform_perspective(K, pixel_pitch)
        sampled_idx = gaussian_primitives.sample_points(num_points)
        gaussian_primitives.scales *= (pc_scale_multiplier * pixel_pitch / gaussian_primitives.scales.mean())
        primitives = gaussian_primitives

    primitives.cull_elements("gsplat_culling", radii.squeeze()[sampled_idx])
    
    if True: # cull_near_90
        primitives.cull_elements('around_90', 5 * math.pi / 180)

    if True: # cull_outside_canvas
        primitives.cull_elements('outside_canvas', 
            [pixel_pitch * target_resolution[0], pixel_pitch * target_resolution[1]]
        )

    primitives.remap_depth_range(z_range, remap_sigma, original_depth_range=original_depth_range)
    if not load_points_as_pixels:
        primitives.flip_z()
    else:
        primitives.set_zero_phase(wavelength)

    target_amp = orthographic_projection_2d(
        primitives, target_resolution, pixel_pitch, dev=dev, 
        alpha_blending=True, illum=illum).float()
    target_amp = torch.clamp(target_amp, 0.0, 1.0)
    return primitives.to(dev), target_amp


def load_points_from_sfm(
    sfm_path, 
    pixel_pitch=1e-6, 
    wavelength=520e-9,
    dev=None, 
    target_resolution=(1024, 1024),
    num_points=None,
    illum=None,
    ch=1,
    camtoworld=None,
    K=None,
    z_range=None,
    remap_sigma=3.0,
    pc_scale_multiplier=1.0,
    original_depth_range=None,
    load_points_as_pixels=False
):
    dev = _get_device(dev)
    try:
        import viz_utils 
    except ImportError:
        logger.error("viz_utils not found, cannot load from SFM")
        raise

    height, width = target_resolution
    viewmat = torch.linalg.inv(torch.tensor(camtoworld, device=dev)).float()
    K = torch.tensor(utils.get_intrinsics_resize_to_fit(K, width, height), device=dev).float()

    parser = viz_utils.parser.COLMAPParser(data_dir=sfm_path, normalize=True)
    means = torch.tensor(parser.points, device=dev).float()
    colors = torch.tensor(parser.points_rgb, device=dev).float() / 255.0
    colors = colors[..., ch]
    quats = torch.zeros((len(means), 4), device=dev).float()
    quats[:, 0] = 1.0
    scales = torch.ones((len(means), 3), device=dev).float() * 1e-8 
    
    radii, _, _, _, _ = fully_fused_projection_2dgs(
        means, quats, scales, viewmat[None], K[None], width, height, radius_clip=0.0
    )
    
    means_homogeneous = torch.cat([means, torch.ones_like(means[:, :1])], dim=-1)
    means_homogeneous = (viewmat @ means_homogeneous.T).T
    means = means_homogeneous[:, :3]
 
    scales = torch.ones_like(scales)
    opacities = torch.ones(len(means), device=dev).float()

    if load_points_as_pixels:
        points_primitives = Points(
            means=means, colors=colors, opacities=torch.ones(len(means)).to(dev)
        )
        points_primitives.transform_perspective(K, pixel_pitch)
        sampled_idx = points_primitives.sample_points(num_points)
        primitives = points_primitives
    else:
        gaussian_primitives = Gaussians(
            means=means, scales=scales, quats=quats, opacities=opacities, colors=colors
        )
        gaussian_primitives.transform_perspective(K, pixel_pitch)
        sampled_idx = gaussian_primitives.sample_points(num_points)
        gaussian_primitives.scales *= (pc_scale_multiplier * pixel_pitch / gaussian_primitives.scales.mean())
        primitives = gaussian_primitives

    primitives.cull_elements("gsplat_culling", radii.squeeze()[sampled_idx])
    
    if True: # cull_near_90
        primitives.cull_elements('around_90', 5 * math.pi / 180)

    if True: # cull_outside_canvas
        primitives.cull_elements('outside_canvas', 
            [pixel_pitch * target_resolution[0], pixel_pitch * target_resolution[1]]
        )

    primitives.remap_depth_range(z_range, remap_sigma, original_depth_range=original_depth_range)
    if not load_points_as_pixels:
        primitives.flip_z()
    else:
        primitives.set_zero_phase(wavelength)

    target_amp = orthographic_projection_2d(
        primitives, target_resolution, pixel_pitch, dev=dev, 
        alpha_blending=True, illum=illum).float()
    target_amp = torch.clamp(target_amp, 0.0, 1.0)
    return primitives.to(dev), target_amp


def load_points_from_gaussians(
    pt_path, 
    pixel_pitch=1e-6, 
    wavelength=520e-9, 
    dev=None, 
    target_resolution=(1024, 1024),
    num_points=None,
    illum=None,
    ch=1,
    alpha_blending=False,
    camtoworld=None,
    K=None,
    z_range=None,
    remap_sigma=3.0,
    pc_scale_multiplier=1.0,
    original_depth_range=None
):
    dev = _get_device(dev)

    height, width = target_resolution
    viewmat = torch.linalg.inv(torch.tensor(camtoworld, device=dev)).float()
    K = torch.tensor(utils.get_intrinsics_resize_to_fit(K, width, height), device=dev).float()

    samples, radii, sh, colors = load_2dgs_ckpt(pt_path, ch, viewmat, K, width, height, num_points, dev)

    rendered_gt, _ = rasterization_2dgs_wrapper(samples, sh, viewmat, K, width, height)
    rendered_gt = rendered_gt.squeeze(0)[..., 0] 
    
    means_homogeneous = torch.cat([samples["means"], torch.ones_like(samples["means"][:, :1])], dim=-1)
    means_homogeneous = (viewmat @ means_homogeneous.T).T
    means = means_homogeneous[:, :3]
    samples["means"] = means

    R_mat = viewmat[:3, :3]
    q_mat = matrix_to_quaternion(R_mat)
    quats = quaternion_multiply(q_mat, samples["quats"])
    quats = F.normalize(quats, dim=-1) 
    samples["quats"] = quats
    samples["scales"] = torch.ones_like(samples["scales"]) 
    samples["colors"] = colors

    gaussian_primitives = Gaussians(**samples)
    gaussian_primitives.transform_perspective(K, pixel_pitch)
    sampled_idx = gaussian_primitives.sample_points(num_points)

    gaussian_primitives.scales = (pc_scale_multiplier * pixel_pitch / gaussian_primitives.scales.mean())

    quats = torch.zeros_like(gaussian_primitives.quats)
    quats[:, 0] = 1.0
    gaussian_primitives.set_quats(quats)

    gaussian_primitives.cull_elements("gsplat_culling", radii.squeeze()[sampled_idx])
    gaussian_primitives.cull_elements("small_scales", 0)
    
    if True:
        gaussian_primitives.cull_elements('around_90', 5 * math.pi / 180)
    if True:
        gaussian_primitives.cull_elements('outside_canvas', 
            [pixel_pitch * target_resolution[0], pixel_pitch * target_resolution[1]]
        )

    gaussian_primitives.remap_depth_range(z_range, remap_sigma, original_depth_range=original_depth_range)
    gaussian_primitives.flip_z() 

    target_amp = orthographic_projection_2d(
        gaussian_primitives, target_resolution, pixel_pitch, 
        dev=dev, alpha_blending=alpha_blending, illum=illum
    )
    target_amp = torch.clamp(target_amp, 0.0, 1.0)
    return gaussian_primitives, target_amp


def generate_target_grid(target_resolution, dev, pixel_pitch):
    height, width = target_resolution
    target_grid_x = torch.linspace(-(width - 1) / 2, (width - 1) / 2, width, device=dev) * pixel_pitch
    target_grid_y = torch.linspace(-(height - 1) / 2, (height - 1) / 2, height, device=dev) * pixel_pitch
    target_grid_x, target_grid_y = torch.meshgrid(target_grid_x, target_grid_y, indexing='xy')
    # Meshgrid indexing='xy' removes the need for transpose usually, but preserving legacy logic if needed:
    target_grid_x = target_grid_x.transpose(0, 1)
    target_grid_y = target_grid_y.transpose(0, 1)   
    return target_grid_x, target_grid_y


def orthographic_projection_2d(primitives, 
                              target_resolution, 
                              pixel_pitch, 
                              dev=None, 
                              alpha_blending=False, 
                              illum=None):
    """
    Generate an orthographic projection from your primitives.
    Vectorized where possible, but retains loop for memory safety on large scenes.
    """
    dev = _get_device(dev)

    if isinstance(primitives, Points):
        return add_point_scatter(primitives, target_resolution, pixel_pitch)

    elif isinstance(primitives, (Polygons, Gaussians)):
        target_grid_x, target_grid_y = generate_target_grid(target_resolution, dev, pixel_pitch)
        target_amp = torch.zeros_like(target_grid_x)
        T = torch.ones_like(target_grid_y)

        # Sort for correct composition (Painter's Algorithm)
        if hasattr(primitives, 'sort'):
            primitives.sort('front2back')
            
        for primitive in tqdm.tqdm(primitives, desc="Projecting primitives"):
            contribution = None
            
            if isinstance(primitive, Gaussian):
                G = add_gaussian_2d_projection(primitive, target_grid_x, target_grid_y)
                
                # Skip if Gaussian is invalid or zero contribution
                if G.sum() == 0:
                    continue

                if primitive.color is not None:
                    c = primitive.color 
                else:
                    c = primitive.get_sh_color(torch.tensor([0., 0., -1.], device=dev))
                    c = torch.clamp_min(c + 0.5, 0.0)
                
                contribution = primitive.opacity * G * c
                target_amp = target_amp + T * contribution

                if alpha_blending:
                    T = T * (1 - primitive.opacity * G)

            elif isinstance(primitive, Polygon):
                P = add_polygon_2d_projection(primitive, target_grid_x, target_grid_y)
                
                if primitive.color is not None:
                    c = primitive.color 
                else:
                    c = primitive.shade_illumination(illum)
                
                contribution = primitive.opacity * P * c
                target_amp = target_amp + T * contribution
                
                if alpha_blending:
                    T = T * (1 - primitive.opacity * P)

            # --- Critical fix: Handle NaNs gracefully ---
            if torch.isnan(target_amp).any():
                logger.warning("NaN detected in projection. Resetting grid to avoid crash.")
                return torch.zeros_like(target_grid_x)

    return target_amp


def add_point_scatter(primitives, target_resolution, pixel_pitch):
    height, width = target_resolution
    positions = primitives.means[:, :2]
    # Scale x and y coordinates to image space
    x = (positions[:, 0] / pixel_pitch + width // 2).clamp(0, width - 1).long()
    y = (positions[:, 1] / pixel_pitch + height // 2).clamp(0, height - 1).long()
    
    image = torch.zeros(height, width, device=positions.device)
    indices = y * width + x
    image.view(-1).scatter_add_(0, indices, primitives.colors)
    return image


def add_polygon_2d_projection(primitive, target_grid_x, target_grid_y, eps=0.0, shrink_factor=1.0):
    v1 = primitive.v1.squeeze()
    v2 = primitive.v2.squeeze()
    v3 = primitive.v3.squeeze()    

    centroid = (v1[:2] + v2[:2] + v3[:2]) / 3.0
    v1[:2] = centroid + shrink_factor * (v1[:2] - centroid)
    v2[:2] = centroid + shrink_factor * (v2[:2] - centroid)
    v3[:2] = centroid + shrink_factor * (v3[:2] - centroid)
    
    target_amp = torch.zeros_like(target_grid_x)
    points = torch.stack([target_grid_x.flatten(), target_grid_y.flatten()], dim=1)
    tri_vertices = torch.stack([v1[:2], v2[:2], v3[:2]])
    
    v0 = tri_vertices[1] - tri_vertices[0] 
    v1_vec = tri_vertices[2] - tri_vertices[0]
    v2_vec = points - tri_vertices[0]
    
    d00 = (v0 * v0).sum(dim=-1)
    d01 = (v0 * v1_vec).sum(dim=-1)
    d11 = (v1_vec * v1_vec).sum(dim=-1)
    d20 = (v2_vec * v0).sum(dim=-1)
    d21 = (v2_vec * v1_vec).sum(dim=-1)
    
    denom = d00 * d11 - d01 * d01
    # Avoid division by zero
    denom[denom == 0] = 1e-6
    
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    
    inside = (u >= 0+eps) & (v >= 0+eps) & (w >= 0+eps)
    target_amp.flatten()[inside] += 1.0

    return target_amp


def add_gaussian_2d_projection(primitive, target_grid_x, target_grid_y):
    """
    Computes 2D Gaussian projection.
    Refactored to use analytic 2x2 inverse for speed and stability.
    """
    device = primitive.device
    
    try:
        R = quaternion_to_matrix(primitive.quat)
        scales = primitive.scale
        scales[2] = 0.0 
        D = torch.diag(scales**2)
        Sigma_3D = R @ D @ R.T
        Sigma_2D = Sigma_3D[:2, :2]

        # --- Analytic Inverse of 2x2 Matrix ---
        # [[a, b], [c, d]]^-1 = 1/det * [[d, -b], [-c, a]]
        a, b = Sigma_2D[0, 0], Sigma_2D[0, 1]
        c, d = Sigma_2D[1, 0], Sigma_2D[1, 1]
        
        det = a * d - b * c
        if torch.abs(det) < 1e-9:
             # Singular matrix, return empty
             return torch.zeros_like(target_grid_x)

        inv_det = 1.0 / det
        Sigma_2D_inv = torch.tensor([
            [d * inv_det, -b * inv_det],
            [-c * inv_det, a * inv_det]
        ], device=device)
        
        # Calculate differences
        x_diff = (primitive.mean[0] - target_grid_x).float()
        y_diff = (primitive.mean[1] - target_grid_y).float()

        # [H, W, 2] stack
        diff = torch.stack([x_diff, y_diff], dim=-1)
        
        # Quadratic form: (diff @ Sigma_inv) * diff
        # Einsum: ...i is (x, y), ij is Sigma_inv, ...j is result
        XA = torch.einsum('...i,ij->...j', diff, Sigma_2D_inv) 
        quad_form = torch.einsum('...i,...i->...', diff, XA)

        gaussian_2d = torch.exp(-0.5 * quad_form)

        # Safety check
        if torch.isnan(gaussian_2d).any() or torch.isinf(gaussian_2d).any() or gaussian_2d.max() > 1e5:
            return torch.zeros_like(gaussian_2d)
        else:
            return gaussian_2d
            
    except Exception as e:
        # Catch unexpected errors to prevent total crash
        return torch.zeros_like(target_grid_x)


def sample_from_mesh(vertices, faces, texture_map_idx, uvs, texture_maps, num_points=None):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    if num_points is None:
        num_points = faces.shape[0]
    
    logger.info(f"[Sampling from mesh]: {num_points} points")
    np.random.seed(0)
    sampled_points, face_indices = trimesh.sample.sample_surface(mesh, num_points, seed=0)
    
    sampled_colors = []
    # Using list accumulation is slow, but kept for compatibility. 
    # Consider vectorizing this if texture lookups allow.
    for i in tqdm.tqdm(range(sampled_points.shape[0]), desc="Sampling textures"):
        pos = sampled_points[i] 
        face_idx = face_indices[i] 
        tri = vertices[faces[face_idx]]
        tmap = texture_maps[texture_map_idx[face_idx]]
        uv = uvs[face_idx]
        
        abc = utils.compute_barycentric_coords(pos, tri)
        lerp_uv = np.sum(abc[..., None] * uv, axis=0)
        textured_h, textured_w, _ = tmap.shape
        pixel_loc = lerp_uv * np.array([textured_w, textured_h])
        pixel_loc = pixel_loc.astype(int) 
        
        # Clamp to avoid index out of bounds
        px = np.clip(pixel_loc[0], 0, textured_w - 1)
        py = np.clip(pixel_loc[1], 0, textured_h - 1)
        
        colors = tmap[py, px]
        sampled_colors.append(colors)
        
    return sampled_points, np.array(sampled_colors)


def preselect_random_points(num_points, *args):
    np.random.seed(0)
    num_total = args[0].shape[0]
    # Ensure we don't sample more than exists
    eff_num_points = min(num_points, num_total)
    
    random_indices = np.random.choice(num_total, eff_num_points, replace=False)
    results = []
    for arg in args:
        results.append(arg[random_indices])
    return tuple(results)


def load_point_cloud_from_ply(ply_path, 
                              pixel_pitch=1e-6, 
                              wavelength=520e-9, 
                              dev=None, 
                              target_resolution=(540, 960), scale=100, num_points=None):
    dev = _get_device(dev)
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("open3d is required for load_point_cloud_from_ply. Install it with: pip install open3d")
    
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    if num_points is not None:
        points, colors = preselect_random_points(num_points, points, colors)

    logger.info(f"{points.shape} points.shape")
    
    positions = torch.tensor(points, dtype=torch.float32, device=dev)
    positions = positions - positions.median(dim=-2, keepdim=True)[0]
    positions = positions / scale
    
    amplitudes = torch.tensor(colors, dtype=torch.float32, device=dev)
    amplitudes = torch.ones(len(points), dtype=torch.float32, device=dev)
    
    if 'bun' in ply_path:
        positions[:, -2] = positions[:, -2] * -1 

    z = positions[:, -1]
    phases = -2 * math.pi * z / wavelength
    
    point_cloud = Points()
    point_cloud.add_points_batch(positions, amplitudes, phases)
    target_amp = orthographic_projection_2d(point_cloud, target_resolution, pixel_pitch, dev=dev, alpha_blending=False)
    
    return point_cloud, target_amp


def load_mesh_from_ply(ply_path, 
                       pixel_pitch=1e-6, 
                       wavelength=520e-9, 
                       dev=None, 
                       scale=100, 
                       target_resolution=(1000, 1000),
                       output='default',
                       illum=None):
    dev = _get_device(dev)
    assert os.path.exists(ply_path), f'{ply_path} does not exist'

    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("open3d is required for load_mesh_from_ply")

    mesh = o3d.io.read_triangle_mesh(ply_path)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)    
    
    if np.asarray(mesh.vertex_colors).size == 0:
        logger.info("No color information found in the PLY file. Using default white color.")
        colors = np.ones((len(vertices), 3), dtype=np.float32) 
    else:
        colors = np.asarray(mesh.vertex_colors)
    
    vertices = torch.tensor(vertices, dtype=torch.float32, device=dev)
    faces = torch.tensor(faces, dtype=torch.int64, device=dev)
    colors = torch.tensor(colors, dtype=torch.float32, device=dev)
    
    vertices = vertices - vertices.mean(dim=-2, keepdim=True)
    
    if 'bun' in ply_path:
        vertices[:, -2] = vertices[:, -2] * -1      
        
    vertices = vertices / scale 
    
    if output in ['polygons', 'default']:
        mesh_primitives = Polygons(vertices[faces], opacities=torch.ones(len(faces)), amplitudes=0.25 * torch.ones(len(faces))).to(dev)
    elif output == 'points':
        means = vertices[faces]
        normal = torch.cross(means[:, 1, ...] - means[:, 2, ...], means[:, 0, ...] - means[:, 2, ...])
        normal = normal / torch.norm(normal, dim=-1, keepdim=True)
        mesh_primitives = Points(vertices[faces], torch.ones(len(faces)), normal=normal)
    elif output == 'gaussians':
        data_dict = {}
        data_dict['means'] = vertices[faces].mean(dim=1)
        
        # Provide default scales/quats
        scales = torch.ones(faces.shape[0], 3, device=dev) * 8e-5 
        scales[:, 2] = 0.0
        data_dict['scales'] = scales

        quats = torch.zeros(faces.shape[0], 4, device=dev)
        quats[:, 0] = 1.0 # Identity quaternion
        
        mesh_primitives = Gaussians(**data_dict, opacities=torch.ones(faces.shape[0], device=dev), quats=quats, sh0=torch.ones_like(faces), shN=None)

    target_amp = orthographic_projection_2d(mesh_primitives, target_resolution, pixel_pitch, dev=dev, alpha_blending=True, illum=illum)
    
    return mesh_primitives.to(dev), target_amp