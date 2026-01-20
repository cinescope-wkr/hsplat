import os
import json
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import assert_never

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
#from pycolmap import SceneManager
import pycolmap
from pycolmap import Reconstruction 

from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths

def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """COLMAP qvec (qw,qx,qy,qz) -> 3x3 rotation matrix."""
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)

def _get_w2c_rt_from_pycolmap_image(im) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (R, t) for world->camera transform: x_cam = R * x_world + t
    Supports multiple pycolmap API variants.
    """
    # Most common in pycolmap: qvec/tvec
    if hasattr(im, "qvec") and hasattr(im, "tvec"):
        R = _qvec_to_rotmat(np.asarray(im.qvec, dtype=np.float64))
        t = np.asarray(im.tvec, dtype=np.float64)
        return R, t

    # Some variants expose rotmat/tvec
    if hasattr(im, "rotmat") and hasattr(im, "tvec"):
        R = np.asarray(im.rotmat, dtype=np.float64)
        t = np.asarray(im.tvec, dtype=np.float64)
        return R, t

    # Some newer APIs: cam_from_world (callable or attribute)
    if hasattr(im, "cam_from_world"):
        cf = getattr(im, "cam_from_world")
        pose = cf() if callable(cf) else cf
        # rotation.matrix() 형태
        if hasattr(pose, "rotation") and hasattr(pose.rotation, "matrix"):
            R = np.asarray(pose.rotation.matrix(), dtype=np.float64)
        else:
            R = np.asarray(pose.rotation, dtype=np.float64)
        t = np.asarray(getattr(pose, "translation"), dtype=np.float64)
        return R, t

    raise AttributeError(
        "pycolmap.Image에서 world->camera (R,t)를 얻을 수 없음. "
        "지원: (qvec,tvec) 또는 (rotmat,tvec) 또는 cam_from_world"
    )

def _camera_params_to_K_and_dist(cam, factor: int) -> Tuple[np.ndarray, np.ndarray, str]:
    model = cam.model if hasattr(cam, "model") else getattr(cam, "camera_type", None)
    params = np.asarray(cam.params, dtype=np.float64) if hasattr(cam, "params") else None

    # ---- Fallback: some pycolmap builds don't expose model/camera_type ----
    if model is None:
        if params is None:
            raise ValueError("Camera has neither model nor params; cannot build intrinsics.")
        # Heuristic by parameter length (most common)
        if len(params) == 4:
            # assume PINHOLE: [fx, fy, cx, cy]
            fx, fy, cx, cy = params[:4]
            dist = np.empty(0, dtype=np.float32)
            camtype = "perspective"
        elif len(params) == 3:
            # assume SIMPLE_PINHOLE: [f, cx, cy]
            f, cx, cy = params[:3]
            fx, fy = f, f
            dist = np.empty(0, dtype=np.float32)
            camtype = "perspective"
        elif len(params) == 8:
            # ambiguous: could be OPENCV [fx,fy,cx,cy,k1,k2,p1,p2]
            fx, fy, cx, cy, k1, k2, p1, p2 = params[:8]
            dist = np.array([k1, k2, p1, p2], dtype=np.float32)
            camtype = "perspective"
        else:
            raise ValueError(f"Unsupported camera params length={len(params)} with model=None params={params}")

        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        K[:2, :] /= float(factor)
        return K, dist, camtype
    # ---------------------------------------------------------------------

    model_str = str(model)

    # (아래는 기존 model별 분기 그대로)
    if ("SIMPLE_PINHOLE" in model_str) or (model == 0):
        f, cx, cy = params[:3]
        fx, fy = f, f
        dist = np.empty(0, dtype=np.float32)
        camtype = "perspective"

    elif ("PINHOLE" in model_str) or (model == 1):
        fx, fy, cx, cy = params[:4]
        dist = np.empty(0, dtype=np.float32)
        camtype = "perspective"

    elif ("SIMPLE_RADIAL" in model_str) or (model == 2):
        f, cx, cy, k1 = params[:4]
        fx, fy = f, f
        dist = np.array([k1, 0.0, 0.0, 0.0], dtype=np.float32)
        camtype = "perspective"

    elif ("RADIAL" in model_str) or (model == 3):
        f, cx, cy, k1, k2 = params[:5]
        fx, fy = f, f
        dist = np.array([k1, k2, 0.0, 0.0], dtype=np.float32)
        camtype = "perspective"

    elif ("OPENCV_FISHEYE" in model_str) or (model == 5):
        fx, fy, cx, cy, k1, k2, k3, k4 = params[:8]
        dist = np.array([k1, k2, k3, k4], dtype=np.float32)
        camtype = "fisheye"

    elif ("OPENCV" in model_str) or (model == 4):
        fx, fy, cx, cy, k1, k2, p1, p2 = params[:8]
        dist = np.array([k1, k2, p1, p2], dtype=np.float32)
        camtype = "perspective"

    else:
        raise ValueError(f"Unsupported camera model: {model_str} with params={params}")

    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    K[:2, :] /= float(factor)
    return K, dist, camtype




class COLMAPParser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        #manager = SceneManager(colmap_dir)
        manager = pycolmap.Reconstruction(colmap_dir)
        #manager.load_cameras()
        #manager.load_images()
        #manager.load_points3D()

        # Extract extrinsic matrices in world-to-camera format.
        imdata = manager.images
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        mask_dict = dict()
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            #rot = im.R()
            #trans = im.tvec.reshape(3, 1)
            rot, t = _get_w2c_rt_from_pycolmap_image(im)
            trans = t.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)
            """
            # camera intrinsics
            cam = manager.cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            type_ = cam.camera_type
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            if type_ == 2 or type_ == "SIMPLE_RADIAL":
                params = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 3 or type_ == "RADIAL":
                params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 4 or type_ == "OPENCV":
                params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
                camtype = "fisheye"
            assert (
                camtype == "perspective" or camtype == "fisheye"
            ), f"Only perspective and fisheye cameras are supported, got {type_}"

            params_dict[camera_id] = params
            imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)
            mask_dict[camera_id] = None
            """
            cam = manager.cameras[camera_id]

            K, params, camtype = _camera_params_to_K_and_dist(cam, factor=factor)
            Ks_dict[camera_id] = K.astype(np.float64)

            params_dict[camera_id] = params
            imsize_dict[camera_id] = (int(cam.width) // factor, int(cam.height) // factor)
            mask_dict[camera_id] = None

        print(
            f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras."
        )

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")
        #if not (type_ == 0 or type_ == 1):
        #    print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        image_names = [imdata[k].name for k in imdata]

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load extended metadata. Used by Bilarf dataset.
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Load bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        # Load images.
        if factor > 1 and not self.extconf["no_factor_suffix"]:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        """
        # 3D points and {image_name -> [point_idx]}
        points = manager.points3D.astype(np.float32)
        points_err = manager.point3D_errors.astype(np.float32)
        points_rgb = manager.point3D_colors.astype(np.uint8)
        point_indices = dict()

        image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
        for point_id, data in manager.point3D_id_to_images.items():
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = manager.point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(image_name, []).append(point_idx)
        point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        }
        """
        # 3D points and {image_name -> [point_idx]}  (pycolmap-compatible)
        points3D_map = manager.points3D  # MapPoint3DIdPoint3D: {point3D_id -> Point3D}

        point_ids = list(points3D_map.keys())
        pts = [points3D_map[pid] for pid in point_ids]

        # xyz / color / error
        points = np.array([p.xyz for p in pts], dtype=np.float32)
        points_rgb = np.array([p.color for p in pts], dtype=np.uint8)
        points_err = np.array([p.error for p in pts], dtype=np.float32)

        # Build mapping: point3D_id -> contiguous index in points array
        point3D_id_to_point3D_idx = {pid: i for i, pid in enumerate(point_ids)}

        # Build mapping: image_name -> [point_idx, ...]
        point_indices: Dict[str, List[int]] = {}

        # images is a map: {image_id -> Image}; Image has .name and (usually) .points2D
        # Each Point2D has .point3D_id (or .point3D_id()) depending on binding
        for image_id, im in manager.images.items():
            image_name = im.name
            idx_list: List[int] = []

            if hasattr(im, "points2D"):
                for p2d in im.points2D:
                    # pycolmap variants: attribute or method
                    pid = p2d.point3D_id() if callable(getattr(p2d, "point3D_id", None)) else getattr(p2d, "point3D_id", -1)
                    if pid is None:
                        continue
                    # In COLMAP, invalid point id is often -1
                    if isinstance(pid, (int, np.integer)) and pid < 0:
                        continue
                    if pid in point3D_id_to_point3D_idx:
                        idx_list.append(point3D_id_to_point3D_idx[pid])

            if len(idx_list) > 0:
                point_indices[image_name] = np.array(idx_list, dtype=np.int32)

        # Ensure all images exist as keys (optional; keeps downstream code simpler)
        for name in image_names:
            point_indices.setdefault(name, np.zeros((0,), dtype=np.int32))


        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principle_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1
        else:
            transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)

        # load one image to check the size. In the case of tanksandtemples dataset, the
        # intrinsics stored in COLMAP corresponds to 2x upsampled images.
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert (
                camera_id in self.params_dict
            ), f"Missing params for camera {camera_id}"
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]

            if camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    K, params, (width, height), 0
                )
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, params, None, K_undist, (width, height), cv2.CV_32FC1
                )
                mask = None
            elif camtype == "fisheye":
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = (
                    1.0
                    + params[0] * theta**2
                    + params[1] * theta**4
                    + params[2] * theta**6
                    + params[3] * theta**8
                )
                mapx = (fx * x1 * r + width // 2).astype(np.float32)
                mapy = (fy * y1 * r + height // 2).astype(np.float32)

                # Use mask to define ROI
                mask = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < width - 1, mapy < height - 1),
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                assert_never(camtype)

            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
            self.mask_dict[camera_id] = mask

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)
        self.scene_scale_nerf2mesh = 1.0 / np.min(dists)
        print(f"[Parser] Scene scale: {self.scene_scale:.2f}")
        print(f"[Parser] Scene scale (nerf2mesh): {self.scene_scale_nerf2mesh:.2f}")

    def get_image(self, idx: int) -> np.ndarray:
        """ Load image by index. """
        test_frame_idx_list = [idx for idx in range(len(self.camtoworlds)) if idx % self.test_every == 0]
        test_idx = test_frame_idx_list[idx]
        img = imageio.imread(self.image_paths[test_idx]) / 255.0
        return img
    
class BlenderParser:
    """ A simple synthetic Blender dataset parser class. """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
    ):
        self.data_dir = data_dir
        self.split = split
        self.image_size = 800
        
        # Loads json file that defines camtoworlds and intrinrics
        json_path = os.path.join(self.data_dir, f"transforms_{self.split}.json")
        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)
        
        # Compute camera intrinsics
        self.camera_angle = json_data["camera_angle_x"] * 0.5
        c = self.image_size // 2 # pricipal point in pixels
        f = c / np.tan(self.camera_angle)
        self.K = np.array([
            [f, 0, c],
            [0, f, c],
            [0, 0, 1]
        ], dtype=np.float32)

        # Load images camera extrinsics
        self.camtoworlds = []
        for frame_data in json_data["frames"]:
            camtoworld = np.array(frame_data["transform_matrix"])
            # Adjust OpenGL v.s. COLMAP coordinate convension (Y-up, Z-back to Y-down, Z-forward)
            camtoworld[:3, 1:3] *= -1
            self.camtoworlds.append(camtoworld)
        self.camtoworlds = np.stack(self.camtoworlds) # [N, 4, 4]
    

    def get_image(self, idx: int) -> np.ndarray:
        """ Load image by index. """
        img_path = os.path.join(self.data_dir, f"{self.split}/r_{idx}.png")
        img = imageio.imread(img_path) / 255.0
        rgb, alpha = img[..., :3], img[..., 3:4]
        img = rgb * alpha + 0 * (1 - alpha)
        return img
