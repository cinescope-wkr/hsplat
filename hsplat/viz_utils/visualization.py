import numpy as np

def get_focus_point(direction, origin):
    """
        direction: [N, 3]
        origin: [N, 3]
    """
    m = np.eye(3) - direction[..., None] * direction[..., None, :]
    mt_m = m.mT @ m
    mt_m_inv = np.linalg.inv(mt_m.mean(axis=0))
    focus_pt = mt_m_inv @ (mt_m @ origin[..., None]).mean(axis=0)
    return focus_pt.squeeze()

def view_pose(lookdir, up):
    def normalize(x):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)
    
    vec2 = normalize(lookdir)
    vec0 = normalize(np.linalg.cross(up[None, :], vec2, axis=-1))
    vec1 = normalize(np.linalg.cross(vec2, vec0, axis=-1))
    return np.stack([vec0, vec1, vec2], axis=-1)

def get_ellipse_path(
    cam2worlds,
    num_frames: int,
    z_variation: float = 0.0,
    z_phase: float = 0.0,
    ellipse_size_factor: float = 1.0
):
    """
        cam2worlds: [N, 4, 4]
    """
    rotations = cam2worlds[:, :3, :3] # [N, 3, 3]
    translations = cam2worlds[:, :3, 3] # [N, 3]

    # get focus point
    center = get_focus_point(rotations[:, :, 2], translations)
    offset = np.array([center[0], center[1], 0])

    # Calculate scaling for ellipse axes based on input camera positions
    sc = np.quantile(np.abs(translations - offset), 0.9, axis=0)
    sc = sc * ellipse_size_factor

    # Use ellipse that is symmetric about the focal point in xy
    low = -sc + offset
    high = sc + offset

    # Optional height variation
    z_avg = translations.mean(axis=0)
    z_low = np.quantile(translations - z_avg, 0.1, axis=0)
    z_high = np.quantile(translations - z_avg, 0.9, axis=0)

    # Compute positions along the ellipse
    def get_positions(theta):
        return np.stack(
            [
                low[0] + (high - low)[0] * (np.cos(theta) * 0.5 + 0.5),
                low[1] + (high - low)[1] * (np.sin(theta) * 0.5 + 0.5),
                z_avg[2] 
                + z_variation
                * (
                    z_low[2]
                    + (z_high - z_low)[2]
                    * (np.cos(theta + 2 * np.pi * z_phase) * 0.5 + 0.5)
                )
            ],
            axis=-1
        )
    
    thetas = np.linspace(0, 2 * np.pi, num_frames + 1)
    positions = get_positions(thetas)[:-1]

    # Set path's up vector to axis cloest to average of input pose up vectors
    avg_up = rotations[:, :, 1].mean(axis=0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    rotations = view_pose(center - positions, up)
    path_cam2worlds = np.concatenate([rotations, positions[..., None]], axis=-1)
    path_cam2worlds = np.concatenate([
        path_cam2worlds, # [N, 3, 4]
        np.stack([np.array([0, 0, 0, 1])] * path_cam2worlds.shape[0])[:, None]
    ], axis=1)

    return path_cam2worlds

def interpolate_path(cam2worlds, num_frames):
    """
    Interpolates a sequence of camera poses.
    
    Args:
        cam2worlds (np.ndarray): Array of shape [N, 4, 4] representing N camera-to-world transformation matrices.
        num_frames (int): Total number of interpolated frames, including original keyframes.
    
    Returns:
        np.ndarray: Array of shape [num_frames, 4, 4], representing the interpolated poses.
    """
    num_poses = cam2worlds.shape[0]
    
    # Compute the total interpolation steps between each pair of poses
    total_intervals = num_frames - 1
    steps_per_interval = total_intervals / (num_poses - 1)
    
    interpolated_poses = []

    for i in range(num_poses - 1):
        pose1 = cam2worlds[i]
        pose2 = cam2worlds[i + 1]

        # Extract rotation and translation components
        t1 = pose1[:3, 3]
        t2 = pose2[:3, 3]

        # Determine how many frames to interpolate for this interval
        if i < num_poses - 2:
            num_steps = int(np.round(steps_per_interval))
        else:
            # Include all remaining frames in the last segment
            num_steps = total_intervals - len(interpolated_poses) + 1

        # Interpolate rotation using Slerp
        slerp = Slerp([0, 1], R.from_matrix([pose1[:3, :3], pose2[:3, :3]]))
        interp_times = np.linspace(0, 1, num_steps + 1)

        # Interpolate translation using linear interpolation
        for t in interp_times[:-1]:  # Exclude the last frame of the interval to avoid duplication
            interpolated_rotation = slerp([t])[0].as_matrix()
            interpolated_translation = (1 - t) * t1 + t * t2

            # Construct the interpolated pose
            interpolated_pose = np.eye(4)
            interpolated_pose[:3, :3] = interpolated_rotation
            interpolated_pose[:3, 3] = interpolated_translation
            interpolated_poses.append(interpolated_pose)

    # Append the last pose
    interpolated_poses.append(cam2worlds[-1])

    return np.array(interpolated_poses)
