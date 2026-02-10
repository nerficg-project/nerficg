"""
Visual/Trajectories/Ellipse.py: A camera trajectory for inward-facing captures, following an elliptical path around scene center.
Adapted from ZipNeRF (https://github.com/jonbarron/camp_zipnerf).
"""

import numpy as np
import torch

import Framework
from Cameras.Base import BaseCamera
from Cameras.Perspective import PerspectiveCamera
from Datasets.utils import View, compute_scaled_image_size, transform_poses_pca, rescale_poses_to_unit_cube
from Visual.Trajectories.utils import CameraTrajectory


class ellipse_path(CameraTrajectory):
    """A camera trajectory for inward-facing captures, following an elliptical path around scene center."""

    def __init__(self, n_views: int = 480, resolution: int = None) -> None:
        super().__init__()
        self.n_views = n_views  # 480 for 60 FPS mipNeRF360 result videos
        self.resolution = resolution  # None for original resolution, 720 for 720p, 1080 for 1080p, etc.

    def _generate(self, default_camera: BaseCamera, reference_views: list[View]) -> list[View]:
        """Generates the camera trajectory using a list of reference views."""
        camera = PerspectiveCamera(
            shared_settings=default_camera.shared_settings, width=default_camera.width, height=default_camera.height,
        )
        if isinstance(default_camera, PerspectiveCamera):
            camera.focal_x = default_camera.focal_x
            camera.focal_y = default_camera.focal_y
            camera.center_x = default_camera.center_x
            camera.center_y = default_camera.center_y
            # TODO: could also copy distortion, but do we want that?
            # camera.distortion = deepcopy(default_camera.distortion)
        # create ellipse path around the camera positions for mipNeRF360-style videos
        ellipse_path_poses = generate_ellipse_path(
            poses=np.stack([view.c2w_numpy for view in reference_views]),
            n_views=self.n_views,
            z_variation=0.0,  # How much height variation in render path.
            z_phase=0.0,  # Phase offset for height variation in render path.
            rad_mult_min=1.0,  # How close to get to the object, relative to 1.
            rad_mult_max=1.0,  # How far to get from the object, relative to 1.
            render_rotate_xaxis=0.0,  # Rotate camera around x-axis.
            render_rotate_yaxis=0.0,  # Rotate camera around y-axis.
            lock_up=False,  # If True, locks the up axis (good for sideways paths).
        )
        if self.resolution is not None:
            scale = self.resolution / camera.height
            target_width, target_height = compute_scaled_image_size((camera.width, camera.height), scale)
            scale_x = target_width / camera.width
            scale_y = target_height / camera.height
            camera.width, camera.height = target_width, target_height
            camera.focal_x *= scale_x
            camera.focal_y *= scale_y
            camera.center_x *= scale_x
            camera.center_y *= scale_y
        views = []
        for frame_idx, pose in enumerate(ellipse_path_poses):
            views.append(View(
                camera=camera,
                camera_index=0,
                frame_idx=frame_idx,
                global_frame_idx=0,
                c2w=pose,
            ))
        return views


def generate_ellipse_path(
    poses: np.ndarray,
    n_views: int = 120,
    z_variation: float = 0.0,
    z_phase: float = 0.0,
    rad_mult_min: float = 1.0,
    rad_mult_max: float = 1.0,
    render_rotate_xaxis: float = 0.0,
    render_rotate_yaxis: float = 0.0,
    lock_up: bool = False,
) -> np.ndarray:
    """
    Generate an elliptical render path based on the given poses.
    This function and all sub-functions are adapted from Zip-NeRF (https://github.com/jonbarron/camp_zipnerf).
    """
    # This function requires rescaled, pca-aligned poses so we need to check what was already applied.
    # Note: During data loading transform_poses_pca always uses all views, while this function may be called
    # with only a subset, which usually results in slightly different alignments.
    needs_pca = not Framework.config.DATASET.get('APPLY_PCA', False)
    if needs_pca:
        # Apply PCA alignment without rescaling to unit cube.
        poses, pca_transform = transform_poses_pca(poses, rescale=False)
    else:
        pca_transform = np.eye(4)

    # Convert to OpenGL/NeRF camera coordinate system for direct compatibility with Zip-NeRF utils.
    colmap2opengl = np.diag([1, -1, -1, 1])
    poses = poses @ colmap2opengl

    # Unapply the aligned2colmap transformation that is applied in transform_poses_pca.
    aligned2colmap = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]]
    )
    colmap2aligned = np.linalg.inv(aligned2colmap)
    poses = colmap2aligned @ poses
    pca_transform = colmap2aligned @ pca_transform

    needs_pca_rescale = needs_pca or not Framework.config.DATASET.get('APPLY_PCA_RESCALE', False)
    if needs_pca_rescale:
        # Rescale to unit cube.
        poses, pca_transform = rescale_poses_to_unit_cube(poses, pca_transform)

    def focus_point_fn(poses):
        """Calculate the nearest point to all focal axes in poses."""
        directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
        m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
        mt_m = np.transpose(m, [0, 2, 1]) @ m
        focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
        return focus_pt

    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Default path height sits at z=0 (in middle of zero-mean capture pattern).
    xy_offset = center[:2]

    # Calculate lengths for ellipse axes based on input camera positions.
    xy_radii = np.percentile(np.abs(poses[:, :2, 3] - xy_offset), 90, axis=0)
    # Use ellipse that is symmetric about the focal point in xy.
    xy_low = xy_offset - xy_radii
    xy_high = xy_offset + xy_radii

    # Optional height variation, need not be symmetric.
    z_min = np.percentile((poses[:, 2, 3]), 10, axis=0)
    z_max = np.percentile((poses[:, 2, 3]), 90, axis=0)
    # Center the path at zero, good for datasets recentered by transform_poses_pca function.
    z_init = 0
    z_low = z_init + z_variation * (z_min - z_init)
    z_high = z_init + z_variation * (z_max - z_init)

    xyz_low = np.array([*xy_low, z_low])
    xyz_high = np.array([*xy_high, z_high])

    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        t_x = np.cos(theta) * 0.5 + 0.5
        t_y = np.sin(theta) * 0.5 + 0.5
        t_z = np.cos(theta + 2 * np.pi * z_phase) * 0.5 + 0.5
        t_xyz = np.stack([t_x, t_y, t_z], axis=-1)
        positions = xyz_low + t_xyz * (xyz_high - xyz_low)
        # Interpolate between min and max radius multipliers so the camera zooms in
        # and out of the scene center.
        t = np.sin(theta) * 0.5 + 0.5
        rad_mult = rad_mult_min + (rad_mult_max - rad_mult_min) * t
        positions = center + (positions - center) * rad_mult[:, None]
        return positions

    theta = np.linspace(0, 2.0 * np.pi, n_views + 1, endpoint=True)
    positions = get_positions(theta)
    # Resample theta angles so that the velocity is closer to constant.
    lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
    theta = sample(theta, np.log(lengths), n_views + 1)
    positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    def viewmatrix(lookdir, up, position, lock_up=False):
        """Construct lookat view matrix."""

        def normalize(x):
            return x / np.linalg.norm(x)

        def orthogonal_dir(a, b):
            return normalize(np.cross(a, b))

        vecs = [None, normalize(up), normalize(lookdir)]
        # x-axis is always the normalized cross product of `lookdir` and `up`.
        vecs[0] = orthogonal_dir(vecs[1], vecs[2])
        # Default is to lock `lookdir` vector, if lock_up is True lock `up` instead.
        ax = 2 if lock_up else 1
        # Set the not-locked axis to be orthogonal to the other two.
        vecs[ax] = orthogonal_dir(vecs[(ax + 1) % 3], vecs[(ax + 2) % 3])
        m = np.stack(vecs + [position], axis=1)
        return m

    poses = np.stack([viewmatrix(p - center, up, p, lock_up) for p in positions])

    def rotation_about_axis(degrees, axis=0):
        """Creates rotation matrix about one of the coordinate axes."""
        radians = degrees / 180.0 * np.pi
        rot2x2 = np.array(
            [[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]]
        )
        r = np.eye(3)
        r[1:3, 1:3] = rot2x2
        r = np.roll(np.roll(r, axis, axis=0), axis, axis=1)
        p = np.eye(4)
        p[:3, :3] = r
        return p

    poses = poses @ rotation_about_axis(-render_rotate_yaxis, axis=1)
    poses = poses @ rotation_about_axis(render_rotate_xaxis, axis=0)

    # Add fourth row [0, 0, 0, 1] to make poses 4x4 matrices.
    poses = np.concatenate([poses, np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (poses.shape[0], 1, 1))], axis=1)

    # Transform back from PCA-aligned space to original space.
    poses = np.linalg.inv(pca_transform) @ poses

    # Transform back to Colmap camera coordinate system.
    poses = poses @ np.linalg.inv(colmap2opengl)

    return poses


def sample(
    t,
    w_logits,
    num_samples,
    deterministic_center=False,
    eps=np.finfo(np.float32).eps,
):
    """Piecewise-Constant PDF sampling from a step function.

    Args:
    rng: random number generator (or None for `linspace` sampling).
    t: [..., num_bins + 1], bin endpoint coordinates (must be sorted)
    w_logits: [..., num_bins], logits corresponding to bin weights
    num_samples: int, the number of samples.
    single_jitter: bool, if True, jitter every sample along each ray by the same
      amount in the inverse CDF. Otherwise, jitter each sample independently.
    deterministic_center: bool, if False, when `rng` is None return samples that
      linspace the entire PDF. If True, skip the front and back of the linspace
      so that the centers of each PDF interval are returned.
    eps: float, something like numerical epsilon.

    Returns:
    t_samples: jnp.ndarray(float32), [batch_size, num_samples].
    """
    if t.shape[-1] != w_logits.shape[-1] + 1:
        raise ValueError(
            f'Invalid shapes ({t.shape}, {w_logits.shape}) for a step function.'
        )

    # Draw uniform samples.
    # Match the behavior of jax.random.uniform() by spanning [0, 1-eps].
    if deterministic_center:
        pad = 1 / (2 * num_samples)
        u = np.linspace(pad, 1.0 - pad - eps, num_samples)
    else:
        u = np.linspace(0, 1.0 - eps, num_samples)
    u = np.broadcast_to(u, t.shape[:-1] + (num_samples,))

    def invert_cdf(u, t, w_logits):
        """Invert the CDF defined by (t, w) at the points specified by u in [0, 1)."""
        if t.shape[-1] != w_logits.shape[-1] + 1:
            raise ValueError(
                f'Invalid shapes ({t.shape}, {w_logits.shape}) for a step function.'
            )
        # Compute the PDF and CDF for each weight vector.
        w = torch.nn.functional.softmax(torch.from_numpy(w_logits), dim=-1).numpy()

        def integrate_weights(w):
            """Compute the cumulative sum of w, assuming all weight vectors sum to 1.

            The output's size on the last dimension is one greater than that of the input,
            because we're computing the integral corresponding to the endpoints of a step
            function, not the integral of the interior/bin values.

            Args:
              w: Tensor, which will be integrated along the last axis. This is assumed to
                sum to 1 along the last axis, and this function will (silently) break if
                that is not the case.

            Returns:
              cw0: Tensor, the integral of w, where cw0[..., 0] = 0 and cw0[..., -1] = 1
            """
            cw = np.minimum(1, np.cumsum(w[Ellipsis, :-1], axis=-1))
            shape = cw.shape[:-1] + (1,)
            # Ensure that the CDF starts with exactly 0 and ends with exactly 1.
            cw0 = np.concatenate([np.zeros(shape), cw, np.ones(shape)], axis=-1)
            return cw0

        cw = integrate_weights(w)

        def sorted_interp(x, xp, fp, eps=np.finfo(np.float32).eps ** 2):
            """A version of interp() where xp and fp must be sorted."""

            def sorted_lookup(x, xp, fps):
                """Lookup `x` into locations `xp` , return indices and each `[fp]` value."""
                if not isinstance(fps, tuple):
                    raise ValueError(f'Input `fps` must be a tuple, but is {type(fps)}.')
                # jnp.searchsorted() has slightly different conventions for boundary
                # handling than the rest of this codebase.
                idx = np.vectorize(
                    lambda a, v: np.searchsorted(a, v, side='right'),
                    signature='(n),(m)->(m)',
                )(xp, x)
                idx1 = np.minimum(idx, xp.shape[-1] - 1)
                idx0 = np.maximum(idx - 1, 0)
                vals = []
                for fp in fps:
                    fp0 = np.take_along_axis(fp, idx0, axis=-1)
                    fp1 = np.take_along_axis(fp, idx1, axis=-1)
                    vals.append((fp0, fp1))
                return (idx0, idx1), vals

            (xp0, xp1), (fp0, fp1) = sorted_lookup(x, xp, (xp, fp))[1]
            offset = np.clip((x - xp0) / np.maximum(eps, xp1 - xp0), 0, 1)
            ret = fp0 + offset * (fp1 - fp0)
            return ret

        # Interpolate into the inverse CDF.
        t_new = sorted_interp(u, cw, t)
        return t_new

    return invert_cdf(u, t, w_logits)
