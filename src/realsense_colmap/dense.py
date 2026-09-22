"""Fuse the RealSense's real (metric) depth maps at the camera poses
recovered by SfM into a single dense, colored point cloud.

We deliberately don't use COLMAP's own dense multi-view stereo here: it would
re-derive depth photometrically from the RGB images alone, which is both more
expensive and less accurate than the depth we already measured with the
sensor. COLMAP is used only for what it's good at -- recovering the camera
trajectory (from infrared images; see sparse.py/capture.py for why).

Color never enters COLMAP or the pose/geometry computation here either: it's
reprojected onto the depth-derived geometry per-point, using the calibrated
depth-to-color extrinsic, purely to paint points.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pycolmap
from tqdm import tqdm

from .intrinsics import ColorCalibration, Intrinsics

# Neighboring depth pixels whose z jumps by more than this are a depth edge /
# "flying pixel" artifact, not real surface -- both the point and any normal
# computed across it are unreliable, so such pixels are dropped.
_MAX_DEPTH_DISCONTINUITY = 0.03  # meters


def fuse_dense_point_cloud(
    reconstruction: pycolmap.Reconstruction,
    rgb_dir: Path,
    depth_dir: Path,
    intrinsics: Intrinsics,
    color_calibration: ColorCalibration,
    scale: float,
    min_depth: float = 0.1,
    max_depth: float = 8.0,
    voxel_size: float = 0.01,
    stride: int = 2,
) -> o3d.geometry.PointCloud:
    """Back-project every valid depth pixel of every registered frame into a
    shared world frame and merge into one point cloud, painted by reprojecting
    each point into the (native, unwarped) RGB image.

    `stride` subsamples pixels (e.g. 2 keeps every other pixel in x and y)
    to keep the raw cloud a manageable size before voxel downsampling.
    """
    fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy
    h, w = intrinsics.height, intrinsics.width

    us, vs = np.meshgrid(np.arange(0, w, stride), np.arange(0, h, stride))
    us = us.astype(np.float32)
    vs = vs.astype(np.float32)
    x_over_z = (us - cx) / fx
    y_over_z = (vs - cy) / fy

    # depth-frame -> color-frame extrinsic, for reprojecting points into the
    # native (unwarped) RGB image to sample color. p_color = R @ p_depth + t.
    color_R = color_calibration.rotation_matrix()
    color_t = np.array(color_calibration.translation, dtype=np.float64)
    cfx, cfy = color_calibration.fx, color_calibration.fy
    ccx, ccy = color_calibration.cx, color_calibration.cy
    color_w, color_h = color_calibration.width, color_calibration.height

    all_points = []
    all_colors = []
    all_normals = []

    registered = [img for img in reconstruction.images.values() if img.has_pose]
    for image in tqdm(registered, desc="fusing depth"):
        depth_path = depth_dir / image.name
        color_path = rgb_dir / image.name
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        color_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        if depth_raw is None or color_bgr is None:
            continue

        depth_m = depth_raw[::stride, ::stride].astype(np.float32) * intrinsics.depth_scale

        z = depth_m
        x = x_over_z * z
        y = y_over_z * z
        grid = np.stack([x, y, z], axis=-1)  # (H', W', 3), depth/IR camera frame

        # Reproject into the color camera to look up each pixel's paint.
        p_color = grid @ color_R.T + color_t
        z_color = p_color[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u_color = cfx * p_color[..., 0] / z_color + ccx
            v_color = cfy * p_color[..., 1] / z_color + ccy
        u_idx = np.round(u_color).astype(np.int32)
        v_idx = np.round(v_color).astype(np.int32)
        color_valid = (
            (z_color > 1e-6)
            & (u_idx >= 0)
            & (u_idx < color_w)
            & (v_idx >= 0)
            & (v_idx < color_h)
        )
        # Clip so out-of-bounds indices don't fault the gather below; those
        # pixels are excluded from the final mask regardless.
        u_safe = np.clip(u_idx, 0, color_w - 1)
        v_safe = np.clip(v_idx, 0, color_h - 1)
        color = color_bgr[v_safe, u_safe].astype(np.float32) / 255.0

        # A depth image is already an organized point cloud, so surface normals
        # come straight from finite differences between grid neighbors -- no
        # neighbor search needed. This is what lets us skip Open3D's generic
        # KD-tree + global-consistency normal estimation later, which is the
        # step that grinds to a halt on multi-million-point clouds.
        dx = np.zeros_like(grid)
        dy = np.zeros_like(grid)
        dx[:, :-1] = grid[:, 1:] - grid[:, :-1]
        dy[:-1, :] = grid[1:, :] - grid[:-1, :]
        normals = np.cross(dx, dy)
        lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = np.divide(
            normals, lengths, out=np.zeros_like(normals), where=lengths > 1e-9
        )
        # The camera sits at the origin in camera space, so a normal that faces
        # the camera satisfies dot(normal, point) < 0; flip the ones that don't.
        facing_away = np.einsum("...i,...i->...", normals, grid) > 0
        normals[facing_away] *= -1

        valid = (
            (z >= min_depth)
            & (z <= max_depth)
            & (lengths[..., 0] > 1e-9)
            & (np.abs(dx[..., 2]) < _MAX_DEPTH_DISCONTINUITY)
            & (np.abs(dy[..., 2]) < _MAX_DEPTH_DISCONTINUITY)
            & color_valid
        )
        if not np.any(valid):
            continue

        cam_from_world = image.cam_from_world()
        R = cam_from_world.rotation.matrix()
        center = image.projection_center() * scale  # metric, world frame

        points_world = grid[valid] @ R + center  # R^T @ p == p @ R
        normals_world = normals[valid] @ R

        bgr = color[valid]
        rgb = bgr[:, ::-1]

        all_points.append(points_world)
        all_colors.append(rgb)
        all_normals.append(normals_world)

    if not all_points:
        raise RuntimeError("no valid depth pixels found in any registered frame")

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    normals = np.concatenate(all_normals, axis=0)

    # A handful of badly-triangulated SfM points (e.g. from a near-degenerate
    # solve) can put a few camera centers at wild coordinates once scaled,
    # producing points thousands of units away from the real scene. That's
    # not just wrong -- Open3D's voxel downsampling hard-crashes ("voxel_size
    # is too small") once the point cloud's bounding box gets absurd relative
    # to voxel_size, since it buckets points into a fixed-range integer grid.
    # Drop such outliers before it ever gets there.
    center = np.median(points, axis=0)
    dist_from_center = np.linalg.norm(points - center, axis=1)
    typical_dist = np.median(dist_from_center) + 1e-9
    keep = dist_from_center < max(typical_dist * 50, 20.0)
    if not np.all(keep):
        print(
            f"warning: dropping {np.sum(~keep)} / {len(points)} points as "
            "extreme outliers (likely from a couple of unreliable camera "
            "poses) before voxel downsampling"
        )
        points, colors, normals = points[keep], colors[keep], normals[keep]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
        pcd.normalize_normals()  # voxel-averaging doesn't preserve unit length

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    return pcd


def mesh_from_point_cloud(
    pcd: o3d.geometry.PointCloud, depth: int = 9
) -> o3d.geometry.TriangleMesh:
    """Poisson surface reconstruction. Needs oriented point normals.

    `fuse_dense_point_cloud` already attaches correctly-oriented normals (from
    each frame's depth-grid structure), so normally there's nothing to do
    here. The fallback below only fires for a point cloud that didn't come
    from that function (e.g. one loaded from an old dense.ply) -- it's a last
    resort, since Open3D's KD-tree + global-consistency orientation is slow
    enough to look hung on point clouds much past a few hundred thousand
    points.
    """
    pcd = o3d.geometry.PointCloud(pcd)
    if not pcd.has_normals():
        print(
            f"no precomputed normals on this cloud ({len(pcd.points)} points); "
            "falling back to Open3D's normal estimation, which can be slow "
            "on large clouds..."
        )
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)

    print(f"running Poisson surface reconstruction (depth={depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )
    # Trim low-density (extrapolated/unsupported) surface.
    densities = np.asarray(densities)
    keep = densities >= np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(~keep)
    return mesh
