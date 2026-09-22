"""COLMAP's monocular SfM recovers camera poses only up to an unknown,
arbitrary scale. We recover the real-world scale by comparing the depth SfM
*predicts* for its own triangulated points against the depth the RealSense
actually measured at those pixels, and taking the robust (median) ratio."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pycolmap

from .intrinsics import Intrinsics


def estimate_scale(
    reconstruction: pycolmap.Reconstruction,
    depth_dir: Path,
    intrinsics: Intrinsics,
    min_depth: float = 0.1,
    max_depth: float = 8.0,
) -> float:
    ratios: list[float] = []

    for image in reconstruction.images.values():
        if not image.has_pose:
            continue

        depth_path = depth_dir / Path(image.name).with_suffix(".png").name
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            continue
        depth_m = depth_raw.astype(np.float32) * intrinsics.depth_scale

        cam_from_world = image.cam_from_world()
        R = cam_from_world.rotation.matrix()
        t = cam_from_world.translation

        for point2D in image.points2D:
            if not point2D.has_point3D():
                continue
            u, v = point2D.xy
            ui, vi = int(round(u)), int(round(v))
            if not (0 <= ui < intrinsics.width and 0 <= vi < intrinsics.height):
                continue

            measured_depth = float(depth_m[vi, ui])
            if not (min_depth <= measured_depth <= max_depth):
                continue

            point3D = reconstruction.points3D[point2D.point3D_id]
            p_cam = R @ point3D.xyz + t
            sfm_depth = float(p_cam[2])
            if sfm_depth <= 1e-6:
                continue

            ratios.append(measured_depth / sfm_depth)

    if not ratios:
        raise RuntimeError(
            "couldn't estimate metric scale: no sparse point had a valid "
            "matching depth reading. Check that depth/ frames line up with "
            "the registered color images."
        )

    ratios = np.array(ratios)
    scale = float(np.median(ratios))
    spread = float(np.std(ratios) / scale) if scale else float("inf")
    print(
        f"estimated metric scale = {scale:.4f} "
        f"(from {len(ratios)} point observations, relative std {spread:.2%})"
    )
    if spread > 0.5:
        print(
            "warning: scale estimate is noisy (relative std > 50%). The "
            "dense reconstruction's absolute scale may be unreliable."
        )
    return scale
