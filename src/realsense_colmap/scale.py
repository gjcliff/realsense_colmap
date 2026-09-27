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
            "the registered infrared images."
        )

    ratios = np.array(ratios)
    raw_median = np.median(ratios)

    # The median alone is already robust to outliers for the center estimate,
    # but degenerately-triangulated points (near-zero sfm_depth from weak
    # parallax) can still produce ratios in the thousands, which dominate the
    # std and make the reported uncertainty meaningless even when the median
    # itself is fine. Trim by a robust (MAD-based) outlier threshold before
    # reporting/using the spread, so it reflects the quality of the data
    # that's actually left, not noise from points we already know are junk.
    mad = np.median(np.abs(ratios - raw_median)) + 1e-12
    modified_z = 0.6745 * (ratios - raw_median) / mad
    keep = np.abs(modified_z) < 5.0
    trimmed = ratios[keep]

    scale = float(np.median(trimmed))
    spread = float(np.std(trimmed) / scale) if scale else float("inf")
    n_dropped = len(ratios) - len(trimmed)
    print(
        f"estimated metric scale = {scale:.4f} "
        f"(from {len(trimmed)} point observations after dropping "
        f"{n_dropped} ({n_dropped / len(ratios):.1%}) as robust outliers; "
        f"relative std {spread:.2%})"
    )
    if spread > 0.5:
        print(
            "warning: scale estimate is noisy (relative std > 50%) even "
            "after outlier trimming. The dense reconstruction's absolute "
            "scale may be unreliable."
        )
    return scale
