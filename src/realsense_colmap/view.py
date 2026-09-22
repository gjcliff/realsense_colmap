"""Visualize a reconstructed .ply (point cloud or mesh) in Rerun."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import rerun as rr


def view_ply(path: Path, save: Path | None = None) -> None:
    mesh = o3d.io.read_triangle_mesh(str(path))
    is_mesh = len(mesh.triangles) > 0

    rr.init("realsense-colmap", spawn=save is None)
    if save is not None:
        rr.save(str(save))

    rr.log("world", rr.ViewCoordinates.RDF, static=True)

    if is_mesh:
        vertex_colors = (
            (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
            if mesh.has_vertex_colors()
            else None
        )
        vertex_normals = (
            np.asarray(mesh.vertex_normals) if mesh.has_vertex_normals() else None
        )
        rr.log(
            "world/mesh",
            rr.Mesh3D(
                vertex_positions=np.asarray(mesh.vertices),
                triangle_indices=np.asarray(mesh.triangles),
                vertex_normals=vertex_normals,
                vertex_colors=vertex_colors,
            ),
        )
        print(
            f"logged mesh: {len(mesh.vertices)} vertices, "
            f"{len(mesh.triangles)} triangles"
        )
    else:
        pcd = o3d.io.read_point_cloud(str(path))
        if len(pcd.points) == 0:
            raise RuntimeError(f"{path} has no points or triangles to display")
        colors = (
            (np.asarray(pcd.colors) * 255).astype(np.uint8)
            if pcd.has_colors()
            else None
        )
        rr.log(
            "world/points",
            rr.Points3D(positions=np.asarray(pcd.points), colors=colors),
        )
        print(f"logged point cloud: {len(pcd.points)} points")

    if save is not None:
        print(f"saved recording to {save} (open with `rerun {save}`)")
    else:
        print("viewer spawned; close its window or Ctrl+C here to exit")
