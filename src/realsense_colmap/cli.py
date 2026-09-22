from __future__ import annotations

import argparse
from pathlib import Path


def _cmd_capture(args: argparse.Namespace) -> None:
    from .capture import capture_sequence

    capture_sequence(
        output_dir=Path(args.output),
        width=args.width,
        height=args.height,
        fps=args.fps,
        num_frames=args.num_frames,
        seconds=args.seconds,
        every_n=args.every_n,
        preview=args.preview,
        color_exposure=args.color_exposure,
        color_gain=args.color_gain,
        auto_exposure_priority=args.auto_exposure_priority,
        laser_power=args.laser_power,
        emitter_enabled=not args.no_emitter,
    )


def _cmd_reconstruct(args: argparse.Namespace) -> None:
    from .dense import fuse_dense_point_cloud, mesh_from_point_cloud
    from .intrinsics import ColorCalibration, Intrinsics
    from .scale import estimate_scale
    from .sparse import run_sparse_reconstruction

    import open3d as o3d

    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir / "reconstruction"
    output_dir.mkdir(parents=True, exist_ok=True)

    ir_dir = input_dir / "ir"
    depth_dir = input_dir / "depth"
    rgb_dir = input_dir / "rgb"
    intrinsics = Intrinsics.load(input_dir / "intrinsics.json")
    color_calibration = ColorCalibration.load(input_dir / "color_calibration.json")

    reconstruction = run_sparse_reconstruction(
        image_dir=ir_dir,
        database_path=output_dir / "database.db",
        sparse_dir=output_dir / "sparse",
        intrinsics=intrinsics,
        matcher=args.matcher,
    )

    scale = estimate_scale(
        reconstruction,
        depth_dir=depth_dir,
        intrinsics=intrinsics,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )

    pcd = fuse_dense_point_cloud(
        reconstruction,
        rgb_dir=rgb_dir,
        depth_dir=depth_dir,
        intrinsics=intrinsics,
        color_calibration=color_calibration,
        scale=scale,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        voxel_size=args.voxel_size,
        stride=args.stride,
    )

    dense_path = output_dir / "dense.ply"
    o3d.io.write_point_cloud(str(dense_path), pcd)
    print(f"wrote {len(pcd.points)} points to {dense_path}")

    if args.mesh:
        mesh = mesh_from_point_cloud(pcd)
        mesh_path = output_dir / "mesh.ply"
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)
        print(f"wrote mesh with {len(mesh.vertices)} vertices to {mesh_path}")


def _cmd_view(args: argparse.Namespace) -> None:
    from .view import view_ply

    view_ply(
        Path(args.input),
        save=Path(args.save) if args.save else None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="realsense-colmap",
        description="Capture RGB-D from a RealSense camera and turn it into "
        "a dense 3D reconstruction, using COLMAP for camera pose estimation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    cap = subparsers.add_parser("capture", help="record color+depth frames")
    cap.add_argument("-o", "--output", required=True, help="output directory")
    cap.add_argument("--width", type=int, default=1280)
    cap.add_argument("--height", type=int, default=720)
    cap.add_argument("--fps", type=int, default=15)
    cap.add_argument(
        "--num-frames", type=int, default=None, help="stop after this many saved frames"
    )
    cap.add_argument(
        "--seconds", type=float, default=None, help="stop after this many seconds"
    )
    cap.add_argument(
        "--every-n",
        type=int,
        default=1,
        help="only save every Nth grabbed frame (reduces redundancy)",
    )
    cap.add_argument(
        "--no-preview",
        dest="preview",
        action="store_false",
        help="don't show the live color/depth preview window (default: shown; "
        "needed for headless/SSH-without-X capture)",
    )
    cap.add_argument(
        "--color-exposure",
        type=float,
        default=None,
        help="manual color exposure, in 100us units (range 1-10000, default "
        "~166 i.e. ~16.6ms). Disables auto-exposure and holds it fixed for "
        "the whole capture (which also helps SfM feature matching stay "
        "consistent frame to frame). Try 800-2000 in a dark room.",
    )
    cap.add_argument(
        "--color-gain",
        type=float,
        default=None,
        help="manual color gain (range 0-128, default 64); pairs with "
        "--color-exposure. Higher = brighter but noisier.",
    )
    cap.add_argument(
        "--auto-exposure-priority",
        action="store_true",
        help="let auto-exposure drop frame rate for a brighter image instead "
        "of capping exposure to hold fps -- simplest low-light fix if you "
        "don't want to set exposure manually. Ignored if --color-exposure is set.",
    )
    cap.add_argument(
        "--laser-power",
        type=float,
        default=None,
        help="depth IR projector power (range 0-360, step 30, default 150). "
        "Raise this in low light / low-texture scenes to improve depth "
        "quality -- the depth sensor relies on its own projected IR pattern "
        "more when there's less ambient light for it to work with.",
    )
    cap.add_argument(
        "--no-emitter",
        action="store_true",
        help="disable the IR dot projector entirely. Depth quality will "
        "degrade in low-texture/dark areas (that's what the emitter is for), "
        "but it also removes the projector's repetitive dot pattern from the "
        "infrared images COLMAP matches on, which can otherwise get "
        "misclassified as a near-static 'watermark' and produce spurious "
        "matches. Mainly useful for testing which effect dominates.",
    )
    cap.set_defaults(func=_cmd_capture)

    rec = subparsers.add_parser(
        "reconstruct", help="run SfM (COLMAP) + depth fusion to build a dense cloud"
    )
    rec.add_argument("-i", "--input", required=True, help="capture directory")
    rec.add_argument(
        "-o",
        "--output",
        default=None,
        help="output directory (default: <input>/reconstruction)",
    )
    rec.add_argument(
        "--matcher",
        choices=["sequential", "exhaustive"],
        default="sequential",
        help="COLMAP feature matching strategy (sequential is appropriate "
        "for a continuous video-like capture; exhaustive is more robust "
        "for a sparse/unordered set of images but much slower)",
    )
    rec.add_argument("--min-depth", type=float, default=0.1, help="meters")
    rec.add_argument("--max-depth", type=float, default=8.0, help="meters")
    rec.add_argument(
        "--voxel-size",
        type=float,
        default=0.01,
        help="voxel size (meters) for downsampling the fused cloud; 0 disables",
    )
    rec.add_argument(
        "--stride",
        type=int,
        default=2,
        help="subsample depth pixels by this factor before fusing",
    )
    rec.add_argument(
        "--mesh", action="store_true", help="also run Poisson surface reconstruction"
    )
    rec.set_defaults(func=_cmd_reconstruct)

    view = subparsers.add_parser(
        "view", help="visualize a .ply point cloud or mesh in Rerun"
    )
    view.add_argument("input", help="path to dense.ply or mesh.ply")
    view.add_argument(
        "--save",
        default=None,
        help="save an .rrd recording to this path instead of spawning the "
        "interactive viewer (view later with `rerun <path>`)",
    )
    view.set_defaults(func=_cmd_view)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
