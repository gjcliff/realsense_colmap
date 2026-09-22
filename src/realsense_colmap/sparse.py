"""Structure-from-motion via pycolmap: recover a camera trajectory (poses,
up to an unknown scale) from the captured color images."""

from __future__ import annotations

from pathlib import Path

import pycolmap

from .intrinsics import Intrinsics


def run_sparse_reconstruction(
    image_dir: Path,
    database_path: Path,
    sparse_dir: Path,
    intrinsics: Intrinsics,
    matcher: str = "sequential",
) -> pycolmap.Reconstruction:
    """Run feature extraction, matching and incremental SfM. Returns the
    largest resulting reconstruction (by number of registered images)."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    # SQLite's WAL-mode sidecar files (-shm/-wal/-journal) can be left behind
    # stale if a previous run was killed mid-write; deleting only the main
    # .db file but leaving those around makes SQLite see inconsistent state
    # and fail to open the fresh database with an I/O error.
    for suffix in ("", "-shm", "-wal", "-journal"):
        sidecar = database_path.with_name(database_path.name + suffix)
        if sidecar.exists():
            sidecar.unlink()
    sparse_dir.mkdir(parents=True, exist_ok=True)

    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = "PINHOLE"
    reader_options.camera_params = intrinsics.colmap_params()

    print("extracting features...")
    pycolmap.extract_features(
        database_path=database_path,
        image_path=image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_options,
    )

    print(f"matching features ({matcher})...")
    if matcher == "sequential":
        pycolmap.match_sequential(database_path=database_path)
    elif matcher == "exhaustive":
        pycolmap.match_exhaustive(database_path=database_path)
    else:
        raise ValueError(f"unknown matcher: {matcher!r}")

    print("running incremental mapping...")
    options = pycolmap.IncrementalPipelineOptions()
    # Trust the RealSense factory calibration rather than letting bundle
    # adjustment refine intrinsics from a possibly small/narrow-baseline scan.
    options.ba_refine_focal_length = False
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = False

    reconstructions = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_dir,
        output_path=sparse_dir,
        options=options,
    )

    if not reconstructions:
        raise RuntimeError(
            "SfM failed to register any images. Try capturing more frames "
            "with more overlap/texture, or use --matcher exhaustive."
        )

    best = max(reconstructions.values(), key=lambda r: r.num_reg_images())
    print(
        f"registered {best.num_reg_images()} / "
        f"{len(list(image_dir.glob('*.png')))} images, "
        f"{best.num_points3D()} sparse points"
    )

    best_dir = sparse_dir / "0"
    best_dir.mkdir(parents=True, exist_ok=True)
    best.write(best_dir)

    return best
