"""Grab depth + infrared + color frames from an Intel RealSense camera.

Depth, at the hardware level, is computed from -- and natively registered to
-- the left infrared imager, which (like the rest of the stereo module) is
global-shutter. The RGB sensor is a separate, rolling-shutter part. Since
rolling shutter breaks the single-rigid-pose-per-frame assumption SfM relies
on, COLMAP is fed the infrared images, never color: geometry (poses, depth
unprojection) comes entirely from the depth+IR pair, and color is only ever
reprojected onto that geometry afterwards, in dense.py, to paint points.
"""

from __future__ import annotations

import tarfile
import time
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

from .intrinsics import ColorCalibration, Intrinsics

# Number of frames to discard at start-up while auto-exposure/white-balance
# settle. Frames grabbed before this converge tend to be dark/washed out.
_WARMUP_FRAMES = 30

# The device's live-reported intrinsics have drifted from factory
# calibration, so both streams are calibrated externally (ROS2
# camera_calibration, checkerboard) instead -- see
# external/ros2-calib-docker. No live-SDK fallback: a missing/unreadable
# calibration archive is a hard error, not a reason to silently trust the
# SDK's (known-wrong) numbers instead.
_CALIB_DIR = (
    Path(__file__).resolve().parents[2]
    / "external" / "ros2-calib-docker" / "calibration_data"
)
_IR_CALIB_TAR = _CALIB_DIR / "infrared" / "calibrationdata.tar.gz"
_COLOR_CALIB_TAR = _CALIB_DIR / "color" / "calibrationdata.tar.gz"


class _CameraCalibration(NamedTuple):
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    coeffs: np.ndarray  # 5-element plumb_bob/brown_conrady (k1,k2,p1,p2,k3)


def _load_camera_calibration(tar_path: Path) -> _CameraCalibration:
    """Load a checkerboard-calibrated camera's intrinsics + distortion from
    the ost.yaml inside a ROS `camera_calibration` calibrationdata.tar.gz
    archive -- read directly out of the tar, never extracted to disk."""
    if not tar_path.is_file():
        raise FileNotFoundError(
            f"calibration archive not found at {tar_path}. capture_sequence "
            "requires checkerboard calibrations for both the infrared and "
            "color streams (see external/ros2-calib-docker/calibration_data) "
            "and will not fall back to the camera's live-reported intrinsics."
        )
    with tarfile.open(tar_path) as tf:
        member = tf.extractfile("ost.yaml")
        if member is None:
            raise FileNotFoundError(f"ost.yaml not found inside {tar_path}")
        doc = yaml.safe_load(member.read())
    m = doc["camera_matrix"]["data"]
    coeffs = np.array(doc["distortion_coefficients"]["data"], dtype=np.float64)
    return _CameraCalibration(
        width=doc["image_width"],
        height=doc["image_height"],
        fx=m[0], fy=m[4], cx=m[2], cy=m[5],
        coeffs=coeffs,
    )


def _build_undistort_maps(
    fx: float, fy: float, cx: float, cy: float,
    coeffs: np.ndarray, width: int, height: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (map1, map2) for cv2.remap, or (None, None) if already
    rectified (zero distortion)."""
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if np.allclose(coeffs, 0.0):
        return None, None
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return cv2.initUndistortRectifyMap(k, coeffs, None, k, (width, height), cv2.CV_32FC1)


def _set_option_checked(sensor: rs.sensor, option: rs.option, value: float) -> None:
    r = sensor.get_option_range(option)
    if not (r.min <= value <= r.max):
        raise ValueError(
            f"{option.name}={value} out of range [{r.min}, {r.max}] "
            f"(step {r.step}) for this device"
        )
    sensor.set_option(option, value)


def _reset_option(sensor: rs.sensor, option: rs.option) -> None:
    """RealSense sensor options live on the device itself and persist across
    separate process runs (unlike a fresh rs.pipeline() object, the physical
    sensor doesn't reset). Without this, a plain `capture` with no exposure
    flags could silently inherit whatever a previous manual run last set."""
    sensor.set_option(option, sensor.get_option_range(option).default)


def _configure_sensors(
    profile: rs.pipeline_profile,
    fps: int,
    color_exposure: float | None,
    color_gain: float | None,
    auto_exposure_priority: bool,
    laser_power: float | None,
    emitter_enabled: bool = True,
) -> None:
    """Apply manual exposure/gain and/or depth-projector settings, mainly
    useful in low light. Color exposure/gain here only affect how the RGB
    (paint-only) image looks -- they no longer have any bearing on pose
    estimation or reconstruction geometry, since color never reaches COLMAP.
    laser_power boosts the IR dot pattern both depth *and* the infrared
    images COLMAP now uses rely on, which matters more the less ambient
    light there is.

    Every option here is explicitly set or explicitly reset to its device
    default -- never left alone -- since these settings are sticky on the
    physical sensor across runs (see _reset_option)."""
    device = profile.get_device()
    color_sensor = device.first_color_sensor()

    # We don't expose white-balance controls, but it's a sensor option like
    # any other -- sticky on the device, and settable by anything that's
    # touched this camera (including ad-hoc debugging scripts). Always force
    # it back to auto so a stray manual value never silently persists.
    _reset_option(color_sensor, rs.option.enable_auto_white_balance)
    _reset_option(color_sensor, rs.option.white_balance)

    if color_exposure is not None or color_gain is not None:
        color_sensor.set_option(rs.option.enable_auto_exposure, 0)
        if color_exposure is not None:
            _set_option_checked(color_sensor, rs.option.exposure, color_exposure)
            exposure_seconds = color_exposure * 1e-4
            frame_period = 1.0 / fps
            if exposure_seconds > frame_period:
                print(
                    f"warning: --color-exposure {color_exposure} is "
                    f"{exposure_seconds * 1000:.1f}ms, longer than the "
                    f"{frame_period * 1000:.1f}ms frame period at --fps {fps}. "
                    "The sensor can't expose longer than one frame interval, "
                    "so this will get silently capped to roughly the frame "
                    "period -- it'll report back as set, but won't actually "
                    "be that bright. Lower --fps to make room for it "
                    f"(need --fps <= {1.0 / exposure_seconds:.1f})."
                )
        else:
            _reset_option(color_sensor, rs.option.exposure)
        if color_gain is not None:
            _set_option_checked(color_sensor, rs.option.gain, color_gain)
        else:
            _reset_option(color_sensor, rs.option.gain)
        print(
            f"manual color exposure={color_sensor.get_option(rs.option.exposure)} "
            f"(x100us), gain={color_sensor.get_option(rs.option.gain)}"
        )
    else:
        _reset_option(color_sensor, rs.option.enable_auto_exposure)

    color_sensor.set_option(
        rs.option.auto_exposure_priority, 1 if auto_exposure_priority else 0
    )
    if auto_exposure_priority:
        print("auto-exposure priority enabled (frame rate may drop in low light)")

    # Order matters here: on this firmware, writing *any* laser_power value
    # -- even just resetting it to default -- silently flips emitter_enabled
    # back to 1, regardless of what it was set to before. So laser_power has
    # to be settled first, and emitter_enabled written last so it's the
    # actual final word on whether the emitter fires.
    depth_sensor = device.first_depth_sensor()
    if laser_power is not None:
        _set_option_checked(depth_sensor, rs.option.laser_power, laser_power)
    else:
        _reset_option(depth_sensor, rs.option.laser_power)
    depth_sensor.set_option(rs.option.emitter_enabled, 1 if emitter_enabled else 0)
    print(
        f"laser power={depth_sensor.get_option(rs.option.laser_power)}, "
        f"emitter_enabled={depth_sensor.get_option(rs.option.emitter_enabled)}"
    )


def capture_sequence(
    output_dir: Path,
    fps: int = 30,
    num_frames: int | None = None,
    seconds: float | None = None,
    every_n: int = 1,
    preview: bool = True,
    color_exposure: float | None = None,
    color_gain: float | None = None,
    auto_exposure_priority: bool = False,
    laser_power: float | None = None,
    emitter_enabled: bool = True,
) -> Intrinsics:
    """Stream depth+infrared+color from the first connected RealSense device
    and save every `every_n`-th frame to `output_dir`/{ir,depth,rgb}.

    Depth and infrared are natively co-registered by the hardware (verified:
    identical intrinsics, identity extrinsic) so need no alignment step.
    Color is captured in its own native frame and never warped -- it's
    reprojected onto geometry per-point later, using the calibrated
    depth-to-color extrinsic saved in color_calibration.json.

    Both streams' resolution comes from their respective checkerboard
    calibrations (see _load_camera_calibration), not a user-configurable
    setting -- a calibration is only valid at the resolution it was taken
    at.

    Returns the Intrinsics for the depth/infrared camera (what everything
    downstream treats as "the" camera), also written to
    `output_dir`/intrinsics.json.
    """
    ir_calib = _load_camera_calibration(_IR_CALIB_TAR)
    color_calib = _load_camera_calibration(_COLOR_CALIB_TAR)

    ir_dir = output_dir / "ir"
    depth_dir = output_dir / "depth"
    rgb_dir = output_dir / "rgb"
    ir_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, ir_calib.width, ir_calib.height, rs.format.z16, fps)
    config.enable_stream(rs.stream.infrared, 1, ir_calib.width, ir_calib.height, rs.format.y8, fps)
    config.enable_stream(rs.stream.color, color_calib.width, color_calib.height, rs.format.bgr8, fps)

    profile = pipeline.start(config)

    _configure_sensors(
        profile,
        fps,
        color_exposure,
        color_gain,
        auto_exposure_priority,
        laser_power,
        emitter_enabled,
    )

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()

    intrinsics = Intrinsics(
        width=ir_calib.width,
        height=ir_calib.height,
        fx=ir_calib.fx,
        fy=ir_calib.fy,
        cx=ir_calib.cx,
        cy=ir_calib.cy,
        depth_scale=depth_scale,
    )
    intrinsics.save(output_dir / "intrinsics.json")

    # ROS2's monocular checkerboard calibration gives each camera's own
    # intrinsics/distortion, but not the baseline *between* the IR and color
    # sensors -- that still comes from the live device's factory extrinsic.
    extrinsics = depth_profile.get_extrinsics_to(color_profile)
    color_calibration = ColorCalibration(
        width=color_calib.width,
        height=color_calib.height,
        fx=color_calib.fx,
        fy=color_calib.fy,
        cx=color_calib.cx,
        cy=color_calib.cy,
        rotation=list(extrinsics.rotation),
        translation=list(extrinsics.translation),
    )
    color_calibration.save(output_dir / "color_calibration.json")

    # IR and depth share a frame, so one undistort map pair serves both.
    ir_map1, ir_map2 = _build_undistort_maps(
        ir_calib.fx, ir_calib.fy, ir_calib.cx, ir_calib.cy,
        ir_calib.coeffs, ir_calib.width, ir_calib.height,
    )
    color_map1, color_map2 = _build_undistort_maps(
        color_calib.fx, color_calib.fy, color_calib.cx, color_calib.cy,
        color_calib.coeffs, color_calib.width, color_calib.height,
    )

    saved = 0
    grabbed = 0
    try:
        for _ in range(_WARMUP_FRAMES):
            pipeline.wait_for_frames()

        print(f"capturing to {output_dir} (Ctrl+C to stop)...")
        start_time = time.monotonic()
        while True:
            if num_frames is not None and saved >= num_frames:
                break
            if seconds is not None and time.monotonic() - start_time >= seconds:
                break

            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            ir_frame = frames.get_infrared_frame(1)
            color_frame = frames.get_color_frame()
            if not depth_frame or not ir_frame or not color_frame:
                continue

            grabbed += 1
            if (grabbed - 1) % every_n != 0:
                continue

            ir_image = np.asanyarray(ir_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())  # uint16, raw units
            rgb_image = np.asanyarray(color_frame.get_data())

            if ir_map1 is not None:
                ir_image = cv2.remap(ir_image, ir_map1, ir_map2, cv2.INTER_LINEAR)
                depth_image = cv2.remap(
                    depth_image, ir_map1, ir_map2, cv2.INTER_NEAREST
                )
            if color_map1 is not None:
                rgb_image = cv2.remap(
                    rgb_image, color_map1, color_map2, cv2.INTER_LINEAR
                )

            stem = f"{saved:06d}"
            cv2.imwrite(str(ir_dir / f"{stem}.png"), ir_image)
            cv2.imwrite(str(depth_dir / f"{stem}.png"), depth_image)
            cv2.imwrite(str(rgb_dir / f"{stem}.png"), rgb_image)
            saved += 1

            if preview:
                try:
                    depth_vis = cv2.convertScaleAbs(depth_image, alpha=0.03)
                    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                    preview_ir = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
                    cv2.putText(
                        preview_ir,
                        f"saved {saved} ({stem}.png)  [q to stop]",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("realsense-colmap: infrared (fed to COLMAP)", preview_ir)
                    cv2.imshow("realsense-colmap: depth", depth_vis)
                    cv2.imshow("realsense-colmap: color (paint only)", rgb_image)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error as e:
                    print(
                        f"\nwarning: preview window failed ({e}); continuing "
                        "capture without it. Pass --no-preview to silence this "
                        "(e.g. over SSH without X forwarding)."
                    )
                    preview = False
                    cv2.destroyAllWindows()

            if saved % 10 == 0:
                print(f"  saved {saved} frames", end="\r")

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        if preview:
            cv2.destroyAllWindows()

    print(f"\nsaved {saved} frames to {output_dir}")
    return intrinsics
