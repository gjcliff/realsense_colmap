"""Grab aligned color + depth frames from an Intel RealSense camera."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from .intrinsics import Intrinsics

# Number of frames to discard at start-up while auto-exposure/white-balance
# settle. Frames grabbed before this converge tend to be dark/washed out.
_WARMUP_FRAMES = 30


def _build_undistort_maps(color_profile) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (map1, map2) for cv2.remap, or (None, None) if the color stream
    is already rectified (no distortion, or a model cv2 can't represent)."""
    intr = color_profile.get_intrinsics()
    coeffs = np.array(intr.coeffs, dtype=np.float64)

    if np.allclose(coeffs, 0.0):
        return None, None

    if intr.model != rs.distortion.brown_conrady:
        print(
            f"warning: color stream distortion model is {intr.model}, which "
            "OpenCV's undistort can't represent; capturing without "
            "undistortion. Points near the image edges will be slightly off."
        )
        return None, None

    k = np.array(
        [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]]
    )
    size = (intr.width, intr.height)
    map1, map2 = cv2.initUndistortRectifyMap(
        k, coeffs, None, k, size, cv2.CV_32FC1
    )
    return map1, map2


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
    profile: "rs.pipeline_profile",
    fps: int,
    color_exposure: float | None,
    color_gain: float | None,
    auto_exposure_priority: bool,
    laser_power: float | None,
) -> None:
    """Apply manual exposure/gain and/or depth-projector settings, mainly
    useful in low light: manual color exposure lets you push past what
    auto-exposure would choose (and holds it steady across the whole scan,
    which also helps SfM feature matching); auto-exposure-priority is a
    lighter touch that just lets the frame rate drop instead of capping
    exposure; laser_power boosts the IR dot pattern depth stereo matches on,
    which matters more the less ambient light there is.

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

    depth_sensor = device.first_depth_sensor()
    if laser_power is not None:
        _set_option_checked(depth_sensor, rs.option.laser_power, laser_power)
    else:
        _reset_option(depth_sensor, rs.option.laser_power)
    print(f"laser power={depth_sensor.get_option(rs.option.laser_power)}")


def capture_sequence(
    output_dir: Path,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    num_frames: int | None = None,
    seconds: float | None = None,
    every_n: int = 1,
    preview: bool = True,
    color_exposure: float | None = None,
    color_gain: float | None = None,
    auto_exposure_priority: bool = False,
    laser_power: float | None = None,
) -> Intrinsics:
    """Stream color+depth from the first connected RealSense device and save
    every `every_n`-th frame pair to `output_dir`/color and `output_dir`/depth.

    Returns the Intrinsics used for the color stream (which the aligned depth
    frames share), also written to `output_dir`/intrinsics.json.
    """
    color_dir = output_dir / "color"
    depth_dir = output_dir / "depth"
    color_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    _configure_sensors(
        profile, fps, color_exposure, color_gain, auto_exposure_priority, laser_power
    )

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    intrinsics = Intrinsics(
        width=intr.width,
        height=intr.height,
        fx=intr.fx,
        fy=intr.fy,
        cx=intr.ppx,
        cy=intr.ppy,
        depth_scale=depth_scale,
    )
    intrinsics.save(output_dir / "intrinsics.json")

    map1, map2 = _build_undistort_maps(color_profile)

    saved = 0
    grabbed = 0
    start_time = None
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
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            grabbed += 1
            if (grabbed - 1) % every_n != 0:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())  # uint16, raw units

            if map1 is not None:
                color_image = cv2.remap(color_image, map1, map2, cv2.INTER_LINEAR)
                depth_image = cv2.remap(depth_image, map1, map2, cv2.INTER_NEAREST)

            stem = f"{saved:06d}"
            cv2.imwrite(str(color_dir / f"{stem}.png"), color_image)
            cv2.imwrite(str(depth_dir / f"{stem}.png"), depth_image)
            saved += 1

            if preview:
                try:
                    depth_vis = cv2.convertScaleAbs(depth_image, alpha=0.03)
                    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                    preview_color = color_image.copy()
                    cv2.putText(
                        preview_color,
                        f"saved {saved} ({stem}.png)  [q to stop]",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("realsense-colmap: color (saved frames)", preview_color)
                    cv2.imshow("realsense-colmap: depth (saved frames)", depth_vis)
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

    print(f"\nsaved {saved} frame pairs to {output_dir}")
    return intrinsics
