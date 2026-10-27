"""Offline face framing with shot-aware motion and temporary two-person layouts."""

import math
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from statistics import median

import av
import cv2
import numpy as np

MODEL = Path(__file__).parent / "assets" / "face_detection_yunet_2023mar.onnx"
SAMPLE_SECONDS = 0.125


@dataclass
class Observation:
    time: float
    center: tuple[float, float] | None
    cut: bool = False
    pair: tuple[tuple[float, float], tuple[float, float]] | None = None


def scene_changed(previous, current):
    """Compare spatial content, including cuts between similarly lit interview shots."""
    if previous is None:
        return False
    difference = np.abs(current.astype(float) - previous.astype(float)).mean(axis=2)
    return bool(difference.mean() > 12 and (difference > 25).mean() > 0.15)


def choose_pair(faces, width):
    """Two substantial, separated faces; stable left/right ordering becomes top/bottom."""
    if len(faces) < 2:
        return None
    first, second = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:2]
    if second[2] * second[3] < first[2] * first[3] * 0.25:
        return None
    if abs(first[0] + first[2] / 2 - second[0] - second[2] / 2) < width * 0.15:
        return None
    return sorted((first, second), key=lambda f: f[0] + f[2] / 2)


def vertical_size(width, height):
    """Exact 9:16, even dimensions, without stretching or upscaling."""
    units = min(width // 18, height // 32, 60)
    if units < 1:
        raise ValueError("Video is too small for a vertical clip (minimum 18 × 32).")
    return 18 * units, 32 * units


class FaceDetector:
    def __init__(self):
        if not MODEL.is_file():
            raise ValueError("The bundled YuNet face model is missing. Restore app/assets before exporting.")
        self.detector = cv2.FaceDetectorYN.create(str(MODEL), "", (320, 320), 0.8, 0.3, 100)

    def detect(self, image):
        self.detector.setInputSize((image.shape[1], image.shape[0]))
        _, faces = self.detector.detect(image)
        return [] if faces is None else faces[:, :4].tolist()


def choose_face(faces, previous, width, height):
    if not faces:
        return None
    if previous is None:
        return max(faces, key=lambda f: f[2] * f[3])
    px, py, pw, ph = previous
    candidates = [
        face
        for face in faces
        if math.hypot(
            (face[0] + face[2] / 2 - px - pw / 2) / width, (face[1] + face[3] / 2 - py - ph / 2) / height
        )
        < 0.25
        and 0.4 < face[2] * face[3] / max(pw * ph, 1) < 2.5
    ]
    return (
        min(
            candidates,
            key=lambda f: (f[0] + f[2] / 2 - px - pw / 2) ** 2 + (f[1] + f[3] / 2 - py - ph / 2) ** 2,
        )
        if candidates
        else None
    )


def observe(source, start, end, detector, rotation=0):
    observations = []
    previous_face, previous_thumb, previous_pair = None, None, None
    first_timestamp = None
    last_seen, next_sample = -math.inf, start
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        origin = float(container.start_time / av.time_base) if container.start_time is not None else 0
        container.seek(int((start + origin) / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            timestamp = float(frame.time) - origin
            if timestamp < start:
                continue
            if timestamp >= end:
                break
            if first_timestamp is None:
                # Match setpts=PTS-STARTPTS, including selections between source frames.
                first_timestamp = timestamp
            # Cheap cut checks on every frame avoid panning across an edit between detector samples.
            thumb = np.rot90(frame.reformat(width=64, height=36, format="rgb24").to_ndarray(), rotation)
            cut = scene_changed(previous_thumb, thumb)
            previous_thumb = thumb
            if timestamp + 0.0001 < next_sample and not cut:
                continue
            next_sample = timestamp + SAMPLE_SECONDS
            if cut or timestamp - last_seen > 1.25:
                previous_face = None
            scale = min(1, 640 / max(frame.width, frame.height))
            image = frame.reformat(
                width=max(32, round(frame.width * scale)),
                height=max(32, round(frame.height * scale)),
                format="bgr24",
            ).to_ndarray()
            image = np.ascontiguousarray(np.rot90(image, rotation))
            width, height = (frame.height, frame.width) if rotation % 2 else (frame.width, frame.height)
            faces = detector.detect(image)
            face = choose_face(faces, previous_face, image.shape[1], image.shape[0])
            if face is None and previous_pair is not None and len(faces) == 1:
                # When one participant leaves, immediately frame the remaining person.
                face = faces[0]

            def face_center(face, x_scale=width / image.shape[1], y_scale=height / image.shape[0]):
                return (
                    (face[0] + face[2] / 2) * x_scale,
                    (face[1] + face[3] * 0.4) * y_scale,
                )

            center = None
            if face is not None:
                previous_face, last_seen = face, timestamp
                # Eyes/upper face land near the upper third when vertical cropping has room to move.
                center = face_center(face)
            pair = choose_pair(faces, image.shape[1])
            previous_pair = pair
            observations.append(
                Observation(
                    max(0, timestamp - first_timestamp),
                    center,
                    cut,
                    tuple(face_center(f) for f in pair) if pair else None,
                )
            )
    return observations


def split_screen_ranges(observations, duration):
    """Confirm pairs offline, bridge brief misses, and never bridge a scene cut."""
    ranges = []
    first = last = None

    def finish(end):
        if first is not None and last - first >= 0.25:
            ranges.append((first, min(end, last + SAMPLE_SECONDS)))

    for observation in observations:
        if observation.cut or (last is not None and observation.time - last > 0.35):
            finish(observation.time)
            first = last = None
        if observation.pair:
            if first is None:
                first = observation.time
            last = observation.time
    finish(duration)
    return ranges


def layout_observations(observations, ranges, panel=None):
    """Reset framing at layout transitions; each panel tracks its own face."""
    result = []
    previous_layout = None
    for observation in observations:
        layout = next((i for i, (start, end) in enumerate(ranges) if start <= observation.time < end), None)
        cut = observation.cut or layout != previous_layout
        if panel is None:
            result.append(replace(observation, cut=cut))
        elif layout is not None:
            result.append(
                replace(observation, center=observation.pair[panel] if observation.pair else None, cut=cut)
            )
        previous_layout = layout
    return result


def camera_path(observations, duration, width, height, crop_width, crop_height):
    """Acquire quickly, then ease into a quiet, speed-limited camera over 2.5 seconds."""

    def startup_weight(age):
        # Exponential ease-out reaches the steady response after 2.5 seconds.
        progress = max(0, min(1, age / 2.5))
        return (math.exp(-3 * progress) - math.exp(-3)) / (1 - math.exp(-3))

    if not observations:
        observations = [Observation(0, None)]
    segments = []
    for observation in observations:
        if not segments or observation.cut:
            segments.append([])
        segments[-1].append(observation)
    path = []
    for segment in segments:
        visible = [o.center for o in segment if o.center is not None]
        fallback = (width / 2, height / 2 - crop_height * 0.18)
        # For a seated presenter, keep the camera completely stationary for the entire shot.
        stationary = bool(visible) and all(
            max(p[axis] for p in visible) - min(p[axis] for p in visible) < extent * 0.20
            for axis, extent in enumerate((crop_width, crop_height))
        )
        anchor = tuple(median(p[a] for p in visible[:6]) for a in (0, 1)) if visible else fallback
        if stationary:
            anchor = tuple(median(p[a] for p in visible) for a in (0, 1))

        def origin(center):
            return [
                max(0, min(width - crop_width, center[0] - crop_width / 2)),
                max(0, min(height - crop_height, center[1] - crop_height * 0.32)),
            ]

        position, target, velocity = origin(anchor), origin(anchor), [0.0, 0.0]
        last_seen, last_center = segment[0].time, anchor
        last_time = segment[0].time
        for index, observation in enumerate(segment):
            nearby = [o.center for o in segment[max(0, index - 3) : index + 4] if o.center is not None]
            if observation.center is not None:
                last_center = tuple(median(p[a] for p in nearby) for a in (0, 1))
                last_seen = observation.time
            center = (
                (anchor if stationary else last_center) if observation.time - last_seen <= 1.25 else fallback
            )
            desired = origin(center)
            dt = max(0, observation.time - last_time)
            age = observation.time - segment[0].time
            # Restart on scene cuts, but not on each detection or reacquisition.
            weight = startup_weight(age)
            for axis, extent in enumerate((crop_width, crop_height)):
                if abs(desired[axis] - target[axis]) > extent * (0.16 - 0.10 * weight):
                    target[axis] = desired[axis]
                # Integrate in small steps for stable behavior even with variable frame rates/gaps.
                steps = max(1, math.ceil(dt / 0.025))
                step = dt / steps
                for substep in range(steps):
                    weight = startup_weight(last_time + (substep + 1) * step - segment[0].time)
                    response = 2.4 + 5.6 * weight
                    acceleration_limit = extent * (0.7 + 1.4 * weight)
                    speed_limit = extent * (0.35 + 0.65 * weight)
                    acceleration = (
                        response**2 * (target[axis] - position[axis]) - 2 * response * velocity[axis]
                    )
                    velocity[axis] += max(-acceleration_limit, min(acceleration_limit, acceleration)) * step
                    velocity[axis] = max(-speed_limit, min(speed_limit, velocity[axis]))
                    position[axis] += velocity[axis] * step
                position[axis] = max(0, min((width - crop_width, height - crop_height)[axis], position[axis]))
            path.append((observation.time, *position, observation.cut))
            last_time = observation.time
    if path[0][0] > 0:
        path.insert(0, (0, *path[0][1:3], False))
    path.append((duration, *path[-1][1:3], False))
    return path


def write_commands(path, destination, tracks=None):
    """Time-based commands work with variable frame rates; cuts switch without interpolation."""
    commands = []
    for target, track in tracks or [("crop", path)]:
        for left, right in pairwise(track):
            steps = max(1, math.ceil((right[0] - left[0]) * 60))
            for index in range(steps):
                fraction = index / steps
                timestamp = left[0] + fraction * (right[0] - left[0])
                x, y = (
                    left[1:3]
                    if right[3]
                    else (
                        left[1] + fraction * (right[1] - left[1]),
                        left[2] + fraction * (right[2] - left[2]),
                    )
                )
                # Truncate instead of rounding a command past its intended video frame.
                time_text = f"{math.floor(timestamp * 1_000_000) / 1_000_000:.6f}"
                commands.append((timestamp, f"{time_text} {target} x {round(x)}, {target} y {round(y)};"))
    destination.write_text("\n".join(line for _, line in sorted(commands)), encoding="utf-8")


def plan_reframe(source, start, end, metadata, mode, commands_file, detector=None):
    width, height = metadata["width"], metadata["height"]
    with av.open(str(source)) as container:
        first = next(container.decode(video=0), None)
        angle = first.rotation if first is not None else 0
    if abs(angle / 90 - round(angle / 90)) > 0.01:
        raise ValueError("Vertical export requires a video rotated by a multiple of 90 degrees.")
    rotation = round(angle / 90) % 4
    if rotation % 2:
        width, height = height, width
    # Keep the largest source crop; independently cap exported resolution at 1080 × 1920.
    crop_width, crop_height = vertical_size(width, height)
    source_units = min(width // 18, height // 32)
    source_crop = (source_units * 18, source_units * 32)
    cw, ch = source_crop
    observations = (
        observe(source, start, end, detector or FaceDetector(), rotation) if mode == "face_follow" else []
    )
    duration = end - start
    ranges = split_screen_ranges(observations, duration)
    path = camera_path(layout_observations(observations, ranges), duration, width, height, cw, ch)
    # Only the generated basename enters filter syntax; arbitrary source paths stay in argv.
    # Normalize before cropping: exact odd offsets on 10-bit subsampled frames can
    # misalign pixel data in FFmpeg. Let crop align to the 8-bit chroma grid as well,
    # so moving the frame does not alternate the luma/chroma alignment.
    prefix = f"setpts=PTS-STARTPTS,format=yuv420p,sendcmd=f={commands_file.name},"
    if ranges:
        panels = [
            camera_path(
                layout_observations(observations, ranges, panel), duration, width, height, cw, ch // 2
            )
            for panel in (0, 1)
        ]
        tracks = [("crop@single", path), ("crop@top", panels[0]), ("crop@bottom", panels[1])]
        write_commands(path, commands_file, tracks)
        branches = []
        for label, (_, track), source_height, output_height in zip(
            ("single", "top", "bottom"),
            tracks,
            (ch, ch // 2, ch // 2),
            (crop_height, crop_height // 2, crop_height // 2),
            strict=True,
        ):
            branches.append(
                f"[{label}_in]crop@{label}={cw}:{source_height}:{round(track[0][1])}:{round(track[0][2])},"
                f"scale={crop_width}:{output_height},setsar=1[{label}]"
            )
        enabled = "+".join(
            f"gte(t,{math.floor(a * 1_000_000) / 1_000_000:.6f})"
            f"*lt(t,{math.floor(b * 1_000_000) / 1_000_000:.6f})"
            for a, b in ranges
        )
        filters = (
            prefix
            + "split=3[single_in][top_in][bottom_in];"
            + ";".join(branches)
            + f";[top][bottom]vstack[duo];[single][duo]overlay=enable='{enabled}'"
        )
    else:
        write_commands(path, commands_file)
        filters = (
            prefix + f"crop={cw}:{ch}:{round(path[0][1])}:{round(path[0][2])},"
            f"scale={crop_width}:{crop_height},setsar=1"
        )
    return (
        filters,
        (crop_width, crop_height),
        {
            "mode": mode,
            "face_samples": sum(o.center is not None for o in observations),
            "samples": len(observations),
            "scene_cuts": sum(o.cut for o in observations),
            "split_screen_ranges": [{"start": a, "end": b} for a, b in ranges],
            "fallback": "center_crop"
            if mode == "face_follow" and not any(o.center for o in observations)
            else None,
        },
    )
