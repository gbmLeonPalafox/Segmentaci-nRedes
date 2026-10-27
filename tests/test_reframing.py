import math
import subprocess
from itertools import pairwise

import av
import numpy as np
import pytest

from app.clips import render_clip
from app.pipeline import ffmpeg_path, inspect_media
from app.reframing import (
    FaceDetector,
    Observation,
    camera_path,
    choose_face,
    choose_pair,
    layout_observations,
    observe,
    scene_changed,
    split_screen_ranges,
    vertical_size,
    write_commands,
)


def test_small_head_movements_produce_a_completely_locked_camera():
    observations = [Observation(i / 8, (720 + 15 * math.sin(i), 240)) for i in range(80)]
    path = camera_path(observations, 10, 1920, 1080, 594, 1056)
    assert max(p[1] for p in path) - min(p[1] for p in path) == 0
    assert max(p[2] for p in path) - min(p[2] for p in path) == 0


def test_large_movement_pans_gently_and_stays_in_bounds():
    observations = [Observation(i / 8, (400 + min(i, 40) * 15, 240)) for i in range(100)]
    path = camera_path(observations, 12.5, 1920, 1080, 594, 1056)
    assert path[-1][1] > path[0][1] + 400
    for left, right in pairwise(path):
        assert 0 <= right[1] <= 1920 - 594
        assert 0 <= right[2] <= 1080 - 1056
        max_speed = 0.35 if left[0] >= 2.5 else 1
        assert abs(right[1] - left[1]) <= 594 * max_speed * (right[0] - left[0]) + 0.001


def test_initial_acquisition_is_faster_than_the_same_movement_later():
    def movement(delay):
        observations = [Observation(i / 8, (400 if i / 8 < delay else 1000, 240)) for i in range(80)]
        return camera_path(observations, 10, 1920, 1080, 594, 1056)

    early, late = movement(0.75), movement(4)
    early_travel = early[10][1] - early[0][1]
    late_travel = late[36][1] - late[0][1]
    assert early_travel > late_travel * 1.5
    # A late acquisition still follows a large movement instead of freezing.
    assert late[-1][1] > late[0][1] + 500


def test_scene_cut_restarts_initial_acquisition():
    first_shot = [Observation(i / 8, (400 if i < 6 else 1000, 240)) for i in range(32)]
    second_shot = [Observation(o.time + 4, o.center, i == 0) for i, o in enumerate(first_shot)]
    path = camera_path(first_shot + second_shot, 8, 1920, 1080, 594, 1056)
    assert path[32][3]
    np.testing.assert_allclose([p[1:3] for p in path[:32]], [p[1:3] for p in path[32:64]])


def test_missing_faces_hold_briefly_then_ease_to_center():
    observations = [Observation(i / 8, (400, 240) if i < 8 else None) for i in range(100)]
    path = camera_path(observations, 12.5, 1920, 1080, 594, 1056)
    assert path[14][1] == path[0][1]
    assert path[-1][1] == pytest.approx((1920 - 594) / 2, abs=2)


def test_scene_cuts_reset_camera_without_a_pan(tmp_path):
    observations = [Observation(i / 8, (400 if i < 16 else 1500, 240), i == 16) for i in range(40)]
    path = camera_path(observations, 5, 1920, 1080, 594, 1056)
    assert path[16][3] and path[16][1] - path[15][1] > 1000
    commands = tmp_path / "crop.cmd"
    write_commands(path, commands)
    lines = commands.read_text().splitlines()
    assert "crop x 103," in next(line for line in lines if line.startswith("1.984375 "))
    assert "crop x 1203," in next(line for line in lines if line.startswith("2.000000 "))


def test_face_association_keeps_nearby_face_when_bystander_is_larger():
    previous = [100, 100, 50, 60]
    assert choose_face([[105, 100, 50, 60], [600, 80, 120, 140]], previous, 960, 540)[0] == 105
    assert choose_face([[600, 80, 120, 140]], previous, 960, 540) is None


def test_scene_cut_detects_spatial_changes_with_identical_average_color():
    before = np.zeros((36, 64, 3), dtype=np.uint8)
    before[:, :24] = 160
    after = np.flip(before, axis=1)
    assert scene_changed(before, after)
    assert not scene_changed(before, np.roll(before, 1, axis=1))
    assert not scene_changed(None, before)


def test_two_faces_keep_left_right_order_and_ignore_tiny_bystanders():
    left, right = [100, 40, 60, 80], [450, 40, 70, 85]
    assert choose_pair([right, left], 640) == [left, right]
    assert choose_pair([left, [450, 40, 10, 10]], 640) is None
    assert choose_pair([left, [110, 40, 60, 80]], 640) is None


def test_split_layout_confirms_pairs_bridges_misses_and_stops_at_cuts():
    pair = ((300, 200), (1500, 200))
    observations = [
        Observation(
            i / 8, (300, 200), cut=i == 24, pair=pair if (8 <= i < 24 and i != 15) or i == 30 else None
        )
        for i in range(40)
    ]
    ranges = split_screen_ranges(observations, 5)
    assert ranges == [(1, 3)]
    assert layout_observations(observations, ranges)[8].cut
    assert layout_observations(observations, ranges)[24].cut
    panels = layout_observations(observations, ranges, 1)
    assert panels[0].center == pair[1]
    assert next(o for o in panels if o.time == 15 / 8).center is None


def test_split_layout_never_bridges_a_scene_cut_even_with_two_faces():
    pair = ((300, 200), (1500, 200))
    observations = [Observation(i / 8, pair[0], i == 8, pair) for i in range(16)]
    assert split_screen_ranges(observations, 2) == [(0, 1), (1, 2)]


@pytest.mark.parametrize("size", [(1920, 1080), (1080, 1920), (721, 721), (3840, 2160)])
def test_vertical_output_is_exact_even_and_never_upscaled(size):
    width, height = vertical_size(*size)
    assert width * 16 == height * 9
    assert width % 2 == height % 2 == 0
    assert width <= size[0] and height <= size[1]
    assert width <= 1080 and height <= 1920


def test_bundled_model_loads_and_handles_no_face():
    assert FaceDetector().detect(np.zeros((320, 320, 3), dtype=np.uint8)) == []


class RedFaceDetector:
    """A known bounding box from a synthetic subject, exercising decoding and real rendering."""

    def detect(self, image):
        mask = (image[:, :, 2] > 160) & (image[:, :, 0] < 80)
        y, x = np.where(mask)
        return [[float(x.min()), float(y.min()), float(x.max() - x.min()), float(y.max() - y.min())]]


class TwoFaceDetector:
    def detect(self, image):
        faces = []
        for channel, opposite in ((2, 0), (0, 2)):
            y, x = np.where((image[:, :, channel] > 160) & (image[:, :, opposite] < 80))
            if len(x):
                faces.append(
                    [float(x.min()), float(y.min()), float(x.max() - x.min()), float(y.max() - y.min())]
                )
        return list(reversed(faces))  # Detection order must not determine the panel order.


@pytest.mark.parametrize("pixel_format", ["yuv420p", "yuv420p10le"])
@pytest.mark.parametrize("subtitles", [False, True])
def test_render_splits_only_two_person_interval_then_reframes_remaining_face(tmp_path, pixel_format, subtitles):
    silent = tmp_path / "two-people.mp4"
    with av.open(str(silent), "w") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width, stream.height, stream.pix_fmt = 640, 360, pixel_format
        for i in range(96):
            image = np.zeros((360, 640, 3), dtype=np.uint8)
            image[:, :, 1] = 80
            if i < 72:
                image[80:170, 100:155] = (240, 0, 0)
            if i >= 24:
                image[80:170, 480:535] = (0, 0, 240)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    source = tmp_path / "two-people-audio.mp4"
    subprocess.run(
        [
            ffmpeg_path(),
            "-v",
            "error",
            "-i",
            str(silent),
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=4",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(source),
        ],
        check=True,
    )
    destination = tmp_path / "mixed-layout.mp4"
    words = [{"word": "Subtitles", "start": 1.2, "end": 2.8}] if subtitles else None
    rendered = render_clip(
        source, destination, 0, 4, inspect_media(source), "face_follow", TwoFaceDetector(), words
    )
    assert rendered["framing"]["split_screen_ranges"] == [{"start": 1, "end": 3}]
    assert rendered["audio_codec"] == "aac"
    assert rendered["duration"] == pytest.approx(4, abs=0.1)
    with av.open(str(destination)) as container:
        for frame in container.decode(video=0):
            rgb = frame.to_ndarray(format="rgb24")
            red = (rgb[:, :, 0] > 160) & (rgb[:, :, 2] < 80)
            blue = (rgb[:, :, 2] > 160) & (rgb[:, :, 0] < 80)
            midpoint = rgb.shape[0] // 2
            if subtitles and 1.3 <= frame.time < 2.8:
                white = np.all(rgb > 170, axis=2)
                assert white[midpoint:].sum() > 20
                assert white[:midpoint].sum() == 0
            if 1 <= frame.time < 3:
                assert red[:midpoint].sum() > 2000
                assert blue[midpoint:].sum() > 2000
                assert red[midpoint:].sum() < 100
                assert blue[:midpoint].sum() < 100
            elif frame.time < 1:
                assert red.sum() > 2000
                assert blue.sum() < 100
            else:
                assert blue.sum() > 2000
                assert red.sum() < 100
    assert not list(tmp_path.glob("*.crop.cmd"))


@pytest.fixture
def walking_subject(tmp_path):
    source = tmp_path / "subject.mp4"
    # A red subject travels right; green background makes it easy to verify the rendered framing.
    with av.open(str(source), "w") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width, stream.height, stream.pix_fmt = 640, 360, "yuv420p"
        for i in range(144):
            image = np.zeros((360, 640, 3), dtype=np.uint8)
            image[:, :, 1] = 80
            x = 100 + round(min(i, 100) * 3.2)
            image[80:170, x : x + 55] = (240, 0, 0)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with_audio = tmp_path / "subject with audio.mp4"
    subprocess.run(
        [
            ffmpeg_path(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=6",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(with_audio),
        ],
        check=True,
    )
    return with_audio


@pytest.mark.parametrize("pixel_format", ["yuv420p", "yuv420p10le"])
def test_face_follow_render_tracks_motion_and_keeps_audio(walking_subject, tmp_path, pixel_format):
    if pixel_format == "yuv420p10le":
        ten_bit = tmp_path / "ten-bit.mp4"
        subprocess.run(
            [
                ffmpeg_path(),
                "-v",
                "error",
                "-i",
                str(walking_subject),
                "-c:v",
                "libx264",
                "-pix_fmt",
                pixel_format,
                "-c:a",
                "copy",
                str(ten_bit),
            ],
            check=True,
        )
        walking_subject = ten_bit
    with av.open(str(walking_subject)) as container:
        assert next(container.decode(video=0)).format.name == pixel_format
    destination = tmp_path / "follow.mp4"
    metadata = inspect_media(walking_subject)
    output = render_clip(walking_subject, destination, 0.25, 5.75, metadata, "face_follow", RedFaceDetector())
    assert output["width"] * 16 == output["height"] * 9
    assert output["audio_codec"] == "aac"
    assert output["duration"] == pytest.approx(5.5, abs=0.15)
    assert output["framing"]["face_samples"] > 35
    assert output["framing"]["scene_cuts"] == 0
    with av.open(str(destination)) as container:
        centers = []
        for frame in container.decode(video=0):
            rgb = frame.to_ndarray(format="rgb24")
            _y, x = np.where((rgb[:, :, 0] > 160) & (rgb[:, :, 2] < 80))
            assert len(x) > 2000, "The moving subject left the crop"
            centers.append(float(x.mean()))
            # Check every frame, including odd crop offsets during pans. A 10-bit
            # crop alignment bug produced magenta noise while keeping the subject visible.
            assert np.abs(rgb[-40:].astype(float) - (0, 80, 0)).mean() < 5
            assert np.abs(rgb[100:140, round(x.mean())].astype(float) - (240, 0, 0)).mean() < 10
    assert max(centers) < output["width"] - 10
    assert not list(tmp_path.glob("*.crop.cmd"))


def test_real_detector_no_face_falls_back_and_center_crop_is_identical(walking_subject, tmp_path):
    metadata = inspect_media(walking_subject)
    follow = render_clip(walking_subject, tmp_path / "follow.mp4", 1, 2, metadata, "face_follow")
    centered = render_clip(walking_subject, tmp_path / "center.mp4", 1, 2, metadata, "center_crop")
    assert follow["framing"]["fallback"] == "center_crop"
    assert follow["framing"]["face_samples"] == 0
    assert (follow["width"], follow["height"]) == (centered["width"], centered["height"])
    with av.open(str(tmp_path / "follow.mp4")) as a, av.open(str(tmp_path / "center.mp4")) as b:
        assert np.array_equal(next(a.decode(video=0)).to_ndarray(), next(b.decode(video=0)).to_ndarray())


def test_observe_uses_only_selected_range(walking_subject):
    observations = observe(walking_subject, 2, 3, RedFaceDetector())
    assert 7 <= len(observations) <= 9
    assert observations[0].time < 0.05
    assert observations[-1].time < 1


def test_observations_align_to_first_rendered_frame_for_start_between_frames(walking_subject):
    observations = observe(walking_subject, 0.26, 1.26, RedFaceDetector())
    assert observations[0].time == 0


def test_rotated_input_exports_upright_vertical_video(walking_subject, tmp_path):
    rotated = tmp_path / "rotated.mp4"
    subprocess.run(
        [
            ffmpeg_path(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-display_rotation:v:0",
            "90",
            "-i",
            str(walking_subject),
            "-c",
            "copy",
            str(rotated),
        ],
        check=True,
    )
    with av.open(str(rotated)) as container:
        assert next(container.decode(video=0)).rotation == 90
    result = render_clip(rotated, tmp_path / "upright.mp4", 0, 1, inspect_media(rotated), "center_crop")
    assert (result["width"], result["height"]) == (360, 640)
    with av.open(str(tmp_path / "upright.mp4")) as container:
        upright = next(container.decode(video=0))
        assert upright.rotation == 0
        with av.open(str(walking_subject)) as original:
            expected = np.rot90(next(original.decode(video=0)).to_ndarray(format="rgb24"))
        assert np.abs(upright.to_ndarray(format="rgb24").astype(float) - expected).mean() < 3
