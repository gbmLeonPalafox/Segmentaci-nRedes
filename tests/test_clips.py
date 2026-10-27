import json
import subprocess
import threading
import time

import av
import pytest
from fastapi.testclient import TestClient

from app.clips import highlight_hash, validate_ranges
from app.main import create_app
from app.pipeline import ffmpeg_path, inspect_media


@pytest.fixture
def media(tmp_path):
    target = tmp_path / "original.mp4"
    subprocess.run(
        [
            ffmpeg_path(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=160x120:r=25:d=1",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=160x120:r=25:d=3",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=4",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map",
            "[v]",
            "-map",
            "2:a",
            "-c:v",
            "libx264",
            "-g",
            "250",
            "-c:a",
            "aac",
            "-shortest",
            str(target),
        ],
        capture_output=True,
        check=True,
    )
    return target


def seed(app, media, identity="video-test"):
    folder = app.state.store.root / identity
    folder.mkdir(exist_ok=True)
    (folder / "source.mp4").write_bytes(media.read_bytes())
    highlights = {
        "highlights": [
            {"id": "highlight_001", "rank": 1, "title": "Blue passage", "start": 1.28, "end": 2.76},
            {"id": "highlight_002", "rank": 2, "title": "Another passage", "start": 3, "end": 3.8},
        ]
    }
    (folder / "highlights.json").write_text(json.dumps(highlights))
    (folder / "transcript.json").write_text(json.dumps({"segments": []}))
    app.state.store.save(
        {
            "id": identity,
            "title": "Test video",
            "source_type": "upload",
            "model": "tiny",
            "status": "completed",
            "highlights_status": "completed",
            "source_file": "source.mp4",
        }
    )
    return folder, highlights


def wait_clips(client, job_id="video-test"):
    for _ in range(400):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job.get("clips_status") in {"completed", "failed"}:
            return job
        time.sleep(0.01)
    pytest.fail("Clip rendering did not finish")


def test_clips_render_accurate_ranges_dimensions_and_downloads(tmp_path, media):
    app = create_app(tmp_path / "data")
    folder, highlights = seed(app, media)
    with TestClient(app) as client:
        assert client.post("/api/jobs/video-test/clips").status_code == 202
        job = wait_clips(client)
        assert job["clips_status"] == "completed", job
        data = client.get("/api/jobs/video-test/clips").json()
        assert data["highlights_sha256"] == highlight_hash(highlights)
        assert len(data["clips"]) == 2
        for item in data["clips"]:
            assert (item["width"], item["height"]) == (160, 120)
            assert abs(item["duration"] - (item["source_end"] - item["source_start"])) < 0.15
            path = folder / data["directory"] / item["filename"]
            metadata = inspect_media(path)
            assert metadata["video_codec"] == "h264" and metadata["audio_codec"] == "aac"
        # The first source second is red. The selected interval starts between keyframes in blue.
        first = data["clips"][0]
        with av.open(str(folder / data["directory"] / first["filename"])) as video:
            frame = next(video.decode(video=0)).reformat(format="rgb24")
            plane = frame.planes[0]
            raw = bytes(plane)
            pixels = b"".join(
                raw[row * plane.line_size:row * plane.line_size + frame.width * 3]
                for row in range(frame.height)
            )
            rgb = [sum(pixels[channel::3]) / (frame.width * frame.height) for channel in range(3)]
            assert rgb[2] > 200 and rgb[0] < 20
        preview = client.get(
            "/api/jobs/video-test/clips/highlight_001/media", headers={"Range": "bytes=0-99"}
        )
        assert preview.status_code == 206 and len(preview.content) == 100
        download = client.get("/api/jobs/video-test/clips/highlight_001/media?download=true")
        assert download.headers["content-type"] == "video/mp4"
        assert "attachment" in download.headers["content-disposition"]
        assert client.get("/api/jobs/video-test/clips/unknown/media").status_code == 404
        assert client.get(
            "/api/jobs/video-test/clips/highlight_001/media?revision=previous-batch"
        ).status_code == 409
        assert client.get(
            f"/api/jobs/video-test/clips/highlight_001/media?revision={data['revision']}",
            headers={"Range": "bytes=0-99"},
        ).status_code == 206
        # A successful rebuild replaces files and removes the old batch.
        previous = folder / data["directory"]
        assert client.post("/api/jobs/video-test/clips").status_code == 202
        assert wait_clips(client)["clips_status"] == "completed"
        assert not previous.exists()
        assert len(list(folder.glob("clips-*"))) == 1


def test_changed_highlights_cannot_download_old_clips(tmp_path, media):
    app = create_app(tmp_path / "data")
    folder, data = seed(app, media)
    with TestClient(app) as client:
        client.post("/api/jobs/video-test/clips")
        assert wait_clips(client)["clips_status"] == "completed"
        data["highlights"][0]["end"] = 2.6
        (folder / "highlights.json").write_text(json.dumps(data))
        assert client.get("/api/jobs/video-test/clips").status_code == 409
        assert client.get("/api/jobs/video-test/clips/highlight_001/media").status_code == 409


@pytest.mark.parametrize("mode", ["center_crop", "face_follow"])
def test_vertical_modes_are_saved_validated_and_rendered(tmp_path, media, mode):
    app = create_app(tmp_path / "data")
    seed(app, media)
    with TestClient(app) as client:
        assert client.post("/api/jobs/video-test/clips", json={"mode": "unknown"}).status_code == 422
        assert client.get("/api/jobs/video-test").json().get("clips_status") is None
        response = client.post("/api/jobs/video-test/clips", json={"mode": mode})
        assert response.status_code == 202
        assert response.json()["clips_mode"] == mode
        job = wait_clips(client)
        assert job["clips_status"] == "completed", job
        batch = client.get("/api/jobs/video-test/clips").json()
        assert batch["mode"] == mode
        for clip in batch["clips"]:
            assert clip["width"] * 16 == clip["height"] * 9
            assert clip["framing"]["mode"] == mode
            if mode == "face_follow":
                assert clip["framing"]["fallback"] == "center_crop"
        # An explicit original export must replace the vertical batch cleanly.
        assert client.post("/api/jobs/video-test/clips", json={"mode": "original"}).status_code == 202
        assert wait_clips(client)["clips_status"] == "completed"
        batch = client.get("/api/jobs/video-test/clips").json()
        assert batch["mode"] == "original"
        assert all((clip["width"], clip["height"]) == (160, 120) for clip in batch["clips"])


def test_delete_removes_all_files_and_metadata_only_for_selected_video(tmp_path, media):
    app = create_app(tmp_path / "data")
    folder, _ = seed(app, media)
    neighbor, _ = seed(app, media, "keep-this-video")
    (folder / "preview.mp4").write_bytes(b"preview")
    (folder / "audio.wav").write_bytes(b"audio")
    (folder / ".clips-abandoned").mkdir()
    (folder / ".clips-abandoned" / "partial.mp4").write_bytes(b"partial")
    with TestClient(app) as client:
        client.post("/api/jobs/video-test/clips")
        assert wait_clips(client)["clips_status"] == "completed"
        size = client.get("/api/jobs/video-test/storage").json()["size_bytes"]
        assert size > media.stat().st_size
        result = client.delete("/api/jobs/video-test")
        assert result.status_code == 200 and result.json()["size_bytes"] == size
        assert not folder.exists() and neighbor.exists()
        assert client.get("/api/jobs/video-test").status_code == 404
        assert client.get("/api/jobs/video-test/media").status_code == 404
        assert client.get("/api/jobs/keep-this-video").status_code == 200
        assert client.delete("/api/jobs/video-test").status_code == 404


@pytest.mark.parametrize(
    "fields",
    [
        {"status": "queued"},
        {"status": "running"},
        {"highlights_status": "queued"},
        {"highlights_status": "running"},
        {"clips_status": "queued"},
        {"clips_status": "running"},
    ],
)
def test_delete_rejects_busy_video_without_removing_files(tmp_path, media, fields):
    app = create_app(tmp_path / "data")
    folder, _ = seed(app, media)
    with TestClient(app) as client:
        app.state.store.update("video-test", **fields)
        assert client.delete("/api/jobs/video-test").status_code == 409
        assert (folder / "source.mp4").exists()
        # Keep shutdown/recovery fixtures out of other background workers.
        app.state.store.update(
            "video-test", status="completed", highlights_status="completed", clips_status=None
        )


def test_running_renderer_blocks_delete_and_highlight_regeneration(tmp_path, media, monkeypatch):
    import app.clips

    actual = app.clips.render_clip
    entered, release = threading.Event(), threading.Event()

    def delayed(*args):
        entered.set()
        assert release.wait(5)
        return actual(*args)

    monkeypatch.setattr(app.clips, "render_clip", delayed)
    app = create_app(tmp_path / "data")
    folder, _ = seed(app, media)
    with TestClient(app) as client:
        client.post("/api/jobs/video-test/clips")
        assert entered.wait(5)
        try:
            assert client.delete("/api/jobs/video-test").status_code == 409
            assert client.post("/api/jobs/video-test/highlights").status_code == 409
            assert client.post("/api/jobs/video-test/clips").status_code == 409
            assert folder.exists()
        finally:
            release.set()
        assert wait_clips(client)["clips_status"] == "completed"
        assert client.delete("/api/jobs/video-test").status_code == 200


def test_failed_render_cleans_partial_output(tmp_path, media, monkeypatch):
    def fail(source, output, *args):
        output.write_bytes(b"partial output")
        raise ValueError("Controlled rendering failure")

    monkeypatch.setattr("app.clips.render_clip", fail)
    app = create_app(tmp_path / "data")
    folder, _ = seed(app, media)
    with TestClient(app) as client:
        assert client.post("/api/jobs/video-test/clips").status_code == 202
        assert wait_clips(client)["clips_status"] == "failed"
        assert not list(folder.glob(".clips-*"))
        assert not (folder / "clips.json").exists()
        assert client.get("/api/jobs/video-test").json()["status"] == "completed"


def test_delete_failure_can_be_retried(tmp_path, media, monkeypatch):
    import app.main as app_module

    actual = app_module.shutil.rmtree
    app = create_app(tmp_path / "data")
    folder, _ = seed(app, media)
    with TestClient(app) as client:

        def fail(*args, **kwargs):
            raise PermissionError("controlled")

        with monkeypatch.context() as patch:
            patch.setattr(app_module.shutil, "rmtree", fail)
            assert client.delete("/api/jobs/video-test").status_code == 500
            assert client.get("/api/jobs/video-test").json()["status"] == "delete_failed"
        assert app_module.shutil.rmtree == actual
        assert client.delete("/api/jobs/video-test").status_code == 200
        assert not folder.exists()


def test_restart_finishes_interrupted_deletion(tmp_path, media):
    app = create_app(tmp_path / "data")
    folder, _ = seed(app, media)
    app.state.store.update("video-test", status="deleting")
    with TestClient(app) as client:
        assert client.get("/api/jobs/video-test").status_code == 404
        assert not folder.exists()


def test_restart_removes_interrupted_clip_batches_and_preserves_published_batch(tmp_path, media):
    app = create_app(tmp_path / "data")
    folder, _ = seed(app, media)
    for name in ["clips-published", "clips-orphan", ".clips-partial"]:
        (folder / name).mkdir()
        (folder / name / "clip.mp4").write_bytes(b"fixture")
    (folder / "clips.json").write_text(json.dumps({"directory": "clips-published"}))
    app.state.store.update("video-test", clips_status="running")
    with TestClient(app) as client:
        assert client.get("/api/jobs/video-test").json()["clips_status"] == "failed"
        assert (folder / "clips-published" / "clip.mp4").exists()
        assert not (folder / "clips-orphan").exists()
        assert not (folder / ".clips-partial").exists()


@pytest.mark.parametrize("start,end", [(-1, 2), (2, 1), (0, 5), (float("nan"), 1), (0, float("inf"))])
def test_invalid_clip_ranges(start, end):
    with pytest.raises(ValueError):
        validate_ranges({"highlights": [{"id": "highlight_001", "start": start, "end": end}]}, 4)


@pytest.mark.parametrize("mode", ["original", "center_crop", "face_follow"])
def test_dynamic_subtitles_render_visible_text_and_can_be_disabled(tmp_path, media, mode):
    import numpy as np

    app = create_app(tmp_path / "data")
    folder, _ = seed(app, media)
    transcript = {"segments": [{"words": [
        {"word": "Hola", "start": 1.5, "end": 1.8},
        {"word": "al", "start": 1.8, "end": 1.8},
        {"word": "mundo", "start": 1.8, "end": 2.2},
        {"word": "Prueba", "start": 3.2, "end": 3.6},
    ]}]}
    saved = json.dumps(transcript)
    (folder / "transcript.json").write_text(saved)
    with TestClient(app) as client:
        response = client.post("/api/jobs/video-test/clips", json={"mode": mode, "subtitles": True})
        assert response.status_code == 202
        assert response.json()["clips_subtitles"] is True
        job = wait_clips(client)
        assert job["clips_status"] == "completed", job
        batch = client.get("/api/jobs/video-test/clips").json()
        assert batch["subtitles"] is True
        clip = batch["clips"][0]
        assert clip["captions"] == {"mode": "dynamic", "word_count": 3}
        with av.open(str(folder / batch["directory"] / clip["filename"])) as container:
            frames = list(container.decode(video=0))
        def white_pixels(frame):
            rgb = frame.to_ndarray(format="rgb24")
            return np.count_nonzero(np.all(rgb > 100, axis=2))  # Anti-aliased text on the tiny 54px fixture.
        assert white_pixels(frames[0]) == 0
        assert white_pixels(frames[9]) > 0  # Word appears at clip-relative ~0.22 s.
        assert white_pixels(frames[-1]) == 0
        assert (folder / "transcript.json").read_text() == saved
        assert client.post("/api/jobs/video-test/clips", json={"mode": mode, "subtitles": False}).status_code == 202
        assert wait_clips(client)["clips_status"] == "completed"
        batch = client.get("/api/jobs/video-test/clips").json()
        assert batch["subtitles"] is False
        assert all(c["captions"]["mode"] == "off" for c in batch["clips"])


def test_subtitles_without_word_times_fail_without_publishing(tmp_path, media):
    app = create_app(tmp_path / "data")
    folder, _ = seed(app, media)
    with TestClient(app) as client:
        assert client.post("/api/jobs/video-test/clips", json={"subtitles": True}).status_code == 202
        job = wait_clips(client)
        assert job["clips_status"] == "failed"
        assert "word timestamps" in job["clips_error"]
        assert not (folder / "clips.json").exists()
        assert not list(folder.glob(".clips-*"))


@pytest.mark.parametrize("subtitles", [False, True])
def test_five_clip_batch_then_regenerate_only_one(tmp_path, media, monkeypatch, subtitles):
    import app.clips as clips_module

    app = create_app(tmp_path / "data")
    folder, highlights = seed(app, media)
    highlights["highlights"] = [
        {"id": f"highlight_{i + 1:03}", "rank": i + 1, "title": f"Clip {i + 1}",
         "start": .2 + .65 * i, "end": .6 + .65 * i} for i in range(5)
    ]
    (folder / "highlights.json").write_text(json.dumps(highlights))
    (folder / "transcript.json").write_text(json.dumps({"segments": [{"words": [
        {"word": "Hola", "start": h["start"], "end": h["end"]} for h in highlights["highlights"]
    ]}]}))
    with TestClient(app) as client:
        assert client.post("/api/jobs/video-test/clips", json={"mode": "center_crop", "subtitles": subtitles}).status_code == 202
        assert wait_clips(client)["clips_status"] == "completed"
        before = client.get("/api/jobs/video-test/clips").json()
        assert len(before["clips"]) == 5
        old_files = {c["id"]: folder / before["directory"] / c["filename"] for c in before["clips"]}
        old_stats = {key: (p.stat().st_ino, p.read_bytes()) for key, p in old_files.items()}
        actual = clips_module.render_clip
        calls = []
        entered, release = threading.Event(), threading.Event()

        def delayed(*args):
            calls.append(args)
            entered.set()
            assert release.wait(5)
            return actual(*args)

        monkeypatch.setattr(clips_module, "render_clip", delayed)
        target = "highlight_003"
        assert client.post(f"/api/jobs/video-test/clips/{target}/regenerate").status_code == 202
        assert entered.wait(5)
        try:
            # Every old player/download remains usable throughout single regeneration.
            for c in before["clips"]:
                assert client.get(f"/api/jobs/video-test/clips/{c['id']}/media?revision={c['revision']}").status_code == 200
            assert client.post("/api/jobs/video-test/clips").status_code == 409
            assert client.post(f"/api/jobs/video-test/clips/{target}/regenerate").status_code == 409
            assert client.delete("/api/jobs/video-test").status_code == 409
            assert client.post("/api/jobs/video-test/highlights").status_code == 409
        finally:
            release.set()
        job = wait_clips(client)
        assert job["clips_status"] == "completed" and not job["clips_error"], job
        after = client.get("/api/jobs/video-test/clips").json()
        assert len(calls) == 1
        assert calls[0][2:4] == (highlights["highlights"][2]["start"], highlights["highlights"][2]["end"])
        assert after["mode"] == "center_crop" and after["subtitles"] == subtitles
        assert len(after["clips"]) == 5
        for c in after["clips"]:
            p = folder / after["directory"] / c["filename"]
            old = next(x for x in before["clips"] if x["id"] == c["id"])
            if c["id"] != target:
                assert (p.stat().st_ino, p.read_bytes()) == old_stats[c["id"]]
                assert c == old
                assert client.get(f"/api/jobs/video-test/clips/{c['id']}/media?revision={old['revision']}").status_code == 200
            else:
                assert c["revision"] != old["revision"]
                assert client.get(f"/api/jobs/video-test/clips/{target}/media?revision={old['revision']}").status_code == 409
        assert len(list(folder.glob("clips-*"))) == 1


@pytest.mark.parametrize("failure", ["render", "publish"])
def test_single_regeneration_failure_preserves_downloads(tmp_path, media, monkeypatch, failure):
    from pathlib import Path

    app = create_app(tmp_path / "data")
    folder, _ = seed(app, media)
    with TestClient(app) as client:
        client.post("/api/jobs/video-test/clips")
        assert wait_clips(client)["clips_status"] == "completed"
        before = client.get("/api/jobs/video-test/clips").json()
        if failure == "render":
            def fail(*args):
                raise ValueError("Controlled render failure")
            monkeypatch.setattr("app.clips.render_clip", fail)
        else:
            actual = Path.replace
            def fail(self, destination):
                if self.name == "clips.tmp":
                    raise OSError("Controlled publication failure")
                return actual(self, destination)
            monkeypatch.setattr(Path, "replace", fail)
        assert client.post("/api/jobs/video-test/clips/highlight_001/regenerate").status_code == 202
        job = wait_clips(client)
        assert job["clips_status"] == "completed"
        assert job["clips_error"]
        assert client.get("/api/jobs/video-test/clips").json() == before
        for c in before["clips"]:
            assert client.get(f"/api/jobs/video-test/clips/{c['id']}/media?revision={c['revision']}").status_code == 200
        assert not list(folder.glob(".clips-*"))
        assert len(list(folder.glob("clips-*"))) == 1


def test_single_regeneration_validates_state_and_recovers_restart(tmp_path, media):
    app = create_app(tmp_path / "data")
    folder, _ = seed(app, media)
    with TestClient(app) as client:
        assert client.post("/api/jobs/video-test/clips/highlight_001/regenerate").status_code == 409
        client.post("/api/jobs/video-test/clips")
        assert wait_clips(client)["clips_status"] == "completed"
        before = client.get("/api/jobs/video-test/clips").json()
        assert client.post("/api/jobs/video-test/clips/unknown/regenerate").status_code == 404
    app.state.store.update("video-test", clips_status="running", clips_target_id="highlight_001")
    (folder / ".clips-interrupted").mkdir()
    (folder / ".clips-interrupted" / "partial.mp4").write_bytes(b"partial")
    with TestClient(app) as client:
        job = client.get("/api/jobs/video-test").json()
        assert job["clips_status"] == "completed"
        assert "interrupted" in job["clips_error"]
        assert client.get("/api/jobs/video-test/clips").json() == before
        assert not (folder / ".clips-interrupted").exists()
        assert client.post("/api/jobs/video-test/clips/highlight_001/regenerate").status_code == 202
        assert not wait_clips(client)["clips_error"]


def test_individual_regeneration_supports_legacy_batches_then_full_rebuild(tmp_path, media):
    app = create_app(tmp_path / "data")
    folder, _ = seed(app, media)
    with TestClient(app) as client:
        client.post("/api/jobs/video-test/clips")
        assert wait_clips(client)["clips_status"] == "completed"
        legacy = client.get("/api/jobs/video-test/clips").json()
        legacy["version"] = 2
        legacy.pop("subtitles", None)
        for c in legacy["clips"]:
            c.pop("revision", None)
            c.pop("captions", None)
        (folder / "clips.json").write_text(json.dumps(legacy))
        assert client.post("/api/jobs/video-test/clips/highlight_001/regenerate").status_code == 202
        assert not wait_clips(client)["clips_error"]
        batch = client.get("/api/jobs/video-test/clips").json()
        untouched = next(c for c in batch["clips"] if c["id"] == "highlight_002")
        assert untouched["revision"] == legacy["revision"]
        assert client.get(f"/api/jobs/video-test/clips/highlight_002/media?revision={legacy['revision']}").status_code == 200
        assert client.post("/api/jobs/video-test/clips", json={"mode": "center_crop"}).status_code == 202
        assert not wait_clips(client)["clips_error"]
        batch = client.get("/api/jobs/video-test/clips").json()
        assert all(c["revision"] == batch["revision"] for c in batch["clips"])
        assert all(c["framing"]["mode"] == "center_crop" for c in batch["clips"])
        assert len(list(folder.glob("clips-*"))) == 1


def seed_caption_words(folder):
    data = {"segments": [{"words": [
        {"word": "Gabin", "start": 1.4, "end": 1.8},
        {"word": "Baker.", "start": 1.8, "end": 2.3},
        {"word": "Nvidea", "start": 3.1, "end": 3.6},
    ]}]}
    (folder / "transcript.json").write_text(json.dumps(data))
    return (folder / "transcript.json").read_bytes()


def test_edit_spelling_save_regenerate_and_restore(tmp_path, media):
    app = create_app(tmp_path / "data")
    folder, _ = seed(app, media)
    original = seed_caption_words(folder)
    url = "/api/jobs/video-test/clips/highlight_001/subtitles"
    with TestClient(app) as client:
        client.post("/api/jobs/video-test/clips", json={"subtitles": True})
        assert not wait_clips(client)["clips_error"]
        before = client.get("/api/jobs/video-test/clips").json()
        editor = client.get(url).json()
        assert [w["text"] for w in editor["words"]] == ["Gabin", "Baker."]
        assert not editor["pending"]
        response = client.put(url, json={"revision": editor["revision"], "texts": ["Gavin", "Baker."]})
        assert response.status_code == 200
        saved = response.json()
        assert saved["pending"] and saved["has_edits"]
        assert saved["words"][0]["start"] == editor["words"][0]["start"]
        assert (folder / "transcript.json").read_bytes() == original
        assert client.get("/api/jobs/video-test/clips/highlight_002/subtitles").json()["words"][0]["text"] == "Nvidea"
        unchanged = client.get("/api/jobs/video-test/clips").json()
        assert unchanged["revision"] == before["revision"]
        assert unchanged["clips"][0]["subtitle_edits_pending"]
        # Optimistic revision prevents overwriting a change from another browser.
        assert client.put(url, json={"revision": editor["revision"], "texts": ["Wrong", "Baker."]}).status_code == 409
        assert client.post("/api/jobs/video-test/clips/highlight_001/regenerate").status_code == 202
        assert not wait_clips(client)["clips_error"]
        after = client.get("/api/jobs/video-test/clips").json()
        assert not after["clips"][0]["subtitle_edits_pending"]
        ass = (folder / after["directory"] / "highlight_001.ass").read_text()
        assert "Gavin" in ass and "Gabin" not in ass
        assert after["clips"][1]["revision"] == before["clips"][1]["revision"]
        assert not client.get(url).json()["pending"]
        # Restoring source spelling is itself a pending edit until re-rendered.
        response = client.put(url, json={"revision": saved["revision"], "texts": ["Gabin", "Baker."]})
        assert response.status_code == 200 and response.json()["pending"]
        assert not response.json()["has_edits"]
        client.post("/api/jobs/video-test/clips", json={"subtitles": True})
        assert not wait_clips(client)["clips_error"]
        assert not client.get(url).json()["pending"]
        assert (folder / "transcript.json").read_bytes() == original


def test_caption_editor_validation_and_failed_render_retains_edits(tmp_path, media, monkeypatch):
    app = create_app(tmp_path / "data")
    folder, _ = seed(app, media)
    seed_caption_words(folder)
    url = "/api/jobs/video-test/clips/highlight_001/subtitles"
    with TestClient(app) as client:
        client.post("/api/jobs/video-test/clips")
        assert wait_clips(client)["clips_status"] == "completed"
        assert client.get(url).status_code == 409  # Subtitles were not enabled.
        client.post("/api/jobs/video-test/clips", json={"subtitles": True})
        assert not wait_clips(client)["clips_error"]
        assert client.get("/api/jobs/video-test/clips/unknown/subtitles").status_code == 404
        editor = client.get(url).json()
        for texts in (["Gavin"], [" ", " "], ["x" * 121, "Baker."]):
            assert client.put(url, json={"revision": editor["revision"], "texts": texts}).status_code == 422
        app.state.store.update("video-test", clips_status="running", clips_target_id="highlight_001")
        assert client.put(url, json={"revision": editor["revision"], "texts": ["Gavin", "Baker."]}).status_code == 409
        app.state.store.update("video-test", clips_status="completed", clips_target_id=None)
        saved = client.put(url, json={"revision": editor["revision"], "texts": ["Gavin", "Baker."]}).json()
        before = client.get("/api/jobs/video-test/clips").json()
        def fail(*args):
            raise ValueError("Controlled caption render failure")
        monkeypatch.setattr("app.clips.render_clip", fail)
        client.post("/api/jobs/video-test/clips/highlight_001/regenerate")
        assert wait_clips(client)["clips_error"]
        assert client.get(url).json()["revision"] == saved["revision"]
        assert client.get(url).json()["pending"]
        assert client.get("/api/jobs/video-test/clips").json()["revision"] == before["revision"]
        assert client.get("/api/jobs/video-test/clips/highlight_001/media").status_code == 200


def test_deleted_word_is_saved_and_excluded_from_regenerated_subtitles(tmp_path, media):
    app = create_app(tmp_path / "data")
    folder, _ = seed(app, media)
    transcript = {"segments": [{"words": [
        {"word": "Nesca", "start": 1.4, "end": 1.7},
        {"word": "y", "start": 1.7, "end": 1.9},
        {"word": "Heiniks", "start": 1.9, "end": 2.3},
        {"word": "Other", "start": 3.1, "end": 3.6},
    ]}]}
    (folder / "transcript.json").write_text(json.dumps(transcript))
    original = (folder / "transcript.json").read_bytes()
    url = "/api/jobs/video-test/clips/highlight_001/subtitles"
    with TestClient(app) as client:
        client.post("/api/jobs/video-test/clips", json={"subtitles": True})
        assert not wait_clips(client)["clips_error"]
        before = client.get("/api/jobs/video-test/clips").json()
        editor = client.get(url).json()
        response = client.put(url, json={"revision": editor["revision"], "texts": ["SK", "", "Hynix"]})
        assert response.status_code == 200, response.text
        assert response.json()["words"][1]["text"] == ""
        client.post("/api/jobs/video-test/clips/highlight_001/regenerate")
        assert not wait_clips(client)["clips_error"]
        after = client.get("/api/jobs/video-test/clips").json()
        ass = (folder / after["directory"] / "highlight_001.ass").read_text()
        assert "SK" in ass and "Hynix" in ass
        assert "Nesca" not in ass and "Heiniks" not in ass
        assert after["clips"][0]["captions"]["word_count"] == 2
        assert after["clips"][1]["revision"] == before["clips"][1]["revision"]
        assert not client.get(url).json()["pending"]
        assert (folder / "transcript.json").read_bytes() == original
