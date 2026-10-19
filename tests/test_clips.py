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
            rgb = next(video.decode(video=0)).to_ndarray(format="rgb24").mean(axis=(0, 1))
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
