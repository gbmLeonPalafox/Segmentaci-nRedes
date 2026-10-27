import json
import subprocess
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.pipeline import Pipeline, export_transcript, ffmpeg_path, inspect_media, youtube_url


def wait_for(client, job_id):
    for _ in range(200):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"completed", "failed"}:
            return job
        time.sleep(0.025)
    pytest.fail("Job did not finish")


@pytest.fixture
def video(tmp_path):
    target = tmp_path / "fixture.mp4"
    subprocess.run(
        [
            ffmpeg_path(),
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(target),
        ],
        check=True,
        capture_output=True,
    )
    return target


@pytest.fixture
def client(tmp_path, monkeypatch):
    segment = SimpleNamespace(
        start=0.1,
        end=0.9,
        text=" Hello <world>.",
        avg_logprob=-0.2,
        no_speech_prob=0.01,
        words=[SimpleNamespace(start=0.1, end=0.9, word=" Hello", probability=0.99)],
    )

    def transcribe(path, language, config_path):
        assert language == "en"
        assert path.name == "audio.wav"
        return iter([segment]), SimpleNamespace(language="en", language_probability=None)

    monkeypatch.setattr("app.transcription.transcribe_api", transcribe)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = tmp_path / "config.toml"
    config.write_text('[llm]\napi_key = "test-secret"\n')
    app = create_app(tmp_path / "app-data", config)
    with TestClient(app) as session:
        yield session


@pytest.mark.parametrize(
    "url",
    [
        "https://youtu.be/BaW_jenozKc?t=3",
        "https://www.youtube.com/watch?v=BaW_jenozKc&list=123",
        "https://youtube.com/shorts/BaW_jenozKc",
        "https://m.youtube.com/watch?v=BaW_jenozKc",
    ],
)
def test_youtube_normalization(url):
    assert youtube_url(url) == "https://www.youtube.com/watch?v=BaW_jenozKc"


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/video",
        "https://youtube.com.evil.com/watch?v=BaW_jenozKc",
        "file:///tmp/video.mp4",
        "https://youtube.com/playlist?list=123",
        "https://user:secret@youtube.com/watch?v=BaW_jenozKc",
    ],
)
def test_reject_non_video_urls(url):
    with pytest.raises(ValueError):
        youtube_url(url)


def test_complete_upload_transcript_and_exports(client, video):
    result = client.post(
        "/api/jobs/upload",
        files={"file": ("interview.mp4", video.read_bytes(), "video/mp4")},
        data={"model": "whisper-1", "language": "en"},
    )
    assert result.status_code == 202
    job_id = result.json()["id"]
    job = wait_for(client, job_id)
    assert job["status"] == "completed", job
    assert job["metadata"]["width"] == 320
    transcript = client.get(f"/api/jobs/{job_id}/transcript").json()
    assert transcript["segments"][0]["words"][0]["probability"] == 0.99
    assert transcript["segments"][0]["id"] == "seg_00001"
    assert "Hello <world>" in transcript["text"]
    for kind in ["json", "txt", "srt", "html"]:
        export = client.get(f"/api/jobs/{job_id}/export/{kind}")
        assert export.status_code == 200
        assert "attachment" in export.headers["content-disposition"]
    report = client.get(f"/api/jobs/{job_id}/export/html").text
    assert "&lt;world&gt;" in report
    assert "Hello <world>" not in report
    srt = client.get(f"/api/jobs/{job_id}/export/srt").text
    assert "00:00:00,100 --> 00:00:00,900" in srt
    media = client.get(f"/api/jobs/{job_id}/media", headers={"Range": "bytes=0-99"})
    assert media.status_code == 206
    assert len(media.content) == 100
    assert client.post(f"/api/jobs/{job_id}/retry").status_code == 409


def test_invalid_input(client):
    assert client.post("/api/jobs/youtube", json={"url": "https://example.com"}).status_code == 422
    assert (
        client.post(
            "/api/jobs/youtube", json={"url": "https://youtu.be/BaW_jenozKc", "model": "fake"}
        ).status_code
        == 422
    )
    assert client.post("/api/jobs/upload", files={"file": ("empty.mp4", b"", "video/mp4")}).status_code == 422
    assert (
        client.post("/api/jobs/upload", files={"file": ("script.py", b"print(1)", "text/plain")}).status_code
        == 422
    )
    assert client.get("/api/jobs/missing/transcript").status_code == 404
    assert client.get("/api/jobs/missing").status_code == 404
    assert client.get("/").status_code == 200


def test_bad_media_fails_and_can_retry(client):
    result = client.post("/api/jobs/upload", files={"file": ("broken.mp4", b"not a video", "video/mp4")})
    job_id = result.json()["id"]
    job = wait_for(client, job_id)
    assert job["status"] == "failed"
    assert job["error"]
    assert client.get(f"/api/jobs/{job_id}/transcript").status_code == 409
    assert client.post(f"/api/jobs/{job_id}/retry").status_code == 202
    assert wait_for(client, job_id)["status"] == "failed"


def test_youtube_job_uses_shared_pipeline(client, video, monkeypatch):
    def download(self, job, folder):
        assert job["source_url"] == "https://www.youtube.com/watch?v=BaW_jenozKc"
        target = folder / "source.mp4"
        target.write_bytes(video.read_bytes())
        self.store.update(job["id"], title="Downloaded video", channel="Example channel")
        return target

    monkeypatch.setattr(Pipeline, "download", download)
    result = client.post(
        "/api/jobs/youtube", json={"url": "https://youtu.be/BaW_jenozKc?t=4", "language": "en"}
    )
    job = wait_for(client, result.json()["id"])
    assert job["status"] == "completed", job
    assert job["title"] == "Downloaded video"
    assert job["channel"] == "Example channel"


def test_restart_requires_explicit_retry_for_legacy_pending_jobs(tmp_path, monkeypatch):
    from app.store import Store

    store = Store(tmp_path)
    store.save({"id": "pending", "status": "running", "stage": "transcribe", "model": "base"})
    calls = []
    monkeypatch.setattr(Pipeline, "run", lambda self, job_id: calls.append(job_id))
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/jobs/pending").json()["status"] == "failed"
        assert calls == []


def test_html_escapes_title_and_json_roundtrip():
    data = {"title": "<script>alert(1)</script>", "language": "en", "segments": []}
    assert "<script>" not in export_transcript(data, "html")
    assert json.loads(export_transcript(data, "json")) == data


def test_probe_dimensions(video):
    metadata = inspect_media(video)
    assert metadata["width"] == 320 and metadata["height"] == 240
    assert 0.9 < metadata["duration"] < 1.2


def test_mkv_preview_preserves_dimensions(video, tmp_path):
    from app.pipeline import make_preview

    mkv = tmp_path / "source.mkv"
    subprocess.run(
        [ffmpeg_path(), "-y", "-i", str(video), "-c", "copy", str(mkv)], check=True, capture_output=True
    )
    metadata = inspect_media(mkv)
    preview = make_preview(mkv, metadata)
    assert preview.name == "preview.mp4"
    result = inspect_media(preview)
    assert (result["width"], result["height"]) == (320, 240)
    assert result["video_codec"] == "h264"
    assert result["audio_codec"] == "aac"


def test_video_without_audio_is_rejected(tmp_path):
    target = tmp_path / "silent.mp4"
    subprocess.run(
        [ffmpeg_path(), "-y", "-f", "lavfi", "-i", "color=s=320x240:d=1", "-c:v", "libx264", str(target)],
        check=True,
        capture_output=True,
    )
    with pytest.raises(ValueError, match="audio track"):
        inspect_media(target)


def test_api_transcription_preserves_exports_and_timeline(tmp_path, monkeypatch, video):
    import httpx
    from openai import OpenAI

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = tmp_path / "api.toml"
    config.write_text('[llm]\napi_key = "test-api-secret"\n')
    def respond(request):
        return httpx.Response(200, json={
            "text": "Hola mundo.", "language": "spanish", "duration": 1,
            "segments": [{"id": 0, "start": 0.1, "end": 0.9, "text": "Hola mundo.",
                          "avg_logprob": -0.1, "no_speech_prob": 0.01}],
            "words": [{"word": "Hola", "start": 0.1, "end": 0.4},
                      {"word": "mundo.", "start": 0.4, "end": 0.9}],
        })
    monkeypatch.setattr("app.transcription.OpenAI", lambda **kwargs: OpenAI(
        **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ))
    with TestClient(create_app(tmp_path / "remote-data", config)) as client:
        response = client.post("/api/jobs/upload", files={
            "file": ("video.mp4", video.read_bytes(), "video/mp4"),
        }, data={"model": "whisper-1", "language": "es"})
        assert response.status_code == 202
        job_id = response.json()["id"]
        job = wait_for(client, job_id)
        assert job["status"] == "completed", job
        transcript = client.get(f"/api/jobs/{job_id}/transcript").json()
        assert transcript["provider"] == "openai"
        assert transcript["model"] == "whisper-1"
        assert transcript["segments"][0]["words"][1]["word"] == "mundo."
        assert "Hola mundo." in client.get(f"/api/jobs/{job_id}/export/srt").text
        assert not (tmp_path / "remote-data" / job_id / "audio.wav").exists()
        assert "test-api-secret" not in json.dumps(job)


def test_chunked_video_upload_processes_only_after_completion(client, video):
    data = video.read_bytes()
    session = client.post("/api/uploads", json={
        "filename": "interview.mp4", "size": len(data), "model": "whisper-1", "language": "en",
    }).json()
    url = f"/api/uploads/{session['id']}"
    split = len(data) // 2
    assert client.put(url + "?offset=0", content=data[:split]).status_code == 200
    assert client.get(f"/api/jobs/{session['id']}").status_code == 404
    assert client.put(url + f"?offset={split}", content=data[split:]).status_code == 200
    response = client.post(url + "/complete")
    assert response.status_code == 202
    job = wait_for(client, session["id"])
    assert job["status"] == "completed", job
    assert client.get(f"/api/jobs/{session['id']}/export/srt").status_code == 200
