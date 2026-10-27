import io
import json
import wave

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import OpenAI

from app.main import create_app
from app.store import Store
from app.transcription import transcribe_api


def configuration(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = tmp_path / "config.toml"
    config.write_text('[llm]\napi_key = "test-secret"\n')
    return config


def test_sdk_upload_chunks_and_word_offsets(tmp_path, monkeypatch):
    config = configuration(tmp_path, monkeypatch)
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as output:
        output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        output.writeframes(b"\0\0" * 40000)
    calls = []

    def respond(request):
        assert request.url.path == "/v1/audio/transcriptions"
        assert request.headers["authorization"] == "Bearer test-secret"
        body = request.read()
        for value in (b"whisper-1", b"verbose_json", b"word", b"segment", b"es"):
            assert value in body
        wav = body[body.index(b"RIFF") :]
        size = int.from_bytes(wav[4:8], "little") + 8
        with wave.open(io.BytesIO(wav[:size]), "rb") as chunk:
            calls.append(chunk.getnframes())
        return httpx.Response(
            200,
            json={
                "text": "Hola.",
                "language": "spanish",
                "duration": 1,
                "segments": [
                    {
                        "id": 0,
                        "start": 0.1,
                        "end": 0.4,
                        "text": "Hola.",
                        "avg_logprob": -0.2,
                        "no_speech_prob": 0.01,
                    }
                ],
                "words": [{"word": "Hola.", "start": 0.1, "end": 0.4}],
            },
        )

    monkeypatch.setattr("app.transcription.CHUNK_SECONDS", 1)
    monkeypatch.setattr(
        "app.transcription.OpenAI",
        lambda **kwargs: OpenAI(
            **kwargs,
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        ),
    )
    segments, info = transcribe_api(audio, "es", config)
    result = list(segments)
    assert calls == [16000, 16000, 8000]
    assert [segment.start for segment in result] == [0.1, 1.1, 2.1]
    assert result[-1].words[0].end == 2.4
    assert info.language == "spanish"
    assert info.language_probability is None
    assert list(tmp_path.glob("whisper-*")) == []


@pytest.mark.parametrize("status, expected", [(401, "rejected"), (429, "quota"), (500, "could not")])
def test_api_errors_are_safe(tmp_path, monkeypatch, status, expected):
    config = configuration(tmp_path, monkeypatch)
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as output:
        output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        output.writeframes(b"\0\0" * 160)
    monkeypatch.setattr(
        "app.transcription.OpenAI",
        lambda **kwargs: OpenAI(
            **(kwargs | {"max_retries": 0}),
            http_client=httpx.Client(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(status, json={"error": {"message": "test-secret"}}),
                )
            ),
        ),
    )
    segments, _ = transcribe_api(audio, "", config)
    with pytest.raises(ValueError) as error:
        list(segments)
    assert expected in str(error.value)
    assert "test-secret" not in str(error.value)
    assert list(tmp_path.glob("whisper-*")) == []


def test_missing_key_rejected_before_enqueue(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = tmp_path / "empty.toml"
    config.write_text("")
    with TestClient(create_app(tmp_path / "data", config)) as client:
        result = client.post(
            "/api/jobs/youtube",
            json={
                "url": "https://youtu.be/BaW_jenozKc",
                "model": "whisper-1",
            },
        )
        assert result.status_code == 422
        assert "api_key" in result.json()["detail"]
        assert client.get("/api/jobs").json() == []


def test_interrupted_api_job_is_not_automatically_rebilled(tmp_path):
    store = Store(tmp_path)
    store.save({"id": "remote", "status": "running", "model": "whisper-1"})
    with TestClient(create_app(tmp_path)) as client:
        job = client.get("/api/jobs/remote").json()
        assert job["status"] == "failed"
        assert "Retry" in job["error"]
        assert "test-secret" not in json.dumps(job)


def test_retry_can_switch_failed_local_job_to_api(tmp_path, monkeypatch):
    config = configuration(tmp_path, monkeypatch)
    app = create_app(tmp_path / "data", config)
    app.state.store.save({
        "id": "local", "status": "failed", "model": "base", "language": "es",
        "source_type": "upload", "source_file": "source.mp4",
    })
    calls = []
    monkeypatch.setattr(app.state.pipeline, "run", lambda job_id: calls.append(job_id))
    with TestClient(app) as client:
        response = client.post("/api/jobs/local/retry")
        assert response.status_code == 202
        job = response.json()
        assert job["model"] == "whisper-1"
        assert job["source_file"] == "source.mp4"
        assert job["language"] == "es"
        assert app.state.store.get("local")["model"] == "whisper-1"
    assert calls == ["local"]


def test_retry_switch_requires_key_and_leaves_failed_job_intact(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = tmp_path / "empty.toml"
    config.write_text("")
    app = create_app(tmp_path / "data", config)
    app.state.store.save({"id": "local", "status": "failed", "model": "base", "language": ""})
    with TestClient(app) as client:
        assert client.post("/api/jobs/local/retry").status_code == 422
        assert client.post("/api/jobs/local/retry", json={"model": "fake"}).status_code == 422
        assert app.state.store.get("local")["model"] == "base"
        assert app.state.store.get("local")["status"] == "failed"


def test_openai_is_default_for_all_sources_and_legacy_models_are_rejected(tmp_path, monkeypatch):
    config = configuration(tmp_path, monkeypatch)
    app = create_app(tmp_path / "data", config)
    calls = []
    monkeypatch.setattr(app.state.pipeline, "run", lambda job_id: calls.append(job_id))
    with TestClient(app) as client:
        assert client.get("/api/health").json()["models"] == ["whisper-1"]
        settings = client.get("/api/settings").json()
        assert settings["llm"] == {"configured": True, "model": "gpt-5.5", "provider": "OpenAI"}
        assert "test-secret" not in json.dumps(settings)
        response = client.post("/api/jobs/youtube", json={"url": "https://youtu.be/BaW_jenozKc"})
        assert response.status_code == 202
        assert response.json()["model"] == "whisper-1"
        response = client.post("/api/jobs/upload", files={"file": ("video.mp4", b"video", "video/mp4")})
        assert response.status_code == 202
        assert response.json()["model"] == "whisper-1"
        response = client.post("/api/uploads", json={"filename": "video.mp4", "size": 1})
        assert response.status_code == 201
        upload_id = response.json()["id"]
        assert client.put(f"/api/uploads/{upload_id}?offset=0", content=b"a").status_code == 200
        assert client.post(f"/api/uploads/{upload_id}/complete").json()["model"] == "whisper-1"
        assert client.post("/api/jobs/youtube", json={
            "url": "https://youtu.be/BaW_jenozKc", "model": "tiny",
        }).status_code == 422
        assert client.post("/api/uploads", json={
            "filename": "video.mp4", "size": 1, "model": "base",
        }).status_code == 422
        assert client.post("/api/jobs/upload", files={
            "file": ("video.mp4", b"a", "video/mp4"),
        }, data={"model": "base"}).status_code == 422
        assert client.post("/api/jobs/youtube", json={
            "url": "https://youtu.be/BaW_jenozKc", "language": "zz",
        }).status_code == 422
    assert len(calls) == 3
