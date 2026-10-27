import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.uploads import CHUNK_BYTES


@pytest.fixture
def upload_client(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = tmp_path / "config.toml"
    config.write_text('[llm]\napi_key = "test-secret"\n')
    app = create_app(tmp_path / "data", config)
    calls = []
    monkeypatch.setattr(app.state.pipeline, "run", lambda job_id: calls.append(job_id))
    with TestClient(app) as client:
        yield client, app, calls


def begin(client, size, filename="video.mp4"):
    return client.post("/api/uploads", json={"filename": filename, "size": size, "model": "whisper-1"})


def test_large_upload_limits_are_enforced_before_receiving_bytes(upload_client):
    client, app, _ = upload_client
    assert client.get("/api/health").json()["max_upload_bytes"] == 5 * 1024**3
    assert begin(client, 3 * 1024**3).status_code == 201
    assert begin(client, 5 * 1024**3).status_code == 201
    assert begin(client, 5 * 1024**3 + 1).status_code == 413
    assert begin(client, 0).status_code == 422
    assert begin(client, 100, "invalid.exe").status_code == 422
    assert app.state.store.all() == []


def test_retry_part_and_completion_do_not_duplicate_data_or_work(upload_client):
    client, app, calls = upload_client
    session = begin(client, 9).json()
    url = f"/api/uploads/{session['id']}"
    assert client.put(url + "?offset=0", content=b"hello").json() == {"offset": 5}
    assert client.put(url + "?offset=0", content=b"hello").json() == {"offset": 5}
    assert client.put(url + "?offset=0", content=b"other").status_code == 409
    assert client.post(url + "/complete").status_code == 409
    assert app.state.store.all() == []
    assert client.put(url + "?offset=6", content=b"!").status_code == 409
    assert client.put(url + "?offset=5", content=b" !!!").json() == {"offset": 9}
    response = client.post(url + "/complete")
    assert response.status_code == 202
    job = response.json()
    assert (app.state.store.root / job["id"] / job["source_file"]).read_bytes() == b"hello !!!"
    assert client.post(url + "/complete").json()["id"] == job["id"]
    assert client.delete(url).status_code == 200
    assert (app.state.store.root / job["id"] / job["source_file"]).exists()
    assert client.put(url + "?offset=9", content=b"x").status_code == 413
    assert calls == [job["id"]]


def test_chunk_size_empty_overrun_and_cancellation(upload_client):
    client, app, _ = upload_client
    session = begin(client, 100).json()
    url = f"/api/uploads/{session['id']}"
    assert client.put(url + "?offset=0", content=b"").status_code == 413
    assert client.put(url + "?offset=0", content=b"x" * 101).status_code == 413
    assert client.put(url + "?offset=0", content=b"x" * (CHUNK_BYTES + 1)).status_code == 413
    assert client.delete(url).status_code == 200
    assert not (app.state.store.root / ".uploads" / session["id"]).exists()
    assert client.put(url + "?offset=0", content=b"a").status_code == 404
    assert client.post("/api/uploads/not-an-id/complete").status_code == 404
