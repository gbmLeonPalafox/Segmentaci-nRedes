import json
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import OpenAI

from app.config import HighlightOptions, load_settings
from app.highlights import (
    CandidateBatch,
    chunk_units,
    deduplicate,
    export_highlights,
    resolve_candidates,
    transcript_units,
)
from app.main import create_app


@pytest.fixture
def transcript():
    return {
        "title": "An interview",
        "language": "en",
        "segments": [
            {
                "id": f"seg_{i:05}",
                "start": i * 10,
                "end": (i + 1) * 10,
                "text": f"Original passage {i}.",
                "words": [],
            }
            for i in range(6)
        ],
    }


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    path = tmp_path / "config.toml"
    path.write_text('[llm]\napi_key = "test-secret-never-expose"\nmodel = "test-model"\n')
    return path


def wait_for_highlights(client, job_id):
    for _ in range(200):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job.get("highlights_status") in {"completed", "failed"}:
            return job
        time.sleep(0.01)
    pytest.fail("Selection did not complete")


def seed(app, transcript, job_id="test-video"):
    app.state.store.save(
        {
            "id": job_id,
            "title": transcript["title"],
            "status": "completed",
            "stage": "complete",
            "model": "tiny",
            "source_type": "upload",
        }
    )
    folder = app.state.store.root / job_id
    folder.mkdir(exist_ok=True)
    (folder / "transcript.json").write_text(json.dumps(transcript))


def batch(start="u000001", end="u000003", title="Key insight"):
    return CandidateBatch.model_validate(
        {
            "summary": "Interview overview.",
            "highlights": [
                {
                    "title": title,
                    "reason": "A complete, useful takeaway.",
                    "start_unit_id": start,
                    "end_unit_id": end,
                    "score": 92,
                }
            ],
        }
    )


def test_config_hot_reload_and_secret_redaction(configured, tmp_path):
    app = create_app(tmp_path / "data", config_path=configured)
    with TestClient(app) as client:
        response = client.get("/api/settings")
        assert response.json()["llm"]["configured"] is True
        assert "test-secret" not in response.text
        configured.write_text('[llm]\napi_key = ""\nmodel = "another-model"\n')
        response = client.get("/api/settings")
        assert response.json()["llm"] == {"configured": False, "model": "another-model", "provider": "OpenAI"}
        # Even malformed config must not echo source lines containing credentials.
        configured.write_text('[llm]\napi_key = "test-secret-never-expose" broken')
        response = client.get("/api/settings")
        assert response.json()["error"]
        assert "test-secret" not in response.text
        assert client.get("/config.toml").status_code == 404


def test_missing_key_preserves_transcript(tmp_path, monkeypatch, transcript):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    path = tmp_path / "empty.toml"
    path.write_text('[llm]\napi_key = ""')
    app = create_app(tmp_path / "data", config_path=path)
    seed(app, transcript)
    with TestClient(app) as client:
        assert client.post("/api/jobs/test-video/highlights").status_code == 422
        assert client.get("/api/jobs/test-video/transcript").status_code == 200
        assert client.get("/api/jobs/test-video").json()["status"] == "completed"


def test_invalid_ids_ranges_and_duration(transcript):
    units = transcript_units(transcript)
    options = HighlightOptions(min_seconds=20, max_seconds=40)
    for candidate in [
        batch(start="fake"),
        batch(start="u000004", end="u000001"),
        batch(end="u000001"),
        batch(end="u000006"),
    ]:
        selected, errors = resolve_candidates(candidate, units, options)
        assert not selected and errors
    selected, errors = resolve_candidates(batch(), units, options)
    assert not errors
    assert selected[0]["start"] == 0 and selected[0]["end"] == 30
    assert selected[0]["text"] == "Original passage 0. Original passage 1. Original passage 2."


def test_units_split_long_segments_at_word_boundaries():
    data = {
        "segments": [
            {
                "id": "seg_1",
                "start": 0,
                "end": 100,
                "text": "",
                "words": [
                    {"word": " Hello", "start": 0, "end": 1},
                    {"word": " there.", "start": 1, "end": 2},
                    {"word": " A", "start": 2.5, "end": 3},
                    {"word": " takeaway!", "start": 3, "end": 4},
                ],
            }
        ]
    }
    units = transcript_units(data)
    assert [(u["text"], u["start"], u["end"]) for u in units] == [
        ("Hello there.", 0, 2),
        ("A takeaway!", 2.5, 4),
    ]


def test_chunking_covers_all_units_with_overlap(transcript):
    units = transcript_units(transcript)
    chunks = chunk_units(units, 380, 20)
    assert len(chunks) > 1
    assert {u["id"] for chunk in chunks for u in chunk} == {u["id"] for u in units}
    assert {u["id"] for u in chunks[0]} & {u["id"] for u in chunks[1]}


def test_deduplicate_overlapping_passages():
    candidates = [{"start": 0, "end": 30}, {"start": 20, "end": 40}, {"start": 40, "end": 60}]
    assert deduplicate(candidates, 5) == [candidates[0], candidates[2]]


def test_real_sdk_request_and_pipeline_exports(configured, tmp_path, transcript, monkeypatch):
    captured = []

    def respond(request):
        payload = json.loads(request.content)
        captured.append(payload)
        assert payload["model"] == "test-model"
        assert payload["store"] is False
        assert payload["text"]["format"]["type"] == "json_schema"
        assert payload["text"]["format"]["strict"] is True
        assert "test-secret" not in json.dumps(payload)
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "created_at": 1,
                "object": "response",
                "status": "completed",
                "model": "test-model",
                "output": [
                    {
                        "id": "msg_1",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": batch(title="<script>test</script>").model_dump_json(),
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            },
        )

    def client_factory(**kwargs):
        return OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond)))

    monkeypatch.setattr("openai.OpenAI", client_factory)
    app = create_app(tmp_path / "data", config_path=configured)
    seed(app, transcript)
    with TestClient(app) as client:
        response = client.post("/api/jobs/test-video/highlights", json={"min_seconds": 20, "max_seconds": 40})
        assert response.status_code == 202
        job = wait_for_highlights(client, "test-video")
        assert job["highlights_status"] == "completed", job
        assert job["status"] == "completed"
        result = client.get("/api/jobs/test-video/highlights").json()
        assert (
            result["highlights"][0]["text"] == "Original passage 0. Original passage 1. Original passage 2."
        )
        assert result["usage"] == {"input_tokens": 100, "output_tokens": 50, "requests": 1}
        assert "test-secret" not in json.dumps(result)
        assert result["transcript_sha256"]
        exported = client.get("/api/jobs/test-video/highlights/export/html")
        assert exported.status_code == 200
        assert "&lt;script&gt;test&lt;/script&gt;" in exported.text
        assert "<script>test</script>" not in exported.text
        assert client.get("/api/jobs/test-video/highlights/export/json").json() == result
        assert client.get("/api/jobs/test-video/highlights/export/txt").status_code == 404
        assert len(captured) == 1


def test_failed_llm_call_redacts_key_and_keeps_transcript(configured, tmp_path, transcript, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("Provider error containing test-secret-never-expose")

    monkeypatch.setattr("app.highlights.ask_model", fail)
    app = create_app(tmp_path / "data", config_path=configured)
    seed(app, transcript)
    with TestClient(app) as client:
        assert client.post("/api/jobs/test-video/highlights").status_code == 202
        job = wait_for_highlights(client, "test-video")
        assert job["highlights_status"] == "failed"
        assert "test-secret" not in json.dumps(job)
        assert client.get("/api/jobs/test-video/transcript").status_code == 200


def test_semantic_retry_repairs_unknown_ids(configured, tmp_path, transcript, monkeypatch):
    replies = iter([batch(start="invented"), batch()])
    calls = []

    def reply(client, model, instructions, payload, schema):
        calls.append(payload)
        return next(replies), None

    monkeypatch.setattr("app.highlights.ask_model", reply)
    app = create_app(tmp_path / "data", config_path=configured)
    seed(app, transcript)
    with TestClient(app) as client:
        assert client.post("/api/jobs/test-video/highlights").status_code == 202
        assert wait_for_highlights(client, "test-video")["highlights_status"] == "completed"
        assert "correction" in calls[1]
        assert len(calls) == 2


def test_restart_does_not_silently_repeat_paid_requests(configured, tmp_path, transcript):
    app = create_app(tmp_path / "data", config_path=configured)
    seed(app, transcript)
    app.state.store.update("test-video", highlights_status="running")
    with TestClient(app) as client:
        job = client.get("/api/jobs/test-video").json()
        assert job["highlights_status"] == "failed"
        assert job["status"] == "completed"
        assert "restarted" in job["highlights_error"]


def test_config_env_override_is_not_serialized(configured, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")
    settings = load_settings(configured)
    assert settings.llm.api_key == "environment-secret"
    assert "environment-secret" not in repr(settings)


def test_html_escape_all_fields():
    data = {"title": "<script>", "model": "<model>", "summary": "<summary>", "highlights": []}
    content = export_highlights(data, "html")
    assert "<script>" not in content and "&lt;summary&gt;" in content


def test_long_transcript_global_ranking(configured, tmp_path, monkeypatch):
    from app.highlights import Ranking

    data = {
        "title": "Long interview",
        "language": "en",
        "segments": [
            {
                "id": f"seg{i}",
                "start": i * 10,
                "end": (i + 1) * 10,
                "text": f"Passage {i}: " + "useful context " * 20,
                "words": [],
            }
            for i in range(45)
        ],
    }
    configured.write_text('[llm]\napi_key="test-secret"\nmax_input_chars=4000\n')
    calls = []

    def reply(client, model, instructions, payload, schema):
        calls.append(schema)
        if schema is CandidateBatch:
            units = payload["units"]
            return batch(units[0]["id"], units[2]["id"]), None
        candidates = payload["candidates"]
        return Ranking(
            summary="A global overview.", candidate_ids=[c["candidate_id"] for c in reversed(candidates)][:2]
        ), None

    monkeypatch.setattr("app.highlights.ask_model", reply)
    app = create_app(tmp_path / "data", config_path=configured)
    seed(app, data)
    with TestClient(app) as client:
        assert client.post("/api/jobs/test-video/highlights").status_code == 202
        job = wait_for_highlights(client, "test-video")
        assert job["highlights_status"] == "completed", job
        result = client.get("/api/jobs/test-video/highlights").json()
        assert calls[-1] is Ranking and calls.count(CandidateBatch) > 1
        assert result["summary"] == "A global overview."
        assert len(result["highlights"]) == 2
        assert result["highlights"][0]["start"] > result["highlights"][1]["start"]


def test_invalid_duration_and_empty_transcript(configured, tmp_path, transcript):
    app = create_app(tmp_path / "data", config_path=configured)
    seed(app, transcript)
    with TestClient(app) as client:
        assert (
            client.post(
                "/api/jobs/test-video/highlights", json={"min_seconds": 90, "max_seconds": 20}
            ).status_code
            == 422
        )
        app.state.store.update("test-video", status="running")
        assert client.post("/api/jobs/test-video/highlights").status_code == 409
        app.state.store.update("test-video", status="completed", highlights_status="running")
        assert client.post("/api/jobs/test-video/highlights").status_code == 409
        app.state.store.update("test-video", highlights_status="failed")
        path = app.state.store.root / "test-video" / "transcript.json"
        path.write_text(json.dumps({**transcript, "segments": []}))
        assert client.post("/api/jobs/test-video/highlights").status_code == 422


def test_repeated_invalid_selection_fails_without_fake_results(configured, tmp_path, transcript, monkeypatch):
    calls = []

    def invalid(*args):
        calls.append(1)
        return batch(start="hallucinated"), SimpleNamespace(input_tokens=1, output_tokens=1)

    monkeypatch.setattr("app.highlights.ask_model", invalid)
    app = create_app(tmp_path / "data", config_path=configured)
    seed(app, transcript)
    with TestClient(app) as client:
        assert client.post("/api/jobs/test-video/highlights").status_code == 202
        job = wait_for_highlights(client, "test-video")
        assert job["highlights_status"] == "failed"
        assert client.get("/api/jobs/test-video/highlights").status_code == 409
        assert len(calls) == 2


def test_empty_selection_is_valid(configured, tmp_path, transcript, monkeypatch):
    monkeypatch.setattr(
        "app.highlights.ask_model",
        lambda *args: (CandidateBatch(summary="No complete takeaways.", highlights=[]), None),
    )
    app = create_app(tmp_path / "data", config_path=configured)
    seed(app, transcript)
    with TestClient(app) as client:
        assert client.post("/api/jobs/test-video/highlights").status_code == 202
        job = wait_for_highlights(client, "test-video")
        assert job["highlights_status"] == "completed" and job["highlights_count"] == 0
        assert client.get("/api/jobs/test-video/highlights").json()["highlights"] == []


def test_sentence_boundary_preserves_negation_across_whisper_segments():
    data = {
        "segments": [
            {
                "id": "a",
                "words": [{"word": " No", "start": 0, "end": 1}, {"word": " porque", "start": 1, "end": 2}],
            },
            {
                "id": "b",
                "words": [
                    {"word": " puedas", "start": 2, "end": 3},
                    {"word": " delegarlo.", "start": 3, "end": 4},
                ],
            },
        ]
    }
    units = transcript_units(data)
    assert len(units) == 1
    assert units[0]["text"] == "No porque puedas delegarlo."
    assert units[0]["segment_ids"] == ["a", "b"]
    selected, errors = resolve_candidates(
        batch(end="u000001"), units, HighlightOptions(min_seconds=1, max_seconds=10)
    )
    assert not errors
    assert selected[0]["segment_ids"] == ["a", "b"]
    assert selected[0]["start"] == 0 and selected[0]["end"] == 4


def test_long_sentence_is_not_cut_at_arbitrary_word_count():
    words = [{"word": " word", "start": i, "end": i + 1} for i in range(50)]
    words[-1]["word"] += "."
    units = transcript_units({"segments": [{"id": "a", "words": words}]})
    assert len(units) == 1 and units[0]["end"] == 50
