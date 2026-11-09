import json
import time

import pytest
from fastapi.testclient import TestClient

from app.clips import highlight_hash
from app.learning import GOALS, AdaptedPrompt, LearningStore
from app.main import create_app
from app.prompts import PromptManager
from app.store import Store


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = tmp_path / "config.toml"
    config.write_text('[llm]\napi_key="private-test-key"\n')
    app = create_app(tmp_path / "data", config)
    app.state.store.save({"id": "video", "title": "Example", "status": "completed"})
    folder = app.state.store.root / "video"
    folder.mkdir()
    (folder / "transcript.json").write_text(json.dumps({
        "title": "Example", "language": "es", "segments": [
            {"id": f"s{i}", "start": i * 10, "end": (i + 1) * 10, "text": f"Idea completa {i}."}
            for i in range(6)
        ],
    }))
    return app


def request(**changes):
    return {"area": "acquisition", "audience": "Docentes de primaria con poco tiempo",
            "min_seconds": 10, "max_seconds": 40, **changes}


def seed_highlights(app):
    data = {"job_id": "video", "created_at": "2026-11-09T12:00:00", "options": {
        "audience_context": "Docentes de primaria con poco tiempo", "objective": "Find useful passages.",
    }, "prompt": {"id": "acquisition", "version": 1, "area": "acquisition"},
        "highlights": [{"id": "highlight_001", "title": "Useful example", "text": "Source passage.",
                        "start": 0, "end": 30, "duration": 30}]}
    (app.state.store.root / "video" / "highlights.json").write_text(json.dumps(data))
    app.state.store.update("video", highlights_status="completed")
    LearningStore(app.state.store).selections(data)
    return data


def publication(data, **changes):
    return {"selection_revision": highlight_hash(data), "area": "acquisition",
            "audience": "Docentes de primaria con poco tiempo", "channel": "Instagram",
            "published_at": "2026-11-09", "evaluation": "successful", "stage": "evaluated",
            "metrics": {"inquiries": 12}, "evidence_source": "measured", "feedback": "Buen ejemplo práctico.",
            **changes}


def test_publications_are_separate_from_selections_and_exportable(app):
    data = seed_highlights(app)
    with TestClient(app) as client:
        initial = client.get('/api/jobs/video/results').json()
        assert initial['selection_revision'] == highlight_hash(data)
        assert initial['records'][0]['stage'] == 'selected'
        assert LearningStore(app.state.store).evidence('acquisition', request()['audience']) == []
        for channel in ('Instagram', 'TikTok'):
            response = client.post('/api/jobs/video/clips/highlight_001/results', json=publication(data, channel=channel))
            assert response.status_code == 201, response.text
            assert response.json()['options'] == data['options']
        exported = client.get('/api/jobs/video/results/export').json()
        assert len(exported['records']) == 3
        assert {r.get('channel') for r in exported['records']} == {None, 'Instagram', 'TikTok'}
        changed = dict(data, created_at='2026-11-10T12:00:00')
        (app.state.store.root / 'video' / 'highlights.json').write_text(json.dumps(changed))
        assert client.post('/api/jobs/video/clips/highlight_001/results', json=publication(data)).status_code == 409
        retained = client.get('/api/jobs/video/results').json()['records']
        assert len(retained) == 4  # Two selections and two original publication records.
        assert len([r for r in retained if r['stage'] == 'evaluated']) == 2


def test_feedback_validation_and_unknown_clip(app):
    data = seed_highlights(app)
    with TestClient(app) as client:
        assert client.post('/api/jobs/video/clips/unknown/results', json=publication(data)).status_code == 404
        for changes in ({'metrics': {'views': -1}}, {'metrics': {}},
                        {'stage': 'published', 'evaluation': 'successful'}, {'audience': '   '}):
            assert client.post('/api/jobs/video/clips/highlight_001/results', json=publication(data, **changes)).status_code == 422


def test_pending_publication_can_be_evaluated_without_duplicating_evidence(app):
    data = seed_highlights(app)
    with TestClient(app) as client:
        published = publication(data, stage="published", evaluation="pending")
        record = client.post('/api/jobs/video/clips/highlight_001/results', json=published).json()
        learning = LearningStore(app.state.store)
        assert learning.evidence('acquisition', request()['audience']) == []
        updated = client.put(f"/api/jobs/video/clips/highlight_001/results/{record['id']}",
                             json=publication(data)).json()
        assert updated['id'] == record['id']
        assert updated['history'][0]['stage'] == 'published'
        assert len(learning.evidence('acquisition', request()['audience'])) == 1
        assert len(client.get('/api/jobs/video/results').json()['records']) == 2


def test_evidence_filters_goal_and_labels_indirect_and_synthetic(tmp_path):
    learning = LearningStore(Store(tmp_path))
    base = {"job_id": "video", "clip_id": "h1", "stage": "evaluated", "area": "acquisition",
            "goal": GOALS['acquisition'], "audience": request()['audience'], "channel": "Instagram",
            "published_at": "2026-11-09", "evaluation": "successful", "metrics": {"inquiries": 4},
            "feedback": "A practical example", "evidence_source": "measured", "clip": {"title": "Example"}}
    real = learning.add_result(base)
    negative = learning.add_result(base | {'evaluation': 'unsuccessful', 'metrics': {'inquiries': 0}})
    synthetic = learning.add_result(base | {'evidence_source': 'synthetic'})
    similar = learning.add_result(base | {'audience': 'Docentes de primaria que buscan ejemplos prácticos'})
    for updates in ({'area': 'activation', 'goal': GOALS['activation']}, {'stage': 'selected'},
                    {'stage': 'published'}, {'audience': 'Ingenieros aeroespaciales'}, {'evaluation': 'pending'}):
        learning.add_result(base | updates)
    evidence = learning.evidence('acquisition', 'DOCENTES de primaria con poco tiempo', 'Instagram')
    assert {r['id'] for r in evidence} == {real['id'], negative['id'], synthetic['id'], similar['id']}
    assert evidence[-1]['relevance'] == 'similar_audience'
    assert next(r for r in evidence if r['id'] == synthetic['id'])['evidence_label'] == 'synthetic_hypothesis'
    assert len(learning.evidence('acquisition', request()['audience'], limit=2)) == 2
    assert {r['evaluation'] for r in learning.evidence('acquisition', request()['audience'], limit=2)} == {'successful', 'unsuccessful'}


def test_adaptation_uses_evidence_requires_review_and_keeps_original_draft(app, monkeypatch):
    data = seed_highlights(app)
    calls = []

    def reply(client, model, instructions, payload, schema):
        calls.append((schema, payload))
        if schema is AdaptedPrompt:
            assert payload['evidence'][0]['evaluation'] == 'successful'
            assert 'no confiables' in instructions
            return AdaptedPrompt(objective='Selecciona ejemplos claros para docentes con poco tiempo.',
                                 changes=['Prioriza ejemplos.'], assumptions=['Resultados reportados, sin causalidad.']), None
        return schema(summary='Overview', highlights=[{
            'title': 'Example', 'reason': 'Practical.', 'score': 90,
            'start_unit_id': 'u000001', 'end_unit_id': 'u000003',
        }]), None

    monkeypatch.setattr('app.highlights.ask_model', reply)
    with TestClient(app) as client:
        client.post('/api/jobs/video/clips/highlight_001/results', json=publication(data))
        draft = client.post('/api/personalizations', json=request()).json()
        assert draft['reviewed_at'] is None
        options = {'prompt_id': 'acquisition', 'personalization_id': draft['id'],
                   'audience_context': request()['audience'], 'min_seconds': 10, 'max_seconds': 40}
        assert client.post('/api/jobs/video/highlights', json=options).status_code == 422
        edited = 'Selecciona explicaciones breves con ejemplos concretos para docentes.'
        approved = client.post(f"/api/personalizations/{draft['id']}/approve", json={'objective': edited})
        assert approved.status_code == 200
        assert approved.json()['draft_objective'] == draft['objective']
        assert client.post(f"/api/personalizations/{draft['id']}/approve", json={'objective': edited}).status_code == 409
        assert client.post('/api/jobs/video/highlights', json=options | {'audience_context': 'Otra audiencia'}).status_code == 422
        assert client.post('/api/jobs/video/highlights', json=options).status_code == 202
        for _ in range(200):
            job = client.get('/api/jobs/video').json()
            if job['highlights_status'] in ('completed', 'failed'):
                break
            time.sleep(.01)
        assert job['highlights_status'] == 'completed', job
        saved = client.get('/api/jobs/video/highlights').json()
        assert saved['options']['objective'] == edited
        assert saved['prompt']['personalization']['evidence'][0]['id']
        assert calls[-1][1]['objective'] == edited
        assert any(r['stage'] == 'selected' for r in client.get('/api/jobs/video/results').json()['records'])


def test_manual_versions_persist_and_honor_master_version(app):
    with TestClient(app) as client:
        master = next(p for p in client.get('/api/prompts').json() if p['id'] == 'activation')
        response = client.post('/api/prompts/activation/revisions', json={
            'base_version': master['revisions'][-1]['version'],
            'objective': 'A new master with different instructions.', 'min_seconds': 20, 'max_seconds': 60,
        })
        assert response.status_code == 201
        body = request(area='activation', master_version=1, objective='A reviewed prompt for existing clients.')
        first = client.post('/api/personalizations/manual', json=body).json()
        second = client.post('/api/personalizations/manual', json=body).json()
        assert first['version'] == 1 and second['version'] == 2
        assert first['master']['objective'] == master['revisions'][0]['objective']
        assert first['reviewed_at']
        assert len(client.get('/api/personalizations').json()) == 2
        assert client.post('/api/personalizations/manual', json=body | {'master_version': 999}).status_code == 422
    assert len(LearningStore(app.state.store).variants()) == 2


def test_existing_catalog_gains_masters_without_losing_edits(tmp_path):
    manager = PromptManager(tmp_path)
    old = [p for p in manager.catalog() if p['status'] != 'master']
    old[0]['revisions'][0]['objective'] = 'Keep this historical custom objective.'
    manager.path.write_text(json.dumps(old))
    catalog = manager.catalog()
    assert catalog[0]['revisions'][0]['objective'] == old[0]['revisions'][0]['objective']
    assert {p['id'] for p in catalog if p['status'] == 'master'} == {'acquisition', 'activation'}


def test_provider_error_is_redacted(app, monkeypatch):
    def fail(*args):
        raise RuntimeError('private-test-key')
    monkeypatch.setattr('app.highlights.ask_model', fail)
    with TestClient(app) as client:
        response = client.post('/api/personalizations', json=request())
        assert response.status_code == 422
        assert 'private-test-key' not in response.text
        assert client.get('/api/personalizations').json() == []


def test_gbm_master_installation_preserves_history_and_is_idempotent(tmp_path):
    from app.prompts import MASTER_RELEASE, TEMPLATES, PromptRevision

    manager = PromptManager(tmp_path)
    old = manager.catalog()
    for profile in old:
        if profile['status'] == 'master':
            profile['revisions'] = [profile['revisions'][0]]
    manager.path.write_text(json.dumps(old))
    manager.install_masters()
    installed = manager.catalog()
    for profile in installed:
        if profile['status'] == 'master':
            assert len(profile['revisions']) == 2
            latest = profile['revisions'][-1]
            assert latest['release'] == MASTER_RELEASE
            assert latest['objective'] == (TEMPLATES / f"{profile['id']}.md").read_text().strip()
            assert len(latest['objective']) > 2000
    before = manager.path.read_text()
    manager.install_masters()
    assert manager.path.read_text() == before
    manager.publish('activation', PromptRevision(base_version=2, objective='A later reviewed custom revision.',
                                                min_seconds=20, max_seconds=70))
    manager.install_masters()
    assert next(p for p in manager.catalog() if p['id'] == 'activation')['revisions'][-1]['version'] == 3


@pytest.mark.parametrize('area', ['acquisition', 'activation'])
def test_gbm_briefs_change_selection_criteria_but_preserve_output_contract(app, monkeypatch, area):
    from app.highlights import CandidateBatch

    calls = []
    def reply(client, model, instructions, payload, schema):
        calls.append(payload)
        assert schema is CandidateBatch
        assert 'GBM' in payload['master_brief']
        assert '{{TRANSCRIPCION_WHISPER}}' not in payload['objective']
        assert '## Entrega' not in payload['objective']
        assert 'Exclude return promises' in instructions
        assert 'Do not generate opening overlays' in instructions
        assert payload['campaign_context'] == 'Definir objetivos antes de decidir.'
        return schema(summary='Una idea completa.', highlights=[{
            'title': 'Un ejemplo útil', 'reason': 'Aclara una inquietud sin promoción.', 'score': 90,
            'start_unit_id': 'u000001', 'end_unit_id': 'u000003',
        }]), None
    monkeypatch.setattr('app.highlights.ask_model', reply)
    with TestClient(app) as client:
        response = client.post('/api/jobs/video/highlights', json={
            'prompt_id': area, 'min_seconds': 10, 'max_seconds': 40,
            'campaign_context': 'Definir objetivos antes de decidir.',
        })
        assert response.status_code == 202
        for _ in range(200):
            job = client.get('/api/jobs/video').json()
            if job['highlights_status'] in ('completed', 'failed'):
                break
            time.sleep(.01)
        assert job['highlights_status'] == 'completed', job
        result = client.get('/api/jobs/video/highlights').json()
        clip = result['highlights'][0]
        assert result['version'] == 1
        assert (clip['start'], clip['end']) == (0, 30)
        assert clip['text'] == 'Idea completa 0. Idea completa 1. Idea completa 2.'
        assert set(clip) == {'id', 'rank', 'title', 'reason', 'score', 'start_unit_id', 'end_unit_id',
                             'start', 'end', 'duration', 'text', 'segment_ids'}
        report = client.get('/api/jobs/video/highlights/export/html').text
        assert '<blockquote>Idea completa 0. Idea completa 1. Idea completa 2.</blockquote>' in report
        assert 'suggested_cta' not in report and 'Texto de apertura' not in report


def test_campaign_context_participates_in_personalization_review(app):
    with TestClient(app) as client:
        body = request(area='activation', objective='Conserva el riesgo y facilita definir objetivos.',
                       campaign_context='Definir objetivos antes de invertir.')
        saved = client.post('/api/personalizations/manual', json=body).json()
        options = {'prompt_id': 'activation', 'personalization_id': saved['id'],
                   'audience_context': body['audience'], 'min_seconds': 10, 'max_seconds': 40,
                   'campaign_context': 'Operar inmediatamente.'}
        assert client.post('/api/jobs/video/highlights', json=options).status_code == 422


def test_gbm_repair_and_global_ranking_keep_selection_goal_and_campaign(app, monkeypatch):
    from app.highlights import CandidateBatch, Ranking

    folder = app.state.store.root / 'video'
    transcript = {'title': 'Long episode', 'language': 'es', 'segments': [
        {'id': f's{i}', 'start': i * 10, 'end': (i + 1) * 10,
         'text': f'Idea {i}: ' + 'contexto financiero relevante ' * 10}
        for i in range(45)
    ]}
    (folder / 'transcript.json').write_text(json.dumps(transcript))
    app.state.highlight_runner.config_path.write_text('[llm]\napi_key="test-secret"\nmax_input_chars=4000\n')
    calls = []
    def reply(client, model, instructions, payload, schema):
        calls.append((schema, payload))
        assert payload['campaign_context'] == 'Revisar objetivos y diversificación.'
        assert 'GBM' in payload['master_brief']
        assert 'Exclude return promises' in instructions
        if schema is Ranking:
            assert "area's acquisition or activation" in instructions
            return Ranking(summary='Global overview', candidate_ids=[payload['candidates'][0]['candidate_id']]), None
        assert schema is CandidateBatch
        units = payload['units']
        return schema(summary='Section overview', highlights=[{
            'title': 'Una idea para revisar', 'reason': 'Práctica respaldada por el episodio.', 'score': 90,
            'start_unit_id': 'invented' if len(calls) == 1 else units[0]['id'], 'end_unit_id': units[2]['id'],
        }]), None
    monkeypatch.setattr('app.highlights.ask_model', reply)
    with TestClient(app) as client:
        client.post('/api/jobs/video/highlights', json={
            'prompt_id': 'activation', 'min_seconds': 20, 'max_seconds': 40,
            'campaign_context': 'Revisar objetivos y diversificación.',
        })
        for _ in range(200):
            job = client.get('/api/jobs/video').json()
            if job['highlights_status'] in ('completed', 'failed'):
                break
            time.sleep(.01)
        assert job['highlights_status'] == 'completed', job
        assert calls[-1][0] is Ranking
        assert 'correction' in calls[1][1]
