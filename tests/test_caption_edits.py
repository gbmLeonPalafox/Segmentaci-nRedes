import json

from app.caption_edits import corrected_words, editor_document, save_edits


def test_multiple_corrected_words_share_original_interval_and_stale_edits_are_ignored(tmp_path):
    data = {"segments": [{"words": [{"word": "Juanpablo", "start": 0, "end": 1}]}]}
    path = tmp_path / "transcript.json"
    path.write_text(json.dumps(data))
    document = editor_document(tmp_path, "highlight_001", 0, 1)
    save_edits(tmp_path, "highlight_001", document, ["Juan Pablo"])
    saved = editor_document(tmp_path, "highlight_001", 0, 1)
    assert corrected_words(saved) == [
        {"word": "Juan", "start": 0, "end": .5},
        {"word": "Pablo", "start": .5, "end": 1},
    ]
    assert editor_document(tmp_path, "highlight_002", 0, 1)["words"][0]["text"] == "Juanpablo"
    data["segments"][0]["words"][0]["word"] = "NewTranscript"
    path.write_text(json.dumps(data))
    fresh = editor_document(tmp_path, "highlight_001", 0, 1)
    assert not fresh["has_edits"]
    assert fresh["words"][0]["text"] == "NewTranscript"
    assert fresh["revision"] != saved["revision"]


def test_remove_misheard_connector_and_keep_name_timing(tmp_path):
    data = {"segments": [{"words": [
        {"word": "Nesca", "start": 1, "end": 1.4},
        {"word": "y", "start": 1.4, "end": 1.6},
        {"word": "Heiniks", "start": 1.6, "end": 2},
    ]}]}
    path = tmp_path / "transcript.json"
    path.write_text(json.dumps(data))
    original = path.read_bytes()
    document = editor_document(tmp_path, "highlight_001", 1, 2)
    save_edits(tmp_path, "highlight_001", document, ["SK", "", "Hynix"])
    saved = editor_document(tmp_path, "highlight_001", 1, 2)
    assert [w["text"] for w in saved["words"]] == ["SK", "", "Hynix"]
    assert corrected_words(saved) == [
        {"word": "SK", "start": 1, "end": 1.6},
        {"word": "Hynix", "start": 1.6, "end": 2},
    ]
    assert path.read_bytes() == original


def test_removed_leading_trailing_words_and_long_pauses():
    document = {"words": [
        {"text": "", "start": 0, "end": .2},
        {"text": "SK", "start": .2, "end": .5},
        {"text": "", "start": .5, "end": .6},
        {"text": "", "start": 5, "end": 5.2},
        {"text": "Hynix", "start": 10, "end": 10.5},
        {"text": "", "start": 10.5, "end": 10.7},
    ]}
    assert corrected_words(document) == [
        {"word": "SK", "start": 0, "end": .6},
        {"word": "Hynix", "start": 10, "end": 10.7},
    ]
