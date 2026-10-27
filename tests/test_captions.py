import copy
from itertools import pairwise

import pytest

from app.captions import caption_words, clip_cues, write_ass


def transcript(intervals):
    return {"segments": [{"words": [
        {"word": f"word{i}", "start": a, "end": b} for i, (a, b) in enumerate(intervals)
    ]}]}


@pytest.mark.parametrize("intervals", [
    [(1.44, 1.72), (1.72, 1.72), (1.72, 2)],
    [(0, .4), (.5, .5), (.6, 1)],
    [(0, .4), (.4, .4), (.4, .4), (.4, 1)],
    [(0, 0), (0, .5)],
    [(0, .5), (.5, .5)],
    [(0, 0)],
    [(0, 1), (5, 5), (10, 11)],
])
def test_repair_preserves_words_order_and_original(intervals):
    original = transcript(intervals)
    before = copy.deepcopy(original)
    words = caption_words(original)
    cues = clip_cues(words, 0, 12)
    assert original == before
    assert len(cues) == len(intervals)
    assert all(c["end"] > c["start"] >= 0 for c in cues)
    assert all(a["end"] <= b["start"] for a, b in pairwise(cues))
    if intervals[-1][0] == 10:
        assert cues[1]["end"] - cues[1]["start"] <= 60


def test_clip_offsets_boundaries_and_pauses():
    words = caption_words(transcript([(9, 10.2), (10.5, 10.8), (11.8, 12.2)]))
    assert clip_cues(words, 10, 12) == [
        {"word": "word0", "start": 0, "end": 20},
        {"word": "word1", "start": 50, "end": 80},
        {"word": "word2", "start": 180, "end": 200},
    ]


def test_missing_word_times_fail_clearly():
    with pytest.raises(ValueError, match="word timestamps"):
        caption_words({"segments": [{"text": "old transcript"}]})


def test_ass_escapes_text_and_limits_animation_to_word_duration(tmp_path):
    path = tmp_path / "captions.ass"
    words = [{"word": r"{\p1}\Ntest", "start": 0, "end": .08}]
    result = write_ass(path, words, 0, 1, 360, 640)
    text = path.read_text()
    assert result["word_count"] == 1
    assert r"\t(0,40,\fscx100\fscy100)" in text
    assert r"{\p1}" not in text
    with pytest.raises(ValueError, match="No timed words"):
        write_ass(path, words, 5, 6, 360, 640)


def test_punctuation_recovered_locally_without_changing_times():
    original = {"segments": [{"text": "Hola me llamo Juan Pablo. Otra frase.", "words": [
        {"word": w, "start": i * .3, "end": (i + 1) * .3}
        for i, w in enumerate(["Hola", "me", "llamo", "Juan", "Pablo", "Otra", "frase"])
    ]}]}
    before = copy.deepcopy(original)
    words = caption_words(original)
    assert words[4]["word"] == "Pablo."
    assert words[4]["start"] == before["segments"][0]["words"][4]["start"]
    assert original == before


def test_phrase_boundaries_count_punctuation_and_pauses():
    from app.captions import phrase_groups

    words = [{"word": w, "start": i * 20, "end": (i + 1) * 20}
             for i, w in enumerate(["Hola", "me", "llamo", "Juan", "Pablo.", "Otra", "frase", "empieza", "aquí", "y", "sigue", "sin", "puntuación", "hasta", "terminar"])]
    groups = phrase_groups(words)
    assert [len(g) for g in groups] == [5, 8, 2]
    words[1]["end"] = -30  # A long pause also ends a short block.
    assert len(phrase_groups(words)[0]) == 2
    assert all(len(g) <= 8 for g in phrase_groups(words))


def test_accumulation_holds_previous_words_until_block_reset(tmp_path):
    path = tmp_path / "phrases.ass"
    words = [{"word": w, "start": i * .3, "end": i * .3 + .2}
             for i, w in enumerate(["Hola", "me", "llamo", "Juan", "Pablo.", "Otra", "frase."])]
    write_ass(path, words, 0, 3, 1080, 1920)
    events = [line for line in path.read_text().splitlines() if line.startswith("Dialogue:")]
    assert len(events) == 7
    # Future words reserve their space invisibly; earlier words stay fully visible.
    assert r"{\alpha&HFF&\fscx100\fscy100}me" in events[0]
    assert r"{\alpha&H00&\fscx100\fscy100}Hola" in events[1]
    assert "0:00:00.00,0:00:00.30" in events[0]  # Remains through the inter-word gap.
    assert "Hola" in events[4] and "Pablo." in events[4]
    assert "Hola" not in events[5] and "Otra" in events[5]
    assert r"\N" in events[0]  # Full phrase uses a fixed two-line layout from the start.


def test_mismatched_segment_keeps_word_times_and_count_fallback():
    original = {"segments": [{"text": "Different sentence.", "words": [
        {"word": "Actual", "start": 0, "end": .2},
        {"word": "words", "start": .2, "end": .4},
    ]}]}
    assert [w["word"] for w in caption_words(original)] == ["Actual", "words"]


def test_caption_size_stays_uniform_across_short_and_long_phrases(tmp_path):
    import re

    text = "Hola. Esta es una frase bastante más larga con palabras normales."
    words = [{"word": w, "start": i * .3, "end": i * .3 + .2}
             for i, w in enumerate(text.split())]
    path = tmp_path / "uniform.ass"
    write_ass(path, words, 0, 5, 1080, 1920)
    events = [line for line in path.read_text().splitlines() if line.startswith("Dialogue:")]
    assert {re.search(r"\\fs([\d.]+)", line)[1] for line in events} == {"84.24"}
    assert all(r"\fscx98\fscy98" in line for line in events)
    assert all(r"\t(0,80,\fscx100\fscy100)" in line for line in events)
