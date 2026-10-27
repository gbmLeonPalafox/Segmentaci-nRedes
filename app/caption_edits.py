"""Local spelling corrections keyed to the original timed words of each clip."""

import hashlib
import json

from app.captions import caption_words


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def editor_document(folder, clip_id, start, end, words=None):
    if words is None:
        words = caption_words(json.loads((folder / "transcript.json").read_text()))
    selected = [{**w, "start": max(start, w["start"]), "end": min(end, w["end"])}
                for w in words if w["end"] > start and w["start"] < end]
    if not selected:
        raise ValueError("No timed words are available to edit in this clip.")
    source_hash = digest({"start": start, "end": end, "words": selected})
    path = folder / "caption-edits.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    entry = saved.get(clip_id, {})
    texts = [w["word"] for w in selected]
    if entry.get("source_sha256") == source_hash and len(entry.get("texts", [])) == len(texts):
        texts = entry["texts"]
    return {
        "revision": digest({"source_sha256": source_hash, "texts": texts}),
        "source_sha256": source_hash,
        "words": [{**w, "text": text} for w, text in zip(selected, texts, strict=True)],
        "has_edits": any(w["word"] != text for w, text in zip(selected, texts, strict=True)),
    }


def save_edits(folder, clip_id, document, texts):
    if len(texts) != len(document["words"]):
        raise ValueError("Keep one entry per original word. A corrected entry can contain several words.")
    normalized = [" ".join(text.split()) for text in texts]
    if any(len(text) > 120 or any(ord(c) < 32 for c in text) for text in normalized):
        raise ValueError("Each correction must contain at most 120 printable characters.")
    if not any(normalized):
        raise ValueError("Keep at least one word in the subtitles. Empty words are removed.")
    path = folder / "caption-edits.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    saved[clip_id] = {"source_sha256": document["source_sha256"], "texts": normalized}
    temp = folder / "caption-edits.tmp"
    temp.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def corrected_words(document):
    words = []
    pending_start, pending_end = None, None
    for word in document["words"]:
        tokens = word["text"].split()
        if not tokens:
            # Give a removed word's nearby interval to a surviving neighbor,
            # so merging a misheard name does not introduce a blank flicker.
            if words and word["start"] - words[-1]["end"] <= 0.45:
                words[-1]["end"] = max(words[-1]["end"], word["end"])
            else:
                if pending_end is None or word["start"] - pending_end > 0.45:
                    pending_start = word["start"]
                pending_end = word["end"]
            continue
        start = word["start"]
        if pending_end is not None and start - pending_end <= 0.45:
            start = pending_start
        pending_start, pending_end = None, None
        step = (word["end"] - start) / len(tokens)
        for i, token in enumerate(tokens):
            words.append({"word": token, "start": start + i * step,
                          "end": start + (i + 1) * step})
    return words


def edits_pending(item, document):
    rendered = item.get("captions_revision")
    return rendered != document["revision"] if rendered else document["has_edits"]
