"""Word captions derived from saved timings; never modify the source transcript."""

import math
import re


def caption_words(transcript):
    words = []
    for segment in transcript.get("segments", []):
        source_words = segment.get("words", [])
        if source_words and segment.get("text"):
            from app.highlights import normalized_letters, punctuated_words

            punctuated = punctuated_words(segment)
            # Recover punctuation only on an exact text match. Otherwise retain
            # the individual word times and use count/pause-based grouping.
            if len(punctuated) == len(source_words) and all(
                normalized_letters(a["word"]) == normalized_letters(b["word"])
                for a, b in zip(source_words, punctuated, strict=True)
            ):
                source_words = punctuated
        for word in source_words:
            start, end = word.get("start"), word.get("end")
            text = str(word.get("word", "")).strip()
            if (text and isinstance(start, (int, float)) and isinstance(end, (int, float))
                    and math.isfinite(start) and math.isfinite(end) and 0 <= start <= end):
                words.append({"word": text, "start": start, "end": end})
    words.sort(key=lambda w: w["start"])
    if not words:
        raise ValueError("Dynamic subtitles require a transcript with word timestamps.")

    # Repair each zero-duration run together. Borrow at most 120 ms from each
    # neighbor, or use a nearby gap. Never stretch captions across a long silence.
    i = 0
    while i < len(words):
        if words[i]["end"] > words[i]["start"]:
            i += 1
            continue
        j = i + 1
        while j < len(words) and words[j]["end"] == words[j]["start"]:
            j += 1
        previous = words[i - 1] if i else None
        following = words[j] if j < len(words) else None
        left = words[i]["start"]
        right = words[j - 1]["end"]
        if previous:
            left = max(previous["start"] + (previous["end"] - previous["start"]) / 2,
                       min(previous["end"], left) - 0.12)
            left = max(left, words[i]["start"] - 0.3)
        if following:
            right = min(following["end"] - (following["end"] - following["start"]) / 2,
                        max(following["start"], right) + 0.12)
            right = min(right, words[j - 1]["end"] + 0.3)
        else:
            right += 0.12 * (j - i)
        left = max(0, left)
        if right > left:
            if previous:
                previous["end"] = min(previous["end"], left)
            if following:
                following["start"] = max(following["start"], right)
            step = (right - left) / (j - i)
            for k in range(i, j):
                words[k]["start"] = left + (k - i) * step
                words[k]["end"] = left + (k - i + 1) * step
        i = j
    return words


def clip_cues(words, start, end):
    """Use centisecond intervals (ASS resolution), clipped to this export."""
    cues = []
    duration = math.floor((end - start) * 100 + 1e-6)
    for i, word in enumerate(words):
        a = max(0, round((word["start"] - start) * 100))
        b = min(duration, round((word["end"] - start) * 100))
        if i + 1 < len(words):
            b = min(b, round((words[i + 1]["start"] - start) * 100))
        if b > a:
            cues.append({"word": word["word"], "start": a, "end": b})
    return cues


def ass_time(ticks):
    seconds, cs = divmod(ticks, 100)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02}:{seconds:02}.{cs:02}"


def escape_ass(text):
    # User text must not become ASS override tags or line-break commands.
    return text.replace("\\", "＼").replace("{", "｛").replace("}", "｝").replace("\n", " ").replace("\r", " ")


def phrase_groups(cues):
    """Purely local boundaries: sentences, pauses, punctuation after 5, cap at 8."""
    groups, current = [], []
    for i, cue in enumerate(cues):
        current.append(cue)
        sentence = re.search(r'[.!?…]["\'»”’)]*$', cue["word"])
        punctuation = re.search(r'[,;:]["\'»”’)]*$', cue["word"])
        pause = i + 1 < len(cues) and cues[i + 1]["start"] - cue["end"] >= 45
        if sentence or pause or len(current) >= 8 or (len(current) >= 5 and punctuation):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def text_width(text):
    """Conservative Arial bold width estimate in ems; no font/API dependency."""
    total = 0
    for char in text:
        if char in " ilI.,:;!'|":
            total += 0.32
        elif char in "mwMW@%":
            total += 0.95
        elif char.isascii():
            total += 0.67
        else:
            total += 1
    return total


def phrase_layout(group, available=14):
    """Pre-plan at most two balanced lines, including unrevealed words."""
    texts = [escape_ass(cue["word"]) for cue in group]
    if len(texts) == 1:
        return texts, None
    # A short phrase remains on one line; larger blocks use two fixed lines.
    if text_width(" ".join(texts)) <= available:
        return texts, None
    split = min(range(1, len(texts)), key=lambda i: max(
        text_width(" ".join(texts[:i])), text_width(" ".join(texts[i:]))
    ))
    return texts, split


def layout_groups(cues, available):
    """Prefer shorter blocks over shrinking every longer phrase."""
    result = []
    for group in phrase_groups(cues):
        current = []
        for cue in group:
            candidate = current + [cue]
            texts, split = phrase_layout(candidate, available)
            longest = max(text_width(" ".join(texts[:split])), text_width(" ".join(texts[split:])))
            if current and longest > available:
                result.append(current)
                current = [cue]
            else:
                current = candidate
        if current:
            result.append(current)
    return result


def write_ass(path, words, start, end, width, height):
    cues = clip_cues(words, start, end)
    if not cues:
        raise ValueError("No timed words are available for subtitles in this clip.")
    size = min(width * 0.078, height * 0.075)
    available = width * 0.82 / size
    x, y = round(width / 2), round(height * 0.74)
    outline = max(1, size * 0.065)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{size:.2f},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline:.2f},0,5,0,0,0,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    groups = layout_groups(cues, available)
    for group_index, group in enumerate(groups):
        texts, split = phrase_layout(group, available)
        longest = max(text_width(" ".join(texts[:split])), text_width(" ".join(texts[split:])))
        # Only an unusually long individual word needs a smaller size.
        fitted = min(size, width * 0.82 / max(1, longest))
        for index, cue in enumerate(group):
            if index + 1 < len(group):
                until = group[index + 1]["start"]
            else:
                until = cue["end"]
                if group_index + 1 < len(groups):
                    next_start = groups[group_index + 1][0]["start"]
                    until = min(next_start, until + 15)  # Brief reading hold; never bridge long pauses.
            animation = min(80, (cue["end"] - cue["start"]) * 5)
            pieces = []
            for j, text in enumerate(texts):
                if j:
                    pieces.append(r"\N" if j == split else r"\h")
                if j < index:
                    tags = r"{\alpha&H00&\fscx100\fscy100}"
                elif j == index:
                    tags = (f"{{\\alpha&H00&\\fscx98\\fscy98"
                            f"\\t(0,{animation},\\fscx100\\fscy100)}}")
                else:
                    tags = r"{\alpha&HFF&\fscx100\fscy100}"
                pieces.append(tags + text)
            positioning = f"{{\\pos({x},{y})\\fs{fitted:.2f}\\bord{max(0.5, fitted * 0.05):.2f}}}"
            lines.append(f"Dialogue: 0,{ass_time(cue['start'])},{ass_time(until)},"
                         f"Default,,0,0,0,,{positioning}{''.join(pieces)}")
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    return {"mode": "dynamic", "word_count": len(cues)}
