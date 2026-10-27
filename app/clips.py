"""Render validated highlight ranges from original media, publishing complete batches atomically."""

import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import uuid
from datetime import UTC, datetime

from app.pipeline import ffmpeg_path, inspect_media

logger = logging.getLogger(__name__)


def highlight_hash(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def validate_ranges(highlights, duration):
    items = highlights.get("highlights", [])
    if not items:
        raise ValueError("No highlights are available to render.")
    ids = set()
    for item in items:
        start, end = item.get("start"), item.get("end")
        if (
            not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
            or end > duration + 0.05
        ):
            raise ValueError("A highlight has an invalid time range. Regenerate highlights before clipping.")
        identity = item.get("id", "")
        if not re.fullmatch(r"highlight_\d{3}", identity) or identity in ids:
            raise ValueError("Highlights contain invalid or duplicate IDs.")
        ids.add(identity)
    return items


def render_clip(source, destination, start, end, metadata, mode="original", detector=None, words=None):
    expected = (metadata["width"], metadata["height"])
    expression, framing = "", {"mode": "original"}
    commands_file = destination.with_suffix(".crop.cmd")
    if mode != "original":
        from app.reframing import plan_reframe

        expression, expected, framing = plan_reframe(
            source, start, end, metadata, mode, commands_file, detector
        )
    captions = {"mode": "off"}
    if words is not None:
        from app.captions import write_ass

        subtitle_file = destination.with_suffix(".ass")
        captions = write_ass(subtitle_file, words, start, end, *expected)
        # Apply after all crop/stack/scale operations, in clip-relative time.
        expression = (expression + "," if expression else "setpts=PTS-STARTPTS,")
        expression += f"ass={subtitle_file.name}"
    filters = ["-vf", expression] if expression else []
    pixel_format = "yuv444p" if expected[0] % 2 or expected[1] % 2 else "yuv420p"
    result = subprocess.run(
        [
            ffmpeg_path(),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.6f}",
            *(["-noautorotate"] if mode == "original" else []),
            "-i",
            str(source.resolve()),
            "-t",
            f"{end - start:.6f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            *filters,
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-pix_fmt",
            pixel_format,
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(destination.resolve()),
        ],
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
        cwd=destination.parent,
    )
    commands_file.unlink(missing_ok=True)
    if result.returncode:
        raise ValueError(
            "FFmpeg could not render this clip. Check the source video and, for subtitles, libass support."
        )
    rendered = inspect_media(destination)
    if (rendered["width"], rendered["height"]) != expected:
        raise ValueError("Rendered dimensions differ from the requested clip format.")
    tolerance = max(0.25, 2 / max(metadata.get("fps", 25), 0.1))
    if abs(rendered["duration"] - (end - start)) > tolerance:
        raise ValueError("Rendered duration differs from the selected passage.")
    # Decode a frame to catch unreadable output rather than trusting container metadata alone.
    import av

    with av.open(str(destination)) as container:
        if next(container.decode(video=0), None) is None:
            raise ValueError("The rendered clip contains no readable video frames.")
    return {**rendered, "framing": framing, "captions": captions}


class ClipRunner:
    def __init__(self, store):
        self.store = store

    def run(self, job_id, clip_id=None):
        folder = self.store.root / job_id
        revision = uuid.uuid4().hex
        staging, published = folder / f".clips-{revision}", folder / f"clips-{revision}"
        try:
            self.store.update(
                job_id,
                clips_status="running",
                clips_progress=1,
                clips_error=None,
                clips_message="Inspecting the original video…",
            )
            job = self.store.get(job_id)
            previous = json.loads((folder / "clips.json").read_text()) if clip_id else None
            mode = previous["mode"] if previous else job.get("clips_mode", "original")
            subtitles = previous.get("subtitles", False) if previous else job.get("clips_subtitles", False)
            words = None
            if subtitles:
                from app.captions import caption_words

                words = caption_words(json.loads((folder / "transcript.json").read_text()))
            detector = None
            if mode == "face_follow":
                from app.reframing import FaceDetector

                detector = FaceDetector()
            source = folder / job["source_file"]
            metadata = inspect_media(source)
            highlights = json.loads((folder / "highlights.json").read_text())
            items = validate_ranges(highlights, metadata["duration"])
            if previous and previous["highlights_sha256"] != highlight_hash(highlights):
                raise ValueError("Highlights have changed. Generate all clips again.")
            staging.mkdir()
            clips = []
            for index, item in enumerate(items):
                if previous and item["id"] != clip_id:
                    saved = next(c for c in previous["clips"] if c["id"] == item["id"])
                    # Reuse the exact files without encoding or duplicating their data.
                    # The old batch remains intact until the new manifest is published.
                    for suffix in (".mp4", ".ass"):
                        old_file = folder / previous["directory"] / (item["id"] + suffix)
                        if old_file.exists():
                            os.link(old_file, staging / old_file.name)
                        elif suffix == ".mp4":
                            raise ValueError("A previous clip is missing. Generate all clips again.")
                    clips.append({**saved, "revision": saved.get("revision", previous["revision"])})
                    continue
                self.store.update(
                    job_id,
                    clips_progress=round(5 + 90 * index / len(items)),
                    clips_message=(
                        f"Regenerating clip {item['rank']}…" if clip_id else
                        f"Following the face and rendering clip {index + 1} of {len(items)}…"
                        if mode == "face_follow"
                        else f"Rendering clip {index + 1} of {len(items)}…"
                    ),
                )
                filename = f"{item['id']}.mp4"
                target = staging / filename
                args = (source, target, item["start"], item["end"], metadata, mode, detector)
                caption_revision = None
                if subtitles:
                    from app.caption_edits import corrected_words, editor_document

                    document = editor_document(folder, item["id"], item["start"], item["end"], words)
                    output = render_clip(*args, corrected_words(document))
                    caption_revision = document["revision"]
                else:
                    output = render_clip(*args)
                clips.append(
                    {
                        "id": item["id"],
                        "revision": revision,
                        "rank": item["rank"],
                        "title": item["title"],
                        "source_start": item["start"],
                        "source_end": item["end"],
                        "duration": output["duration"],
                        "width": output["width"],
                        "height": output["height"],
                        "size_bytes": target.stat().st_size,
                        "filename": filename,
                        "framing": output["framing"],
                        "captions": output.get("captions", {"mode": "off"}),
                        "captions_revision": caption_revision,
                    }
                )
            result = {
                "version": 4,
                "mode": mode,
                "subtitles": subtitles,
                "job_id": job_id,
                "revision": revision,
                "created_at": datetime.now(UTC).isoformat(),
                "highlights_sha256": highlight_hash(highlights),
                "directory": published.name,
                "clips": clips,
            }
            staging.rename(published)
            temp = folder / "clips.tmp"
            temp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            temp.replace(folder / "clips.json")
            # Only the new complete batch is exposed. Remove previous batches and abandoned partials.
            for old in [*folder.glob("clips-*"), *folder.glob(".clips-*")]:
                if old != published and old.is_dir() and not old.is_symlink():
                    shutil.rmtree(old)
            self.store.update(
                job_id,
                clips_status="completed",
                clips_progress=100,
                clips_message="Clips ready.",
                clips_count=len(clips),
                clips_revision=revision,
                clips_target_id=None,
            )
        except Exception as exc:  # noqa: BLE001 -- Keep failures within the background job.
            logger.warning("Clip job %s failed (%s)", job_id, type(exc).__name__)
            shutil.rmtree(staging, ignore_errors=True)
            # Remove an unpublished directory if publication failed between rename and manifest write.
            try:
                current_directory = json.loads((folder / "clips.json").read_text())["directory"]
            except (OSError, ValueError, KeyError):
                current_directory = None
            if published.exists() and current_directory != published.name:
                shutil.rmtree(published, ignore_errors=True)
            self.store.update(
                job_id,
                clips_status="completed" if clip_id and current_directory else "failed",
                clips_target_id=None,
                **({"clips_revision": json.loads((folder / "clips.json").read_text())["revision"]}
                   if clip_id and current_directory else {}),
                clips_message="Clip rendering stopped.",
                clips_error=str(exc)
                if isinstance(exc, ValueError)
                else "Clip rendering failed. Check available disk space and retry.",
            )
