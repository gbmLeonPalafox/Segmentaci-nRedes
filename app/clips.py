"""Render validated highlight ranges from original media, publishing complete batches atomically."""

import hashlib
import json
import logging
import math
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


def render_clip(source, destination, start, end, metadata):
    pixel_format = "yuv444p" if metadata["width"] % 2 or metadata["height"] % 2 else "yuv420p"
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
            "-noautorotate",
            "-i",
            str(source),
            "-t",
            f"{end - start:.6f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
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
            str(destination),
        ],
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    if result.returncode:
        raise ValueError("FFmpeg could not render this clip. Check that the original video is readable.")
    rendered = inspect_media(destination)
    if (rendered["width"], rendered["height"]) != (metadata["width"], metadata["height"]):
        raise ValueError("Rendered dimensions differ from the original video.")
    tolerance = max(0.25, 2 / max(metadata.get("fps", 25), 0.1))
    if abs(rendered["duration"] - (end - start)) > tolerance:
        raise ValueError("Rendered duration differs from the selected passage.")
    # Decode a frame to catch unreadable output rather than trusting container metadata alone.
    import av

    with av.open(str(destination)) as container:
        if next(container.decode(video=0), None) is None:
            raise ValueError("The rendered clip contains no readable video frames.")
    return rendered


class ClipRunner:
    def __init__(self, store):
        self.store = store

    def run(self, job_id):
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
            source = folder / job["source_file"]
            metadata = inspect_media(source)
            highlights = json.loads((folder / "highlights.json").read_text())
            items = validate_ranges(highlights, metadata["duration"])
            staging.mkdir()
            clips = []
            for index, item in enumerate(items):
                self.store.update(
                    job_id,
                    clips_progress=round(5 + 90 * index / len(items)),
                    clips_message=f"Rendering clip {index + 1} of {len(items)}…",
                )
                filename = f"{item['id']}.mp4"
                target = staging / filename
                output = render_clip(source, target, item["start"], item["end"], metadata)
                clips.append(
                    {
                        "id": item["id"],
                        "rank": item["rank"],
                        "title": item["title"],
                        "source_start": item["start"],
                        "source_end": item["end"],
                        "duration": output["duration"],
                        "width": output["width"],
                        "height": output["height"],
                        "size_bytes": target.stat().st_size,
                        "filename": filename,
                    }
                )
            result = {
                "version": 1,
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
                clips_status="failed",
                clips_message="Clip rendering stopped.",
                clips_error=str(exc)
                if isinstance(exc, ValueError)
                else "Clip rendering failed. Check available disk space and retry.",
            )
