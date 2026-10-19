import html
import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, urlparse

MODELS = {"tiny", "base", "small", "medium", "large-v3", "turbo"}
MAX_DURATION = int(os.getenv("MAX_VIDEO_SECONDS", "10800"))
logger = logging.getLogger(__name__)


def youtube_url(raw: str) -> str:
    parsed = urlparse(raw.strip())
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
        raise ValueError("Enter a valid YouTube video URL.")
    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/")
    elif host in {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        else:
            match = re.fullmatch(r"/(?:shorts|embed|live)/([A-Za-z0-9_-]{11})/?", parsed.path)
            video_id = match[1] if match else ""
    else:
        video_id = ""
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("Use a YouTube video link, rather than a playlist or channel URL.")
    return f"https://www.youtube.com/watch?v={video_id}"


def ffmpeg_path():
    import imageio_ffmpeg

    return shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()


def inspect_media(path: Path):
    import av

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("This file does not contain a video track.")
        if not container.streams.audio:
            raise ValueError("This video does not contain an audio track to transcribe.")
        video = container.streams.video[0]
        duration = float(container.duration / av.time_base) if container.duration else 0
        if duration <= 0:
            raise ValueError("Could not determine the video duration.")
        if duration > MAX_DURATION:
            raise ValueError(f"Video exceeds the {MAX_DURATION // 60}-minute duration limit.")
        # Audio is decoded separately; retain the offset against the displayed video timeline.
        audio = container.streams.audio[0]
        origin = float(container.start_time / av.time_base) if container.start_time is not None else 0
        audio_start = float(audio.start_time * audio.time_base) if audio.start_time is not None else origin
        return {
            "duration": duration,
            "width": video.width,
            "height": video.height,
            "fps": float(video.average_rate) if video.average_rate else 25,
            "audio_offset": audio_start - origin,
            "video_codec": video.codec_context.name,
            "audio_codec": audio.codec_context.name,
        }


def make_preview(source: Path, metadata):
    if (
        source.suffix.lower() in {".mp4", ".m4v"}
        and metadata["video_codec"] == "h264"
        and metadata["audio_codec"] == "aac"
    ):
        return source
    preview = source.parent / "preview.mp4"
    codecs = (
        ["-c:v", "copy"]
        if metadata["video_codec"] == "h264"
        else [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
        ]
    )
    # H.264 yuv420p needs even dimensions. For odd-sized inputs use yuv444p to avoid resizing.
    if metadata["width"] % 2 or metadata["height"] % 2:
        codecs = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv444p"]
    result = subprocess.run(
        [
            ffmpeg_path(),
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            *codecs,
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(preview),
        ],
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )
    if result.returncode:
        raise ValueError("Could not create a browser video preview. Try an H.264 MP4 file.")
    return preview


def timestamp(seconds, srt=False):
    millis = max(0, round(seconds * 1000))
    hours, rest = divmod(millis, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}{',' if srt else '.'}{ms:03}"


def export_transcript(transcript, kind):
    segments = transcript["segments"]
    if kind == "json":
        return json.dumps(transcript, ensure_ascii=False, indent=2)
    if kind == "txt":
        return "\n".join(s["text"] for s in segments) + "\n"
    if kind == "srt":
        return (
            "\n\n".join(
                f"{i}\n{timestamp(s['start'], True)} --> {timestamp(s['end'], True)}\n{s['text']}"
                for i, s in enumerate(segments, 1)
            )
            + "\n"
        )
    if kind == "html":
        rows = "\n".join(
            f"<article><time>{timestamp(s['start'])} – {timestamp(s['end'])}</time>"
            f"<p>{html.escape(s['text'])}</p></article>"
            for s in segments
        )
        title = html.escape(transcript["title"])
        return (
            "<!doctype html><html lang='en'><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{title} — Transcript</title><style>"
            "body{font:17px/1.7 system-ui;max-width:850px;margin:60px auto;padding:0 24px;color:#182b28}"
            "article{border-top:1px solid #ddd;padding:20px 0}time{font:13px monospace;color:#596e67}"
            f"p{{margin:8px 0}}</style><h1>{title}</h1><p>Whisper transcript · "
            f"{html.escape(transcript['language'])}</p>{rows}</html>"
        )
    raise ValueError("Unknown export format")


class Pipeline:
    def __init__(self, store):
        self.store = store
        self.models = {}

    def run(self, job_id):
        job = self.store.get(job_id)
        folder = self.store.root / job_id
        try:
            self.store.update(
                job_id,
                status="running",
                stage="acquire",
                progress=2,
                message="Preparing your video…",
                error=None,
            )
            if job["source_type"] == "youtube" and not (
                job.get("source_file") and (folder / job["source_file"]).is_file()
            ):
                source = self.download(job, folder)
            else:
                source = folder / job["source_file"]
            self.store.update(
                job_id,
                source_file=source.name,
                stage="audio",
                progress=25,
                message="Inspecting video and extracting audio…",
            )
            metadata = inspect_media(source)
            self.store.update(job_id, metadata=metadata)
            audio = folder / "audio.wav"
            result = subprocess.run(
                [
                    ffmpeg_path(),
                    "-nostdin",
                    "-y",
                    "-i",
                    str(source),
                    "-vn",
                    "-map",
                    "0:a:0",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(audio),
                ],
                capture_output=True,
                text=True,
                timeout=1800,
                check=False,
            )
            if result.returncode:
                raise ValueError("Audio extraction failed. Try a different video file.")
            self.store.update(job_id, message="Preparing a browser-compatible video preview…")
            preview = make_preview(source, metadata)
            self.store.update(job_id, preview_file=preview.name)
            self.store.update(
                job_id,
                stage="model",
                progress=35,
                message=f"Loading Whisper {job['model']}. First use downloads the model…",
            )
            model = self.get_model(job["model"])
            self.store.update(
                job_id,
                stage="transcribe",
                progress=40,
                message="Listening and creating timestamped transcript…",
            )
            segments, info = model.transcribe(
                str(audio),
                language=job["language"] or None,
                word_timestamps=True,
                vad_filter=True,
                beam_size=5,
            )
            collected = []
            offset = metadata["audio_offset"]
            for segment in segments:
                collected.append(
                    {
                        "id": f"seg_{len(collected) + 1:05}",
                        "start": max(0, segment.start + offset),
                        "end": max(0, segment.end + offset),
                        "text": segment.text.strip(),
                        "avg_logprob": segment.avg_logprob,
                        "no_speech_prob": segment.no_speech_prob,
                        "words": [
                            {
                                "word": w.word,
                                "start": max(0, w.start + offset),
                                "end": max(0, w.end + offset),
                                "probability": w.probability,
                            }
                            for w in segment.words or []
                        ],
                    }
                )
                progress = min(97, 40 + 57 * segment.end / metadata["duration"])
                self.store.update(
                    job_id,
                    progress=round(progress),
                    message=f"Transcribing · {timestamp(segment.end).split('.')[0]} processed",
                )
            current = self.store.get(job_id)
            transcript = {
                "version": 1,
                "job_id": job_id,
                "title": current["title"],
                "model": job["model"],
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": metadata["duration"],
                "segments": collected,
                "text": " ".join(s["text"] for s in collected),
            }
            temp = folder / "transcript.tmp"
            temp.write_text(export_transcript(transcript, "json"), encoding="utf-8")
            temp.replace(folder / "transcript.json")
            audio.unlink(missing_ok=True)
            self.store.update(
                job_id,
                status="completed",
                stage="complete",
                progress=100,
                detected_language=info.language,
                segment_count=len(collected),
                message="Transcript ready." if collected else "Finished. No speech was detected.",
            )
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            self.store.update(job_id, status="failed", message="Processing stopped.", error=str(exc)[:1200])

    def get_model(self, name):
        from faster_whisper import WhisperModel

        if name not in self.models:
            self.models.clear()  # Keep one model in memory on local machines.
            self.models[name] = WhisperModel(name, device="cpu", compute_type="int8")
        return self.models[name]

    def download(self, job, folder):
        from yt_dlp import YoutubeDL

        def progress(data):
            if data["status"] == "downloading":
                total = data.get("total_bytes") or data.get("total_bytes_estimate")
                fraction = min(1, data.get("downloaded_bytes", 0) / total) if total else 0
                self.store.update(
                    job["id"],
                    progress=round(3 + 20 * fraction),
                    message="Downloading video and audio from YouTube…",
                )

        def match_filter(info, *, incomplete=False):
            if info.get("is_live") or info.get("live_status") == "is_upcoming":
                return "Live and upcoming videos are not supported. Use a finished video."
            if info.get("duration", 0) and info["duration"] > MAX_DURATION:
                return f"Video exceeds the {MAX_DURATION // 60}-minute duration limit."

        options = {
            "format": "bestvideo+bestaudio/best",
            "noplaylist": True,
            "outtmpl": str(folder / "source.%(ext)s"),
            "merge_output_format": "mkv",
            "ffmpeg_location": ffmpeg_path(),
            "progress_hooks": [progress],
            "match_filter": match_filter,
            "quiet": True,
            "no_warnings": False,
            "retries": 3,
            "socket_timeout": 30,
            "js_runtimes": {"node": {}},
        }
        with YoutubeDL(options) as downloader:
            info = downloader.extract_info(job["source_url"], download=True)
        if not info or info.get("_type") == "playlist":
            raise ValueError("No downloadable video found at this URL.")
        files = [
            p
            for p in folder.glob("source.*")
            if p.suffix not in {".part", ".ytdl"} and not re.search(r"\.f\d+\.", p.name)
        ]
        if len(files) != 1:
            raise ValueError("Download did not produce a complete video. Please retry.")
        self.store.update(
            job["id"],
            title=info.get("title", "YouTube video"),
            youtube_id=info.get("id"),
            channel=info.get("channel"),
        )
        return files[0]
