import asyncio
import json
import os
import re
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.clips import ClipRunner, highlight_hash
from app.config import HighlightOptions
from app.highlights import HighlightRunner, export_highlights
from app.pipeline import MODELS, Pipeline, export_transcript, youtube_url
from app.store import Store
from app.transcription import LANGUAGES
from app.uploads import CHUNK_BYTES, VIDEO_SUFFIXES, Uploads

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024**3)))
LANGUAGE = re.compile(r"[a-z]{2,3}")


class YouTubeInput(BaseModel):
    url: str = Field(max_length=2048)
    model: str = "whisper-1"
    language: str = ""


class UploadInput(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    size: int = Field(gt=0)
    model: str = "whisper-1"
    language: str = ""


class RetryInput(BaseModel):
    model: str | None = None


class ClipInput(BaseModel):
    mode: Literal["original", "center_crop", "face_follow"] = "original"


def create_app(data_dir=None, config_path=None):
    store = Store(Path(data_dir or os.getenv("DATA_DIR", "data")).resolve())
    pipeline = Pipeline(store, config_path)
    highlight_runner = HighlightRunner(store, config_path)
    clip_runner = ClipRunner(store)
    lifecycle_lock = threading.RLock()

    def serialized(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with lifecycle_lock:
                return function(*args, **kwargs)

        return wrapped

    def remove_files(job):
        folder = store.root / job["id"]
        if folder.parent != store.root or folder.is_symlink():
            raise ValueError("Invalid video storage directory.")
        if folder.exists():
            shutil.rmtree(folder)
        store.delete(job["id"])

    @asynccontextmanager
    async def lifespan(app):
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="transcription")
        app.state.executor = executor
        llm_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="highlights")
        app.state.llm_executor = llm_executor
        clip_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clips")
        app.state.clip_executor = clip_executor
        for job in reversed(store.all()):
            if job["status"] in {"deleting", "delete_failed"}:
                try:
                    remove_files(job)
                except (OSError, ValueError):
                    store.update(
                        job["id"], status="delete_failed", error="File cleanup failed. Try deletion again."
                    )
                continue
            if job.get("clips_status") in {"queued", "running"}:
                store.update(
                    job["id"],
                    clips_status="failed",
                    clips_error="Rendering was interrupted. Generate clips again.",
                )
                folder = store.root / job["id"]
                try:
                    previous_batch = json.loads((folder / "clips.json").read_text())["directory"]
                except (OSError, ValueError, KeyError):
                    previous_batch = None
                for partial in [*folder.glob(".clips-*"), *folder.glob("clips-*")]:
                    if partial.name != previous_batch and partial.is_dir() and not partial.is_symlink():
                        shutil.rmtree(partial, ignore_errors=True)
            if job.get("highlights_status") in {"queued", "running"}:
                store.update(
                    job["id"],
                    highlights_status="failed",
                    highlights_error="The server restarted during selection. Click Generate highlights to retry.",
                    highlights_message="Selection interrupted.",
                )
            if job["status"] in {"queued", "running"}:
                store.update(
                    job["id"], status="failed", message="Transcription interrupted.",
                    error="Server restarted. Retry to transcribe with OpenAI Whisper API.",
                )
        yield
        executor.shutdown(wait=True)
        llm_executor.shutdown(wait=True)
        clip_executor.shutdown(wait=True)

    app = FastAPI(title="Frame / Transcript Studio", lifespan=lifespan)
    app.state.store = store
    app.state.pipeline = pipeline
    app.state.highlight_runner = highlight_runner
    app.state.clip_runner = clip_runner

    def get_job(job_id):
        job = store.get(job_id)
        if not job:
            raise HTTPException(404, "Video job not found.")
        return job

    def options(model, language):
        if model not in MODELS:
            raise HTTPException(422, "Transcription uses OpenAI Whisper API (whisper-1).")
        try:
            settings = highlight_runner.settings()
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        if not settings.llm.api_key:
            raise HTTPException(422, "Add your OpenAI API key to [llm] api_key in config.toml.")
        language = language.strip().lower()
        if language and (not LANGUAGE.fullmatch(language) or language not in LANGUAGES):
            raise HTTPException(422, "Use a supported language code such as en or es, or select Auto-detect.")
        return language

    def new_job(source_type, title, model, language, create_folder=True, **extra):
        job_id = uuid.uuid4().hex
        folder = store.root / job_id
        if create_folder:
            folder.mkdir()
        job = {
            "id": job_id,
            "title": title,
            "source_type": source_type,
            "model": model,
            "language": language,
            "created_at": datetime.now(UTC).isoformat(),
            "status": "queued",
            "stage": "acquire",
            "progress": 0,
            "message": "Queued. Videos are processed one at a time.",
            "error": None,
            **extra,
        }
        return job, folder

    def enqueue(job):
        store.save(job)
        app.state.executor.submit(pipeline.run, job["id"])
        return job

    uploads = Uploads(store, enqueue)

    @app.post("/api/uploads", status_code=201)
    @serialized
    def begin_upload(body: UploadInput):
        language = options(body.model, body.language)
        if body.size > MAX_UPLOAD:
            raise HTTPException(413, "Video exceeds the configured upload size limit.")
        filename = Path(body.filename).name
        suffix = Path(filename).suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            raise HTTPException(422, "Choose a supported video file.")
        job, _ = new_job(
            "upload", filename, body.model, language,
            create_folder=False, source_file=f"source{suffix}",
        )
        return uploads.create(job, body.size)

    @serialized
    def append_upload(upload_id, offset, data):
        return uploads.append(upload_id, offset, data)

    @app.put("/api/uploads/{upload_id}")
    async def upload_part(upload_id: str, offset: int, request: Request):
        data = bytearray()
        async for part in request.stream():
            if len(data) + len(part) > CHUNK_BYTES:
                raise HTTPException(413, "Upload chunk exceeds the size limit.")
            data.extend(part)
        return await asyncio.to_thread(append_upload, upload_id, offset, data)

    @app.delete("/api/uploads/{upload_id}")
    @serialized
    def cancel_upload(upload_id: str):
        return uploads.cancel(upload_id)

    @app.post("/api/uploads/{upload_id}/complete", status_code=202)
    @serialized
    def finish_upload(upload_id: str):
        return uploads.complete(upload_id)

    @app.get("/api/health")
    def health():
        from app.pipeline import MAX_DURATION, ffmpeg_path

        return {
            "status": "ok",
            "ffmpeg": bool(Path(ffmpeg_path()).is_file()),
            "javascript_runtime": shutil.which("node") is not None,
            "max_upload_bytes": MAX_UPLOAD,
            "max_video_seconds": MAX_DURATION,
            "models": sorted(MODELS),
        }

    @app.get("/api/jobs")
    def list_jobs():
        return store.all()

    @app.get("/api/settings")
    def public_settings():
        try:
            settings = highlight_runner.settings()
            return {
                "llm": {
                    "configured": bool(settings.llm.api_key),
                    "model": settings.llm.model,
                    "provider": "OpenAI",
                },
                "highlights": settings.highlights.model_dump(),
                "error": None,
            }
        except ValueError:
            return {
                "llm": {"configured": False, "model": "", "provider": "OpenAI"},
                "highlights": HighlightOptions().model_dump(),
                "error": "Check config.toml: invalid TOML, unknown settings, or invalid highlight limits.",
            }

    @app.post("/api/jobs/youtube", status_code=202)
    def create_youtube(body: YouTubeInput):
        language = options(body.model, body.language)
        try:
            url = youtube_url(body.url)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        job, _ = new_job("youtube", "YouTube video", body.model, language, source_url=url)
        return enqueue(job)

    @app.post("/api/jobs/upload", status_code=202)
    async def create_upload(request: Request):
        # Multipart parser streams uploaded files to disk instead of loading whole videos into memory.
        async with request.form(max_files=1, max_fields=3) as form:
            upload = form.get("file")
            if not upload or not hasattr(upload, "read"):
                raise HTTPException(422, "Select a video file to upload.")
            upload: UploadFile
            model = str(form.get("model", "whisper-1"))
            language = options(model, str(form.get("language", "")))
            if upload.size is not None and upload.size > MAX_UPLOAD:
                raise HTTPException(413, "Video exceeds the configured upload size limit.")
            filename = Path(upload.filename or "video").name
            suffix = Path(filename).suffix.lower()
            if suffix not in VIDEO_SUFFIXES:
                raise HTTPException(422, "Choose an MP4, MOV, MKV, WebM, AVI, M4V, MPEG, or MTS video.")
            job, folder = new_job("upload", filename, model, language, source_file=f"source{suffix}")
            try:
                with (folder / job["source_file"]).open("wb") as dest:
                    size = 0
                    while chunk := await upload.read(1024 * 1024):
                        size += len(chunk)
                        if size > MAX_UPLOAD:
                            raise HTTPException(413, "Video exceeds the upload size limit.")
                        await asyncio.to_thread(dest.write, chunk)
                if not size:
                    raise HTTPException(422, "This file is empty.")
            except BaseException:
                shutil.rmtree(folder, ignore_errors=True)
                raise
        return enqueue(job)

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        return get_job(job_id)

    @app.get("/api/jobs/{job_id}/storage")
    def video_storage(job_id: str):
        get_job(job_id)
        folder = store.root / job_id
        total = 0
        for path in folder.rglob("*"):
            if path.is_file() and not path.is_symlink():
                try:
                    total += path.stat().st_size
                except FileNotFoundError:
                    pass
        return {"size_bytes": total}

    @app.delete("/api/jobs/{job_id}")
    @serialized
    def delete_video(job_id: str):
        job = get_job(job_id)
        if job["status"] in {"queued", "running"} or any(
            job.get(key) in {"queued", "running"} for key in ("highlights_status", "clips_status")
        ):
            raise HTTPException(
                409, "Wait for this video's queued or running work to finish before deleting it."
            )
        size = video_storage(job_id)["size_bytes"]
        store.update(job_id, status="deleting")
        try:
            remove_files(job)
        except (OSError, ValueError):
            store.update(job_id, status="delete_failed", error="File cleanup failed. Try deletion again.")
            raise HTTPException(
                500, "Some files could not be removed. Try deleting the video again."
            ) from None
        return {"deleted": True, "size_bytes": size}

    @app.post("/api/jobs/{job_id}/retry", status_code=202)
    @serialized
    def retry(job_id: str, body: RetryInput | None = None):
        job = get_job(job_id)
        if job["status"] != "failed":
            raise HTTPException(409, "Only failed jobs can be retried.")
        model = body.model if body and body.model is not None else "whisper-1"
        options(model, job.get("language", ""))
        job = store.update(
            job_id,
            expected_status="failed",
            model=model,
            status="queued",
            stage="acquire",
            progress=0,
            error=None,
            message="Queued for retry…",
        )
        if job is None:
            raise HTTPException(409, "This job has already been retried.")
        app.state.executor.submit(pipeline.run, job_id)
        return job

    @app.get("/api/jobs/{job_id}/transcript")
    def transcript(job_id: str):
        job = get_job(job_id)
        if job["status"] != "completed":
            raise HTTPException(409, "The transcript is not ready yet.")
        import json

        return json.loads((store.root / job_id / "transcript.json").read_text())

    @app.get("/api/jobs/{job_id}/media")
    def media(job_id: str):
        job = get_job(job_id)
        filename = job.get("preview_file") or job.get("source_file")
        if not filename or not (store.root / job_id / filename).is_file():
            raise HTTPException(404, "Video is not available yet.")
        return FileResponse(store.root / job_id / filename)

    @app.get("/api/jobs/{job_id}/export/{kind}")
    def export(job_id: str, kind: str):
        if kind not in {"json", "txt", "srt", "html"}:
            raise HTTPException(404, "Unknown export format.")
        data = transcript(job_id)
        content_types = {
            "json": "application/json",
            "txt": "text/plain",
            "srt": "application/x-subrip",
            "html": "text/html",
        }
        return Response(
            export_transcript(data, kind),
            media_type=content_types[kind],
            headers={"Content-Disposition": f'attachment; filename="transcript-{job_id[:8]}.{kind}"'},
        )

    @app.post("/api/jobs/{job_id}/highlights", status_code=202)
    @serialized
    def generate_highlights(job_id: str, body: HighlightOptions | None = None):
        job = get_job(job_id)
        if job["status"] != "completed":
            raise HTTPException(409, "Finish transcription before selecting highlights.")
        if job.get("highlights_status") in {"queued", "running"}:
            raise HTTPException(409, "Highlight selection is already in progress.")
        if job.get("clips_status") in {"queued", "running"}:
            raise HTTPException(409, "Wait for clip rendering to finish before regenerating highlights.")
        try:
            settings = highlight_runner.settings()
            options = HighlightOptions.model_validate(
                settings.highlights.model_dump() | (body.model_dump(exclude_unset=True) if body else {})
            ).check_range()
        except ValueError:
            raise HTTPException(422, "Check config.toml and your highlight duration limits.") from None
        if not settings.llm.api_key:
            raise HTTPException(422, "Add your OpenAI API key under [llm] in config.toml, then try again.")
        if not transcript(job_id)["segments"]:
            raise HTTPException(422, "No speech was detected, so there are no passages to select.")
        job = store.update(
            job_id,
            expected_fields={"highlights_status": job.get("highlights_status")},
            highlights_status="queued",
            highlights_progress=0,
            highlights_error=None,
            highlights_message="Queued for highlight selection…",
            highlights_options=options.model_dump(),
        )
        if job is None:
            raise HTTPException(409, "Highlight selection is already in progress.")
        app.state.llm_executor.submit(highlight_runner.run, job_id, settings, options)
        return job

    @app.get("/api/jobs/{job_id}/highlights")
    def highlights(job_id: str):
        job = get_job(job_id)
        if job.get("highlights_status") != "completed":
            raise HTTPException(409, "Highlights are not ready yet.")
        import json

        return json.loads((store.root / job_id / "highlights.json").read_text())

    @app.get("/api/jobs/{job_id}/highlights/export/{kind}")
    def highlight_export(job_id: str, kind: str):
        if kind not in {"json", "html"}:
            raise HTTPException(404, "Unknown highlight export format.")
        return Response(
            export_highlights(highlights(job_id), kind),
            media_type="application/json" if kind == "json" else "text/html",
            headers={"Content-Disposition": f'attachment; filename="highlights-{job_id[:8]}.{kind}"'},
        )

    @app.post("/api/jobs/{job_id}/clips", status_code=202)
    @serialized
    def generate_clips(job_id: str, options: ClipInput | None = None):
        job = get_job(job_id)
        if job["status"] != "completed" or job.get("highlights_status") != "completed":
            raise HTTPException(409, "Finish transcription and highlight selection before generating clips.")
        if job.get("clips_status") in {"queued", "running"}:
            raise HTTPException(409, "Clip rendering is already in progress.")
        if not highlights(job_id)["highlights"]:
            raise HTTPException(422, "No highlights are available to render.")
        if not job.get("source_file") or not (store.root / job_id / job["source_file"]).is_file():
            raise HTTPException(409, "The original video is missing. Upload or download it again.")
        job = store.update(
            job_id,
            clips_status="queued",
            clips_progress=0,
            clips_error=None,
            clips_message="Queued for clip rendering…",
            clips_mode=options.mode if options else "original",
        )
        app.state.clip_executor.submit(clip_runner.run, job_id)
        return job

    @app.get("/api/jobs/{job_id}/clips")
    def clips(job_id: str):
        job = get_job(job_id)
        if job.get("clips_status") != "completed":
            raise HTTPException(409, "Clips are not ready yet. Generate clips for the current highlights.")
        import json

        data = json.loads((store.root / job_id / "clips.json").read_text())
        if data["highlights_sha256"] != highlight_hash(highlights(job_id)):
            raise HTTPException(409, "Highlights have changed. Generate clips again.")
        return data

    @app.get("/api/jobs/{job_id}/clips/{clip_id}/media")
    def clip_media(job_id: str, clip_id: str, download: bool = False, revision: str | None = None):
        data = clips(job_id)
        if revision is not None and revision != data["revision"]:
            raise HTTPException(409, "This clip belongs to an older render. Reload the video workspace.")
        item = next((c for c in data["clips"] if c["id"] == clip_id), None)
        if not item:
            raise HTTPException(404, "Clip not found.")
        path = store.root / job_id / data["directory"] / item["filename"]
        if not path.is_file():
            raise HTTPException(404, "Clip file is missing. Generate clips again.")
        return FileResponse(
            path,
            media_type="video/mp4",
            filename=f"clip-{item['rank']:02}-{clip_id}.mp4" if download else None,
        )

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="frontend")
    return app


app = create_app()
