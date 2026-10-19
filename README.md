# Frame — Transcript Studio

A local web app that takes a video upload or YouTube URL through acquisition, audio extraction, and timestamped Whisper transcription, with an optional OpenAI pass to select and rank important passages, followed by local MP4 clip generation.

## Run

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 or 3.13. uv can install the Python runtime if needed. YouTube downloading also needs a supported JavaScript runtime; Node.js 22 or newer is recommended.

```sh
uv sync
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. Keep the terminal running while jobs process. Use one server process; the queue is designed for a single local worker.

The FFmpeg executable is bundled through `imageio-ffmpeg`, so a system FFmpeg installation is optional. An existing FFmpeg on PATH takes precedence. Transcription runs on the CPU using faster-whisper with int8 computation. No API key is required. On first use, the selected model and voice-activity detector are downloaded from their upstream model hosts. Models remain in the standard Hugging Face cache for later runs. Start with Tiny to test quickly, or Base for a balanced default; larger models need substantially more memory and time.

## Use

1. Upload a video or paste a single YouTube video/Short URL.
2. Choose a Whisper model and auto-detect or select the spoken language.
3. Create the transcript and follow job progress. Multiple videos queue sequentially.
4. Click a transcript timestamp to seek and play the video. Search passages or enable Follow playback.
5. Export plain text, SRT subtitles, structured JSON with word timestamps, or a standalone readable HTML report.

Original media is retained. Cropping, resizing, translation, and social publishing are outside this version. LLM highlight selection is optional and sends transcript text to OpenAI only when you click Generate highlights. Unsupported browser video containers/codecs get an H.264/AAC MP4 preview at the same dimensions. Rare odd-sized sources use H.264 4:4:4 to preserve dimensions, which some browsers cannot play; their transcript exports remain useful. For dependable playback, use even-sized H.264/AAC MP4 input.

Uploads default to 2 GB and all videos default to a three-hour duration limit. YouTube metadata is checked before download; live/upcoming streams are rejected. The best available video/audio streams are downloaded and merged without changing dimensions. YouTube may reject requests because of availability, restrictions, or anti-bot checks; the failure is shown in the UI. Cookie-based sign-in is not implemented. Keep yt-dlp current when YouTube changes:

```sh
uv lock --upgrade-package yt-dlp --upgrade-package yt-dlp-ejs
uv sync
```


## Select highlights with an LLM

The actual `config.toml` is already created locally and excluded from Git. It is never served to the browser. A shareable template is in `config.example.toml`.

1. Add an OpenAI API key to `api_key` in the `[llm]` section of `config.toml`. `OPENAI_API_KEY` in the server environment takes precedence when set.
2. Keep `model = "gpt-5.6-terra"`, or choose another OpenAI model supporting the Responses API and Structured Outputs. API usage requires billing/access separate from a ChatGPT subscription.
3. Open any completed video. Under **Find the important parts**, click **Check config** (the app also checks periodically).
4. Adjust the maximum number of highlights, minimum/maximum duration, and editorial goal, then click **Generate highlights**. No restart or retranscription is required.
5. Read the ranked passages, selection reasons, and exact source text; preview each passage or export a standalone HTML report / structured JSON.

By default, the pass selects up to five continuous passages of 20–90 seconds. Short videos may produce zero highlights; lower the minimum for those. Fewer passages are allowed when quality or duration rules cannot be met. Whisper word timings are grouped into sentences across segment boundaries, so selections preserve sentence openings and negations. Sentences longer than the maximum duration cannot be selected on their own; increase the maximum for those. Older transcripts without word timings use whole segments. Timestamps remain estimates inherited from transcription.

Only transcript text, language, and selection instructions are sent to OpenAI. Media stays local. The SDK uses Structured Outputs with `store=False`. Long transcripts are read in bounded, overlapping windows and candidates are ranked across the video. Selections are validated against actual unit IDs and durations, with one targeted repair attempt for semantic errors. Overlapping passages are removed. The model never supplies the saved text or timestamps: those are resolved from the local transcript.

Each successful selection is stored as `data/<job-id>/highlights.json`, with the source transcript hash, exact passages, reasons, ranks, settings, model name, and token/request usage. Scores are subjective editorial assessments. Regenerating replaces the current highlight result only after success. The original transcript stays available when selection fails. Interrupted LLM jobs are marked as failed on restart and require an explicit retry, rather than automatically repeating potentially billed requests. SDK transient-error retries are enabled.

Additional configuration: `APP_CONFIG_FILE` selects a different TOML path. Configuration is read on every selection request; `max_input_chars` controls transcript window size and `timeout_seconds` sets each API request's timeout. Do not commit `config.toml` or paste API keys into the frontend.

Additional routes:

| Route | Purpose |
| --- | --- |
| `GET /api/settings` | Model, configured flag, and selection defaults; never returns the key |
| `POST /api/jobs/{id}/highlights` | Queue selection; optional JSON overrides for `max_count`, `min_seconds`, `max_seconds`, `objective` |
| `GET /api/jobs/{id}/highlights` | Completed ranked passages |
| `GET /api/jobs/{id}/highlights/export/{format}` | `html` or `json` attachment |

Highlight tests exercise the actual OpenAI SDK against an HTTP mock, validation/repair, long-transcript ranking, exports, key redaction, config reload, empty results, and restart handling. A live pass with GPT-5.6 Terra was also verified on a 53-minute Spanish video: five non-overlapping 20–90-second passages were selected, checked against source units and timestamps, and exported to HTML/JSON. Browser checks covered generation, results, preview seeking, and stopping at passage boundaries using an isolated fixture.

## Generate and download video clips

After selecting highlights, click **Generate clips**. Each passage becomes an H.264/AAC MP4 cut from the original source video, with its width, height, and aspect ratio preserved. There is no crop, resize, caption overlay, or platform reframing. Cuts are re-encoded for accurate start times rather than limited to existing keyframes. Odd-sized inputs use yuv444p to preserve their exact dimensions; browser support for that pixel format varies.

The frontend shows rendering progress, then adds an embedded player and **Download clip** link to each passage. Metadata shows duration, dimensions, and file size. The original video remains available. Clips are rendered in a separate single-worker queue, and this stage makes no LLM calls.

Rendered batches are stored in `data/<job-id>/clips-<revision>/` with a `clips.json` manifest. The batch is published only after all outputs pass duration, dimensions, audio-track, and decodable-frame checks. Successful regeneration removes previous batches. Failed and interrupted work cleans up partial output and can be retried. Regenerating highlights marks existing clips stale; generate clips again before downloading them. Highlight regeneration is blocked while clips render.

## Delete videos and reclaim storage

Select a video and click **Delete video** beside Export transcript. The dialog shows the video title and stored file size, describes what will be removed, and lets you keep the video or permanently delete it.

Deletion removes that video's entire directory: original media, playback preview, temporary audio, transcripts, highlight reports, generated clips, and unfinished outputs. It also removes the job from SQLite and the sidebar. Shared Whisper model downloads and other videos are retained. The app releases its media players before deletion.

Deletion is disabled while that video has queued/running transcription, highlight selection, or clipping. The backend also enforces this restriction and serializes deletion against starting new work. If filesystem cleanup fails, the record remains available for another deletion attempt. Startup retries incomplete cleanup after an interrupted deletion.

Additional routes:

| Route | Purpose |
| --- | --- |
| `POST /api/jobs/{id}/clips` | Queue MP4 rendering for all current highlights |
| `GET /api/jobs/{id}/clips` | Current rendered batch and file metadata |
| `GET /api/jobs/{id}/clips/{clip_id}/media` | Play a clip, including range requests; `?download=true` downloads it |
| `GET /api/jobs/{id}/storage` | Current stored file size for a video |
| `DELETE /api/jobs/{id}` | Permanently remove an idle video and all its files |

## Persistence and configuration

Transcription, highlight selection, and clipping use separate single-worker queues, so the transcript stays available during selection and rendering. Job metadata is in `data/jobs.sqlite3`. Each video has its own `data/<job-id>/` directory containing its source, optional playback preview, and `transcript.json`. The temporary WAV is deleted after successful transcription. Interrupted transcription jobs are queued again at startup; retries reuse a completed download when available. Files remain on disk until you use Delete video; there is no automatic expiration.

Optional environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `DATA_DIR` | `data` | Persistent job/media directory |
| `MAX_UPLOAD_BYTES` | `2147483648` | Maximum accepted uploaded file size |
| `MAX_VIDEO_SECONDS` | `10800` | Maximum video duration, uploads and YouTube |

This is a local, single-user application with no authentication. The default command binds to loopback. The UI uses Google Fonts when reachable, with system font fallbacks.

## Structure

- `app/main.py`: FastAPI routes, uploads, worker lifecycle, media range requests, and exports.
- `app/store.py`: SQLite metadata store with synchronized updates.
- `app/config.py`: Validated, hot-reloaded server-only TOML configuration.
- `app/highlights.py`: Timestamped passage construction, structured LLM selection, validation, ranking, and HTML/JSON exports.
- `app/clips.py`: FFmpeg clipping from the original source, output validation, complete-batch publishing, and old-batch cleanup.
- `app/pipeline.py`: YouTube acquisition, media inspection, FFmpeg audio/preview generation, Whisper, and export serialization.
- `app/static/`: Responsive vanilla HTML/CSS/JavaScript frontend. No frontend build step.
- `tests/`: API/pipeline integration tests with real FFmpeg media and a stubbed transcription model, plus validation/export/recovery tests.

The worker writes a canonical transcript with stable segment IDs, source-video-relative segment/word timestamps, detected language, model name, and available probability signals. Highlight selection consumes this saved transcript without rerunning ingestion or transcription.

API routes:

| Route | Purpose |
| --- | --- |
| `GET /api/health` | Runtime capabilities and limits |
| `GET /api/jobs` | Persistent video library |
| `POST /api/jobs/upload` | Multipart `file`, `model`, `language` |
| `POST /api/jobs/youtube` | JSON `url`, `model`, `language` |
| `GET /api/jobs/{id}` | Job status/progress |
| `POST /api/jobs/{id}/retry` | Retry a failed job |
| `GET /api/jobs/{id}/transcript` | Completed structured transcript |
| `GET /api/jobs/{id}/media` | Source or browser preview, including range requests |
| `GET /api/jobs/{id}/export/{format}` | `txt`, `srt`, `json`, `html` attachment |

## Verify

```sh
uv run pytest -q
uv run ruff check app tests
node --check app/static/app.js
```

Automated tests don't download model weights or call YouTube. They exercise real extraction, background execution, media playback responses, exports, failures, retries, and restart recovery using controlled fixtures. A real transcription smoke check requires network access to obtain a model on its first run.

During implementation, both a real uploaded speech sample and a real YouTube download were transcribed with the Tiny model. Browser checks confirmed search, timestamp seeking, playback, and unchanged media dimensions. PyAV is constrained below version 19 because the released faster-whisper 1.2.1 still calls an audio-decoder argument removed in PyAV 19; the decoder regression test covers that compatibility boundary.
