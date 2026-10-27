# Frame — Transcript Studio

A local web app that takes a video upload or YouTube URL through acquisition, audio extraction, and timestamped Whisper transcription, with an optional OpenAI pass to select and rank important passages, followed by local MP4 clip generation.

## Run

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 or 3.13. uv can install the Python runtime if needed. YouTube downloading also needs a supported JavaScript runtime; Node.js 22 or newer is recommended.

```sh
uv sync
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. Keep the terminal running while jobs process. Use one server process; the queue is designed for a single local worker.

The FFmpeg executable is bundled through `imageio-ffmpeg`, so a system FFmpeg installation is optional. An existing FFmpeg on PATH takes precedence. Transcription always uses **OpenAI Whisper API** (`whisper-1`). Passage selection defaults to **GPT-5.5**, and both API stages use the same OpenAI API key. No local transcription models or model-weight downloads are used. FFmpeg renders the selected passages as MP4 clips locally.

## Use

1. Upload a video or paste a single YouTube video/Short URL.
2. Select or auto-detect the spoken language. Transcription uses OpenAI Whisper API.
3. Create the transcript and follow job progress. Multiple videos queue sequentially.
4. Click a transcript timestamp to seek and play the video. Search passages or enable Follow playback.
5. Export plain text, SRT subtitles, structured JSON with word timestamps, or a standalone readable HTML report.

Original media is retained. Clip exports support original dimensions or vertical 9:16 framing. Translation and social publishing are outside this version. Whisper API sends extracted audio to OpenAI when you create a transcript with the app. LLM highlight selection is optional and sends transcript text to OpenAI when you click Generate highlights. Unsupported browser video containers/codecs get an H.264/AAC MP4 preview at the same dimensions. Rare odd-sized sources use H.264 4:4:4 to preserve dimensions, which some browsers cannot play; their transcript exports remain useful. For dependable playback, use even-sized H.264/AAC MP4 input.

Uploads default to 5 GB and all videos default to a three-hour duration limit. YouTube metadata is checked before download; live/upcoming streams are rejected. The best available video/audio streams are downloaded and merged without changing dimensions. YouTube may reject requests because of availability, restrictions, or anti-bot checks; the failure is shown in the UI. Cookie-based sign-in is not implemented. Keep yt-dlp current when YouTube changes:

```sh
uv lock --upgrade-package yt-dlp --upgrade-package yt-dlp-ejs
uv sync
```


## Transcribe with Whisper API

1. Put your OpenAI API key in `[llm]` → `api_key` in the local `config.toml`. Alternatively set `OPENAI_API_KEY` in the server environment. Both transcription and highlight selection use this credential. Do not put the key in the browser or commit it.
2. Restart the server after installing this code update, then refresh the browser. Later key edits need no restart.
3. Upload a video or paste a YouTube URL, and click **Create transcript**. The transcription model is fixed to **Whisper API · OpenAI**. API usage is billed by OpenAI.

The app uploads mono 16 kHz PCM audio in parts of at most ten minutes (about 19.2 MB each), below the 25 MB file limit. It requests segment and word timestamps and shifts each part's times back onto the source video's timeline. Words at a part boundary may lose sentence context. Transcript exports, highlight selection, and clips work with these timestamps. API responses do not provide word confidence or language probability; those values are stored as null. The reported language may be a full language name.

Failed requests show safe errors for key, quota, or connection issues. An interrupted API job requires an explicit retry after server restart, as retrying can repeat paid requests. Within a request, SDK transient retries remain enabled. Temporary upload parts are cleaned up on completion or failure; a completed transcript also removes the extracted WAV. The **Retry with Whisper API** button also switches older failed local-model jobs to OpenAI, reusing their video file. Completed historical transcripts remain available. All upload and YouTube endpoints default to `whisper-1`; requesting a local model is rejected. Pending jobs, including legacy local jobs, require an explicit retry after restart to avoid automatically repeating paid API calls.

OpenAI currently lists `whisper-1` for shutdown on February 26, 2027; see the [speech-to-text guide](https://developers.openai.com/api/docs/guides/speech-to-text). This implementation uses Whisper as requested for its word and segment timestamps.

The [OpenAI transcription API reference](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create) documents `whisper-1`, `verbose_json`, and timestamp options.

## Upload large video files

The interface accepts videos up to **5 GB** by default (`5 × 1024³` bytes). `MAX_UPLOAD_BYTES` changes the server limit and the interface reads that value from `/api/health`. Uploads use sequential 8 MB requests and retry a failed part up to three times. Each attempt has a two-minute timeout; progress shows bytes acknowledged by the server. Retrying an identical part cannot duplicate bytes. Completing an upload twice returns the same job and never queues it twice.

The processing queue receives a video only after its declared byte count has arrived. Failed uploads are discarded when the client can reach the server to cancel them. Interrupted sessions after a browser/server crash can leave temporary files in `data/.uploads/`; these sessions do not appear in the video library. Retrying from a new page starts a fresh upload. The existing single multipart `/api/jobs/upload` endpoint remains available to API clients with the same 5 GB limit. Allow free disk space for the video, audio, preview, and generated clips; the three-hour duration limit still applies.

## Select highlights with an LLM

The actual `config.toml` is already created locally and excluded from Git. It is never served to the browser. A shareable template is in `config.example.toml`.

1. Add an OpenAI API key to `api_key` in the `[llm]` section of `config.toml`. `OPENAI_API_KEY` in the server environment takes precedence when set.
2. Keep `model = "gpt-5.5"`, or choose another OpenAI model supporting the Responses API and Structured Outputs. [GPT-5.5 supports both](https://developers.openai.com/api/docs/models/gpt-5.5). API usage requires billing/access separate from a ChatGPT subscription.
3. Open any completed video. Under **Find the important parts**, click **Check config** (the app also checks periodically).
4. Adjust the maximum number of highlights, minimum/maximum duration, and editorial goal, then click **Generate highlights**. No restart or retranscription is required.
5. Read the ranked passages, selection reasons, and exact source text; preview each passage or export a standalone HTML report / structured JSON.

By default, the pass selects up to five continuous passages of 20–90 seconds. Short videos may produce zero highlights; lower the minimum for those. Fewer passages are allowed when quality or duration rules cannot be met. Sentence units align the punctuation in Whisper segment text with existing word timestamps. This also works with saved Whisper API transcripts whose word entries contain no punctuation, without retranscribing or changing the saved transcript. Exact text alignment preserves word boundaries; when segment text and words differ, the entire segment is retained with its source timestamps instead of guessing an internal cut. Sentence continuations are grouped across segment boundaries, preserving sentence openings and negations. Sentences longer than the maximum duration cannot be selected on their own; increase the maximum for those. Older transcripts without word timings use whole segments. Timestamps remain estimates inherited from transcription.

For highlight selection, only transcript text, language, and selection instructions are sent to OpenAI. Whisper API transcription separately sends extracted audio to OpenAI. The SDK uses Structured Outputs with `store=False`. Long transcripts are read in bounded, overlapping windows and candidates are ranked across the video. Selections are validated against actual unit IDs and durations, with one targeted repair attempt for semantic errors. Overlapping passages are removed. The model never supplies the saved text or timestamps: those are resolved from the local transcript.

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

After selecting highlights, choose an **Export format**, then click **Generate clips**. Each passage becomes an H.264/AAC MP4 cut from the original source video. Available formats:

- **Vertical · Face follow (9:16)** is the UI default. A bundled OpenCV YuNet detector analyzes only the selected ranges, approximately eight times per second. Nearby face detections are associated to keep following the same person. This does not identify the active speaker.
- **Vertical · Center crop (9:16)** keeps the crop fixed at the center and requires no face detection.
- **Original dimensions** preserves the source dimensions. This remains the API default when no options are supplied. Odd-sized originals use yuv444p; browser support for that pixel format varies.

Face follow prioritizes a steady frame: median filtering removes detection noise; shots with small movements use a fixed crop; the camera responds quickly at the start of each shot, then its response and speed/acceleration limits decay exponentially over 2.5 seconds. The dead zone grows from 6% to 16% of the crop to ignore small movements once the framing has settled. Each layout uses a fixed crop size, preventing zoom wobble. Short detection gaps hold the framing, and longer gaps ease toward the center. Spatial scene-cut detection immediately reframes at detected edits, restarting the 2.5-second settling period. When two substantial, separated faces persist for at least 0.25 seconds, only that interval uses a stacked layout (left participant above, right participant below). Brief detection misses are bridged to avoid layout flicker; scene cuts end the previous layout immediately. Other intervals keep a single full-height frame. This does not identify who is speaking or analyze lip movements. Large subject movements, occlusions, and subtle cuts can still need manual review.

Vertical clips use the largest exact 9:16 crop with even dimensions, with output capped at 1080 × 1920 and no upscaling or stretching. Smaller sources produce smaller exports. No-face shots use the centered crop, and the clip preview identifies exports where no faces were found. The model and MIT license are bundled in `app/assets/`; no runtime model download or external face-tracking service is required. Run `uv sync` after updating to install OpenCV.

Cuts are re-encoded for accurate start times rather than limited to existing keyframes. Social publishing is not included.

Enable the **Dynamic subtitles** toggle before generating clips to burn cumulative phrases into the MP4. Words appear in sync with the voice and stay visible until the block resets. Blocks end at sentence punctuation or pauses of at least 450 ms; commas, semicolons, and colons can end a block after five words, and every block is capped at eight words. Punctuation is recovered locally from segment text only when it matches the timed words exactly; otherwise grouping uses word count and pauses. No AI calls are needed. It defaults to off. Captions use white bold text with a dark outline, positioned at 74% of the output height, with a subtle 98% → 100% scale entrance on each new word over up to 80 ms. Earlier words remain at full size. Text uses a uniform font size (84 px at 1080 × 1920). Longer phrases break into shorter blocks of up to two lines rather than shrinking the font; only unusually long individual words shrink to fit. Each block reserves its full layout, with unrevealed words invisible. They are applied after the vertical crop or stacked layout. Saved Whisper word timestamps are reused without new API calls. Each clip gets its own relative timing; captions stop at clip boundaries and preserve long pauses. Completed blocks can remain for up to 150 ms for readability, capped by the next block.

Zero-duration words are interpolated within nearby gaps or by borrowing up to 120 ms from each adjacent word (at most half that neighbor's duration). Consecutive zero-duration words share the interval. This is a display approximation, not a new audio alignment; the original transcript stays unchanged. Old transcripts without usable word timestamps fail with a clear message when subtitles are enabled. The FFmpeg build must include libass (the bundled macOS build does). Long words shrink to fit. Generated ASS files remain in the rendered batch for inspection. Existing clips must be regenerated to add or remove subtitles.

The frontend shows rendering progress, then adds an embedded player and **Download clip** link to each passage. Metadata shows duration, dimensions, and file size. The original video remains available. Clips are rendered in a separate single-worker queue, and this stage makes no LLM calls.

Each clip also has a **Regenerate this clip** button. Generate the full batch first, then rebuild an individual passage as needed. Individual regeneration uses the published batch's format and subtitle settings; the controls above apply to full-batch generation. It renders only the selected clip and reuses the other files with local hard links. Previous players and downloads remain available during rendering. A new manifest is published atomically only after validation; failures and server restarts preserve the last published clips and show a retryable error. Unchanged clips keep their revisions and playback URLs. Individual regeneration makes no API calls. Clips generated with subtitles also have an **Edit subtitles** button. The editor shows the subtitle text as a continuous paragraph beside the video preview. Click a word to edit it inline; corrected words are highlighted. Selecting a word reveals its source spelling and lets you play from its timestamp. **Save corrections** stores the draft without changing the MP4; **Save & regenerate this clip** applies it using individual regeneration. Pending corrections are labeled on the clip and persist across reloads. **Restore original text** restores the editor fields; save and regenerate to apply the restoration.

Corrections are stored per clip in `caption-edits.json`; source transcript text, timing, and highlight selection remain unchanged. Normal spelling changes retain each word's interval. A corrected field may contain multiple words, which share that interval evenly; empty words are removed from the rendered subtitles. A small × keeps their place in the editor for restoration; it is not rendered. A removed word’s interval is assigned to a nearby surviving neighbor (within 450 ms), preserving longer pauses. At least one subtitle word must remain. Full-batch regeneration also uses saved corrections. Corrections are tied to the source words and clip range, so regenerated highlights or a changed transcript cannot accidentally inherit unrelated corrections. Conflicting editor saves are rejected instead of silently overwriting changes. Editing is blocked during rendering or selection; saved drafts survive failed renders. No OpenAI requests are made by editing or regeneration.

Rendered batches are stored in `data/<job-id>/clips-<revision>/` with a `clips.json` manifest. The batch is published only after all outputs pass duration, dimensions, audio-track, and decodable-frame checks. Successful regeneration removes previous batches. Failed and interrupted work cleans up partial output and can be retried. Regenerating highlights marks existing clips stale; generate clips again before downloading them. Highlight regeneration is blocked while clips render.

## Delete videos and reclaim storage

Select a video and click **Delete video** beside Export transcript. The dialog shows the video title and stored file size, describes what will be removed, and lets you keep the video or permanently delete it.

Deletion removes that video's entire directory: original media, playback preview, temporary audio, transcripts, highlight reports, generated clips, and unfinished outputs. It also removes the job from SQLite and the sidebar. Other videos are retained. The app releases its media players before deletion.

Deletion is disabled while that video has queued/running transcription, highlight selection, or clipping. The backend also enforces this restriction and serializes deletion against starting new work. If filesystem cleanup fails, the record remains available for another deletion attempt. Startup retries incomplete cleanup after an interrupted deletion.

Additional routes:

| Route | Purpose |
| --- | --- |
| `POST /api/jobs/{id}/clips` | Queue MP4 rendering; optional JSON `{"mode":"original"}`, `{"mode":"center_crop"}`, or `{"mode":"face_follow","subtitles":true}`; `subtitles` defaults to `false` |
| `GET /api/jobs/{id}/clips/{clip_id}/subtitles` | Timed spelling editor with saved corrections and pending status |
| `PUT /api/jobs/{id}/clips/{clip_id}/subtitles` | Save `revision` and corrected `texts` without re-rendering |
| `POST /api/jobs/{id}/clips/{clip_id}/regenerate` | Regenerate a single existing clip using the published batch settings |
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
| `MAX_UPLOAD_BYTES` | `5368709120` | Maximum accepted uploaded file size |
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
node --test tests/upload-client.test.cjs
```

Automated tests make no live OpenAI or YouTube requests. They exercise real extraction, background execution, media playback, exports, uploads, failures, retries, and restart recovery using controlled fixtures. SDK tests verify actual HTTP requests against a mock, including Whisper word/segment timestamps and GPT-5.5 structured selection. Existing completed transcripts created before the OpenAI-only change remain readable.
