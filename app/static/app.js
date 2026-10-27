const $ = (id) => document.getElementById(id);
const state = {
  source: "upload",
  file: null,
  jobs: [],
  active: null,
  transcript: null,
  rendered: null,
  uploading: false,
};
let maxUploadBytes = 5 * 1024 ** 3;
api("/api/health").then((health) => {
  maxUploadBytes = health.max_upload_bytes;
  $("upload-limit").textContent = `${(maxUploadBytes / 1024 ** 3).toFixed(0)} GB`;
}).catch(() => {});
let pollTimer;
let loadVersion = 0;
const steps = ["acquire", "audio", "transcribe", "complete"];

function formatTime(seconds, precise = false) {
  const n = Math.max(0, Math.floor(seconds || 0));
  const parts = [Math.floor(n / 3600), Math.floor(n / 60) % 60, n % 60];
  return (parts[0] || precise ? parts : parts.slice(1))
    .map((v) => String(v).padStart(2, "0"))
    .join(":");
}
function showError(message) {
  $("form-error").textContent = message;
  $("form-error").hidden = false;
}
function notify(message) {
  $("toast").textContent = message;
  $("toast").hidden = false;
  clearTimeout(notify.timer);
  notify.timer = setTimeout(() => {
    $("toast").hidden = true;
  }, 4500);
}
async function api(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message;
    try {
      const body = await response.json();
      message =
        typeof body.detail === "string"
          ? body.detail
          : "Please check your input and try again.";
    } catch {
      message = `Request failed (${response.status}).`;
    }
    throw new Error(message);
  }
  return response.json();
}
function setSource(source) {
  state.source = source;
  for (const tab of ["upload", "youtube"]) {
    $(`tab-${tab}`).setAttribute("aria-selected", String(tab === source));
    $(`tab-${tab}`).tabIndex = tab === source ? 0 : -1;
    $(`panel-${tab}`).hidden = tab !== source;
  }
  $("form-error").hidden = true;
}
for (const source of ["upload", "youtube"]) {
  $(`tab-${source}`).addEventListener("click", () => setSource(source));
  $(`tab-${source}`).addEventListener("keydown", (e) => {
    if (["ArrowLeft", "ArrowRight"].includes(e.key)) {
      e.preventDefault();
      const next = source === "upload" ? "youtube" : "upload";
      setSource(next);
      $(`tab-${next}`).focus();
    }
  });
}
function chooseFile(file) {
  if (!file) return;
  if (file.size > maxUploadBytes) {
    showError(`This video exceeds the ${(maxUploadBytes / 1024 ** 3).toFixed(1)} GB upload limit.`);
    return;
  }
  if (!/\.(mp4|mov|mkv|webm|avi|m4v|mpeg|mpg|mts)$/i.test(file.name)) {
    showError("Please choose a supported video file.");
    return;
  }
  state.file = file;
  $("file-title").textContent = file.name;
  $("file-description").textContent =
    `${(file.size / 1024 ** 2).toFixed(1)} MB · Click to choose another`;
  $("form-error").hidden = true;
}
$("video-file").addEventListener("change", (e) =>
  chooseFile(e.target.files[0]),
);
$("dropzone").addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    $("video-file").click();
  }
});
for (const event of ["dragenter", "dragover"])
  $("dropzone").addEventListener(event, (e) => {
    e.preventDefault();
    $("dropzone").classList.add("dragover");
  });
for (const event of ["dragleave", "drop"])
  $("dropzone").addEventListener(event, (e) => {
    e.preventDefault();
    $("dropzone").classList.remove("dragover");
  });
$("dropzone").addEventListener("drop", (e) =>
  chooseFile(e.dataTransfer.files[0]),
);

function updateWorkflow(stage = "acquire", completed = false) {
  const index = steps.indexOf(stage === "model" ? "transcribe" : stage);
  for (const [i, key] of steps.entries()) {
    const el = document.querySelector(`[data-step="${key}"]`);
    el.classList.toggle("active", i === index && !completed);
    el.classList.toggle("done", i < index || completed);
    el.querySelector(".step-number").textContent =
      i < index || completed ? "✓" : String(i + 1);
  }
}
function newTranscript() {
  if (state.uploading) {
    notify("Please wait for the upload to finish.");
    return;
  }
  state.active = null;
  state.transcript = null;
  state.rendered = null;
  loadVersion++;
  localStorage.removeItem("frame-active-job");
  $("video-player").pause();
  $("video-player").removeAttribute("src");
  $("video-player").load();
  $("import-view").hidden = false;
  $("workspace-view").hidden = true;
  $("form-error").hidden = true;
  updateWorkflow();
  renderLibrary();
}
$("new-video").addEventListener("click", newTranscript);

let librarySignature = "";
function renderLibrary() {
  const signature = JSON.stringify([
    state.active,
    state.jobs.map((j) => [j.id, j.title, j.status]),
  ]);
  if (signature === librarySignature) return;
  librarySignature = signature;
  $("library-count").textContent = state.jobs.length;
  const list = $("job-list");
  list.replaceChildren();
  if (!state.jobs.length) {
    const empty = document.createElement("p");
    empty.className = "library-empty";
    empty.textContent =
      "Your videos will appear here. Start with something worth hearing.";
    list.append(empty);
    return;
  }
  for (const job of state.jobs) {
    const button = document.createElement("button");
    button.className = `job-link${state.active === job.id ? " selected" : ""}`;
    button.setAttribute("aria-label", `${job.title}, ${job.status}`);
    const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    use.setAttribute("href", "#i-video");
    icon.append(use);
    const details = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = job.title;
    const description = document.createElement("small");
    description.textContent = `${job.source_type === "youtube" ? "YouTube" : "Upload"} · ${job.status === "completed" ? "Ready" : job.status[0].toUpperCase() + job.status.slice(1)}`;
    details.append(title, description);
    const dot = document.createElement("span");
    dot.className = `job-state ${job.status}`;
    button.append(icon, details, dot);
    button.addEventListener("click", () => selectJob(job.id));
    list.append(button);
  }
}
const emptyMarkup = $("transcript-segments").innerHTML;
function selectJob(id) {
  if (state.uploading) {
    notify("Please wait for the upload to finish.");
    return;
  }
  resetHighlights();
  resetClips();
  state.active = id;
  state.transcript = null;
  state.rendered = null;
  loadVersion++;
  localStorage.setItem("frame-active-job", id);
  $("import-view").hidden = true;
  $("workspace-view").hidden = false;
  $("transcript-search").value = "";
  $("transcript-search").disabled = true;
  $("search-count").textContent = "";
  $("segment-count").textContent = "";
  $("language-badge").textContent = "";
  $("transcript-segments").innerHTML = emptyMarkup;
  $("video-player").pause();
  $("video-player").removeAttribute("src");
  $("video-player").load();
  $("video-player").hidden = true;
  $("media-placeholder").hidden = false;
  const job = state.jobs.find((j) => j.id === id);
  if (job) {
    if (job.highlights_options) {
      fillHighlightOptions(job.highlights_options);
      highlightFormDirty = true;
    }
    renderJob(job);
  }
  renderLibrary();
  refreshStorage(id);
}
function renderJob(job) {
  if (job.id !== state.active) return;
  renderClipState(job);
  renderDeleteState(job);
  renderHighlightState(job);
  $("video-title").textContent = job.title;
  $("source-label").textContent =
    job.source_type === "youtube" ? "FROM YOUTUBE" : "UPLOADED VIDEO";
  const meta = [
    job.model.charAt(0).toUpperCase() + job.model.slice(1) + " model",
  ];
  if (job.metadata)
    meta.push(
      formatTime(job.metadata.duration, true),
      `${job.metadata.width} × ${job.metadata.height}`,
    );
  if (job.channel) meta.push(job.channel);
  $("video-meta").textContent = meta.join(" · ");
  $("resolution-label").textContent = job.metadata
    ? `${job.metadata.width} × ${job.metadata.height}`
    : "";
  updateWorkflow(job.stage, job.status === "completed");
  const failed = job.status === "failed";
  const completed = job.status === "completed";
  $("progress-card").hidden = completed;
  $("progress-card").classList.toggle("failed", failed);
  $("processing-spinner").hidden = failed;
  const labels = {
    acquire:
      job.source_type === "youtube"
        ? "Downloading your video"
        : "Preparing your video",
    audio: "Preparing audio & video",
    model: "Loading the Whisper model",
    transcribe: "Turning audio into words",
  };
  $("progress-title").textContent = failed
    ? "This video needs another try"
    : job.status === "queued"
      ? "Waiting in the queue"
      : labels[job.stage] || "Processing";
  $("progress-percent").textContent = failed ? "" : `${job.progress}%`;
  $("progress-fill").style.width = `${job.progress}%`;
  $("progress-track").setAttribute("aria-valuenow", job.progress);
  $("progress-message").textContent = job.message;
  $("job-error").hidden = !failed;
  $("job-error").textContent = job.error || "";
  $("retry-button").hidden = !failed;
  $("export-menu").hidden = !completed;
  if (completed && state.rendered !== job.id) loadTranscript(job);
  if (
    (completed || job.preview_file) &&
    !$("video-player").getAttribute("src")
  ) {
    $("video-player").src = `/api/jobs/${job.id}/media`;
    $("video-player").hidden = false;
    $("media-placeholder").hidden = true;
  }
}
async function loadTranscript(job) {
  state.rendered = job.id;
  const version = loadVersion;
  try {
    const transcript = await api(`/api/jobs/${job.id}/transcript`);
    if (version !== loadVersion) return;
    state.transcript = transcript;
    $("transcript-search").disabled = false;
    $("segment-count").textContent = `${transcript.segments.length} segments`;
    let language = transcript.language;
    try {
      language = new Intl.DisplayNames(["en"], { type: "language" }).of(
        language,
      );
    } catch {
      /* Keep language code. */
    }
    $("language-badge").textContent = language;
    for (const link of document.querySelectorAll("[data-format]"))
      link.href = `/api/jobs/${job.id}/export/${link.dataset.format}`;
    renderSegments();
  } catch (e) {
    if (version === loadVersion) {
      state.rendered = null;
      notify(e.message);
    }
  }
}
function highlightText(text, query) {
  const fragment = document.createDocumentFragment();
  if (!query) {
    fragment.append(document.createTextNode(text));
    return fragment;
  }
  let from = 0;
  let index;
  while ((index = text.toLocaleLowerCase().indexOf(query, from)) !== -1) {
    fragment.append(document.createTextNode(text.slice(from, index)));
    const mark = document.createElement("mark");
    mark.textContent = text.slice(index, index + query.length);
    fragment.append(mark);
    from = index + query.length;
  }
  fragment.append(document.createTextNode(text.slice(from)));
  return fragment;
}
function renderSegments() {
  if (!state.transcript) return;
  const query = $("transcript-search").value.trim().toLocaleLowerCase();
  const segments = state.transcript.segments.filter(
    (s) => !query || s.text.toLocaleLowerCase().includes(query),
  );
  $("search-count").textContent = query ? `${segments.length} found` : "";
  const container = $("transcript-segments");
  container.replaceChildren();
  if (!segments.length) {
    const empty = document.createElement("div");
    empty.className = "no-matches";
    empty.textContent = query
      ? "No matching passages. Try another word."
      : "No speech was detected in this video.";
    container.append(empty);
    return;
  }
  const fragment = document.createDocumentFragment();
  for (const segment of segments) {
    const article = document.createElement("article");
    article.className = "transcript-segment";
    article.dataset.segmentId = segment.id;
    const time = document.createElement("button");
    time.className = "segment-time";
    time.textContent = formatTime(segment.start);
    time.title = `${formatTime(segment.start, true)} – ${formatTime(segment.end, true)}`;
    time.setAttribute(
      "aria-label",
      `Play from ${formatTime(segment.start, true)}`,
    );
    time.addEventListener("click", () => {
      const video = $("video-player");
      video.currentTime = segment.start;
      video
        .play()
        .catch(() => notify("Press play on the video to start playback."));
    });
    const text = document.createElement("p");
    text.className = "segment-text";
    text.append(highlightText(segment.text, query));
    article.append(time, text);
    fragment.append(article);
  }
  container.append(fragment);
  markHighlightedSegments();
  updatePlayback();
}
$("transcript-search").addEventListener("input", renderSegments);
let activeSegment;
function updatePlayback() {
  if (!state.transcript) return;
  const now = $("video-player").currentTime;
  const segment = state.transcript.segments.find(
    (s) => s.start <= now && s.end > now,
  );
  const id = segment?.id;
  for (const row of document.querySelectorAll(".transcript-segment"))
    row.classList.toggle("active", row.dataset.segmentId === id);
  if (id !== activeSegment && $("follow-playback").checked) {
    const row = document.querySelector(`[data-segment-id="${id}"]`);
    if (row)
      $("transcript-segments").scrollTo({
        top: row.offsetTop - $("transcript-segments").offsetTop - 50,
        behavior: "smooth",
      });
  }
  activeSegment = id;
}
$("video-player").addEventListener("timeupdate", updatePlayback);
$("video-player").addEventListener("error", () =>
  notify(
    "The video preview could not load. Your transcript and exports are still available.",
  ),
);
async function retryTranscription() {
  $("retry-button").disabled = true;
  try {
    const job = await api(`/api/jobs/${state.active}/retry`, {
      method: "POST",
    });
    renderJob(job);
    await refreshJobs();
  } catch (e) {
    notify(e.message);
  } finally {
    $("retry-button").disabled = false;
  }
}
$("retry-button").addEventListener("click", () => retryTranscription());

async function uploadFile(file, model, language) {
  const session = await api("/api/uploads", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename: file.name, size: file.size, model, language }),
  });
  async function retryRequest(path, options) {
    for (let attempt = 0; ; attempt++) {
      try {
        const response = await fetch(path, { ...options, signal: AbortSignal.timeout(120000) });
        if (response.ok) return await response.json();
        if (response.status < 500) {
          const body = await response.json();
          const error = new Error(body.detail || "The upload failed.");
          error.permanent = true;
          throw error;
        }
        throw new Error("The server could not accept this part of the upload.");
      } catch (error) {
        if (error.permanent || attempt >= 3) throw error;
        $("submit-button").querySelector("span").textContent = "Connection interrupted · retrying this part…";
        await new Promise((resolve) => setTimeout(resolve, 1000 * 2 ** attempt));
      }
    }
  }
  try {
    let offset = 0;
    while (offset < file.size) {
      const end = Math.min(offset + session.chunk_bytes, file.size);
      const result = await retryRequest(`/api/uploads/${session.id}?offset=${offset}`, {
        method: "PUT",
        headers: { "Content-Type": "application/octet-stream" },
        body: file.slice(offset, end),
      });
      if (result.offset !== end) throw new Error("Upload position mismatch. Please try again.");
      offset = result.offset;
      $("submit-button").querySelector("span").textContent =
        `Uploading video · ${Math.round(offset / file.size * 100)}%`;
    }
    $("submit-button").querySelector("span").textContent = "Upload complete · preparing your video…";
    const job = await retryRequest(`/api/uploads/${session.id}/complete`, { method: "POST" });
    $("submit-button").querySelector("span").textContent = "Upload complete";
    return job;
  } catch (error) {
    await fetch(`/api/uploads/${session.id}`, { method: "DELETE", signal: AbortSignal.timeout(10000) }).catch(() => {});
    throw error;
  }
}
$("import-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (state.uploading) return;
  $("form-error").hidden = true;
  if (state.source === "upload" && !state.file) {
    showError("Choose a video file first.");
    return;
  }
  if (state.source === "youtube" && !$("youtube-url").value.trim()) {
    showError("Paste a YouTube video URL first.");
    return;
  }
  const model = $("model").value;
  const language = $("language").value;
  state.uploading = true;
  $("submit-button").disabled = true;
  $("submit-button").querySelector("span").textContent =
    state.source === "youtube" ? "Creating your job…" : "Uploading video…";
  try {
    let job;
    if (state.source === "youtube") {
      job = await api("/api/jobs/youtube", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          url: $("youtube-url").value.trim(),
          model,
          language,
        }),
      });
    } else {
      job = await uploadFile(state.file, model, language);
    }
    state.uploading = false;
    state.jobs = [job, ...state.jobs.filter((j) => j.id !== job.id)];
    selectJob(job.id);
    await refreshJobs();
  } catch (e) {
    showError(e.message);
  } finally {
    state.uploading = false;
    $("submit-button").disabled = false;
    $("submit-button").querySelector("span").textContent = "Create transcript";
  }
});
async function refreshJobs() {
  try {
    state.jobs = await api("/api/jobs");
    renderLibrary();
    const job = state.jobs.find((j) => j.id === state.active);
    if (job) renderJob(job);
    else if (state.active) {
      newTranscript();
      notify("This video was removed from the workspace.");
    }
  } catch {
    notify("Cannot reach the local server. Checking again shortly…");
  }
}
async function poll() {
  await refreshJobs();
  if (Date.now() - settingsLastRead > 10000) await refreshLLMSettings();
  pollTimer = setTimeout(
    poll,
    state.jobs.some(
      (j) =>
        ["queued", "running"].includes(j.status) ||
        ["queued", "running"].includes(j.highlights_status) ||
        ["queued", "running"].includes(j.clips_status),
    )
      ? 1500
      : 5000,
  );
}
for (let i = 0; i < 54; i++) {
  const bar = document.createElement("i");
  bar.style.setProperty(
    "--height",
    `${8 + Math.abs(Math.sin(i * 0.63) * Math.sin(i * 0.19 + 1)) * 72}px`,
  );
  $("wave-bars").append(bar);
}
(async () => {
  await refreshJobs();
  const saved = localStorage.getItem("frame-active-job");
  if (saved && state.jobs.some((j) => j.id === saved)) selectJob(saved);
  else updateWorkflow();
  pollTimer = setTimeout(poll, 1500);
})();

// Highlight selection is a separate job: the transcript remains available throughout.
let llmSettings = null;
let highlightData = null;
let highlightRequest = null;
let settingsLastRead = 0;
let highlightFormDirty = false;
let highlightedSegmentIds = new Set();
let highlightEnd = null;

function resetHighlights() {
  highlightData = null;
  highlightRequest = null;
  highlightedSegmentIds = new Set();
  highlightEnd = null;
  $("highlight-results").hidden = true;
  $("highlight-error").hidden = true;
  $("highlight-progress").hidden = true;
}
function fillHighlightOptions(options) {
  $("highlight-count").value = options.max_count;
  $("highlight-min").value = options.min_seconds;
  $("highlight-max").value = options.max_seconds;
  $("highlight-objective").value = options.objective;
}
async function refreshLLMSettings() {
  try {
    llmSettings = await api("/api/settings");
    settingsLastRead = Date.now();
    if (!highlightFormDirty) fillHighlightOptions(llmSettings.highlights);
    const notice = $("llm-config-notice");
    notice.replaceChildren();
    notice.classList.toggle("ready", llmSettings.llm.configured);
    const text = document.createElement("span");
    text.textContent =
      llmSettings.error ||
      (llmSettings.llm.configured
        ? "API key configured. Ready to select passages with OpenAI."
        : "Add your OpenAI API key under [llm] in config.toml, then check the configuration.");
    const check = document.createElement("button");
    check.type = "button";
    check.textContent = "Check config";
    check.addEventListener("click", async () => {
      check.disabled = true;
      await refreshLLMSettings();
    });
    notice.append(text, check);
    $("llm-model-badge").textContent =
      llmSettings.llm.model || "Configuration needed";
    const job = state.jobs.find((j) => j.id === state.active);
    if (job) renderHighlightState(job);
  } catch {
    $("llm-config-notice").textContent =
      "Could not read LLM settings. Check the local server.";
  }
}
function renderHighlightState(job) {
  if (job.id !== state.active) return;
  $("highlights-section").hidden = job.status !== "completed";
  if (job.status !== "completed") return;
  const busy = ["queued", "running"].includes(job.highlights_status);
  const renderingClips = ["queued", "running"].includes(job.clips_status);
  for (const field of $("highlights-form").querySelectorAll("input, textarea"))
    field.disabled = busy;
  $("generate-highlights").disabled =
    busy || renderingClips || !llmSettings?.llm.configured;
  const buttonText = busy
    ? "Selecting passages…"
    : job.highlights_status === "completed"
      ? "Regenerate highlights"
      : job.highlights_status === "failed"
        ? "Try highlight selection again"
        : "Generate highlights";
  $("generate-highlights").firstChild.textContent = buttonText + " ";
  $("highlight-progress").hidden = !busy;
  $("highlight-progress-message").textContent =
    `${job.highlights_message || "Queued…"} ${busy ? `${job.highlights_progress || 0}%` : ""}`;
  $("highlight-error").hidden = job.highlights_status !== "failed";
  $("highlight-error").textContent = job.highlights_error || "";
  if (
    job.highlights_status === "completed" &&
    !highlightData &&
    highlightRequest !== job.id
  )
    loadHighlights(job.id);
}
async function loadHighlights(jobId) {
  const version = loadVersion;
  highlightRequest = jobId;
  try {
    const data = await api(`/api/jobs/${jobId}/highlights`);
    if (version !== loadVersion) return;
    highlightData = data;
    renderHighlights(data);
  } catch (e) {
    if (version === loadVersion) notify(e.message);
  } finally {
    if (version === loadVersion) highlightRequest = null;
  }
}
function renderHighlights(data) {
  $("highlight-results").hidden = false;
  $("highlight-results-title").textContent =
    `${data.highlights.length} selected passage${data.highlights.length === 1 ? "" : "s"}`;
  $("highlight-result-meta").textContent =
    `${data.model} · ranked by editorial importance · ${data.usage.requests} API request${data.usage.requests === 1 ? "" : "s"}`;
  $("highlight-summary").textContent = data.summary;
  $("highlight-export-html").href =
    `/api/jobs/${data.job_id}/highlights/export/html`;
  $("highlight-export-json").href =
    `/api/jobs/${data.job_id}/highlights/export/json`;
  const list = $("highlight-cards");
  list.replaceChildren();
  highlightedSegmentIds = new Set(
    data.highlights.flatMap((h) => h.segment_ids),
  );
  if (!data.highlights.length) {
    const empty = document.createElement("p");
    empty.className = "empty-highlights";
    empty.textContent =
      "No suitable passages met these limits. Try a shorter minimum duration or adjust what to look for.";
    list.append(empty);
  }
  for (const highlight of data.highlights) {
    const card = document.createElement("article");
    card.className = "highlight-card";
    card.dataset.highlightId = highlight.id;
    const rank = document.createElement("span");
    rank.className = "highlight-rank";
    rank.textContent = `PASSAGE ${String(highlight.rank).padStart(2, "0")}`;
    const title = document.createElement("h3");
    title.textContent = highlight.title;
    const timing = document.createElement("p");
    timing.className = "highlight-timing";
    timing.textContent = `${formatTime(highlight.start, true)} → ${formatTime(highlight.end, true)} · ${Math.round(highlight.duration)}s`;
    const reason = document.createElement("p");
    reason.className = "highlight-reason";
    reason.textContent = highlight.reason;
    const text = document.createElement("blockquote");
    text.textContent = highlight.text;
    const play = document.createElement("button");
    play.type = "button";
    play.className = "secondary-button";
    play.textContent = "Play this passage →";
    play.setAttribute(
      "aria-label",
      `Play passage ${highlight.rank}: ${highlight.title}`,
    );
    play.addEventListener("click", () => {
      const video = $("video-player");
      highlightEnd = highlight.end;
      video.currentTime = highlight.start;
      video
        .play()
        .catch(() =>
          notify("Press play on the video to preview this passage."),
        );
      video.scrollIntoView({ block: "center", behavior: "smooth" });
    });
    card.append(rank, title, timing, reason, text, play);
    list.append(card);
  }
  markHighlightedSegments();
  attachClipPreviews();
  const current = state.jobs.find((j) => j.id === state.active);
  if (current) renderClipState(current);
}
function markHighlightedSegments() {
  for (const row of document.querySelectorAll(".transcript-segment"))
    row.classList.toggle(
      "selected-highlight",
      highlightedSegmentIds.has(row.dataset.segmentId),
    );
}
$("video-player").addEventListener("timeupdate", () => {
  if (highlightEnd !== null && $("video-player").currentTime >= highlightEnd) {
    $("video-player").pause();
    highlightEnd = null;
  }
});
// Manual scrubbing or seeking to a transcript timestamp resumes normal playback.
$("video-player").addEventListener("seeking", () => {
  if (highlightEnd !== null && $("video-player").currentTime >= highlightEnd)
    highlightEnd = null;
});
$("highlights-form").addEventListener("input", () => {
  highlightFormDirty = true;
});
$("highlights-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const id = state.active;
  const version = loadVersion;
  const options = {
    max_count: Number($("highlight-count").value),
    min_seconds: Number($("highlight-min").value),
    max_seconds: Number($("highlight-max").value),
    objective: $("highlight-objective").value.trim(),
  };
  if (options.min_seconds > options.max_seconds) {
    $("highlight-error").textContent =
      "Minimum duration must not exceed maximum duration.";
    $("highlight-error").hidden = false;
    return;
  }
  $("generate-highlights").disabled = true;
  try {
    const job = await api(`/api/jobs/${id}/highlights`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(options),
    });
    if (version !== loadVersion) return;
    highlightData = null;
    resetClips();
    $("highlight-results").hidden = true;
    state.jobs = state.jobs.map((j) => (j.id === id ? job : j));
    renderHighlightState(job);
  } catch (e) {
    if (version === loadVersion) {
      $("highlight-error").textContent = e.message;
      $("highlight-error").hidden = false;
      $("generate-highlights").disabled = !llmSettings?.llm.configured;
    }
  }
});
refreshLLMSettings();

let clipData = null;
let clipLoadKey = null;
let storageSize = 0;
let deleteTarget = null;

function bytesLabel(bytes) {
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  let size = bytes;
  while (size >= 1024 && i < units.length - 1) {
    size /= 1024;
    i++;
  }
  return `${size.toFixed(i > 1 ? 1 : 0)} ${units[i]}`;
}
function jobBusy(job) {
  return (
    ["queued", "running", "deleting"].includes(job.status) ||
    ["queued", "running"].includes(job.highlights_status) ||
    ["queued", "running"].includes(job.clips_status)
  );
}
function renderDeleteState(job) {
  $("delete-video").disabled = jobBusy(job);
  $("delete-availability").textContent = jobBusy(job)
    ? "Deletion is available when this video’s queued or running work finishes."
    : "";
}
async function refreshStorage(id) {
  const version = loadVersion;
  $("video-storage").textContent = "";
  try {
    const data = await api(`/api/jobs/${id}/storage`);
    if (version !== loadVersion) return;
    storageSize = data.size_bytes;
    $("video-storage").textContent =
      `${bytesLabel(storageSize)} stored for this video`;
  } catch {
    /* The video might have been removed in another tab. */
  }
}
function resetClips() {
  clipData = null;
  clipLoadKey = null;
  $("clip-error").hidden = true;
  $("clip-progress").hidden = true;
  $("clip-ready-summary").hidden = true;
  for (const element of document.querySelectorAll(
    ".clip-preview, .clip-download, .clip-file-info",
  )) {
    if (element.tagName === "VIDEO") {
      element.pause();
      element.removeAttribute("src");
      element.load();
    }
    element.remove();
  }
}
function renderClipState(job) {
  if (job.id !== state.active) return;
  const busy = ["queued", "running"].includes(job.clips_status);
  $("clip-mode").disabled = busy;
  if ($("clip-mode").dataset.job !== job.id) {
    $("clip-mode").dataset.job = job.id;
    $("clip-mode").value = job.clips_mode || "face_follow";
    updateClipFormatHelp();
  }
  $("generate-clips").disabled =
    busy ||
    job.highlights_status !== "completed" ||
    !highlightData?.highlights.length;
  $("generate-clips").firstChild.textContent = busy
    ? "Rendering clips… "
    : job.clips_status === "completed"
      ? "Regenerate clips "
      : "Generate clips ";
  $("clip-progress").hidden = !busy;
  $("clip-progress-message").textContent =
    `${job.clips_message || "Queued…"} ${busy ? `${job.clips_progress || 0}%` : ""}`;
  $("clip-error").hidden = job.clips_status !== "failed";
  $("clip-error").textContent = job.clips_error || "";
  if (job.clips_status === "stale" || job.highlights_status !== "completed") {
    if (clipData) resetClips();
    return;
  }
  const key = `${job.id}:${job.clips_revision || ""}`;
  if (job.clips_status === "completed" && clipLoadKey !== key)
    loadClips(job, key);
}
async function loadClips(job, key) {
  const version = loadVersion;
  clipLoadKey = key;
  try {
    const data = await api(`/api/jobs/${job.id}/clips`);
    if (version !== loadVersion) return;
    clipData = data;
    attachClipPreviews();
    refreshStorage(job.id);
  } catch (e) {
    if (version === loadVersion) {
      $("clip-error").textContent = e.message;
      $("clip-error").hidden = false;
    }
  }
}
function attachClipPreviews() {
  if (!clipData || !highlightData || clipData.job_id !== state.active) return;
  $("clip-ready-summary").hidden = false;
  const total = clipData.clips.reduce((sum, item) => sum + item.size_bytes, 0);
  const format = { original: "original dimensions", center_crop: "vertical · center crop", face_follow: "vertical · face follow" }[clipData.mode || "original"];
  $("clip-ready-summary").textContent =
    `${clipData.clips.length} MP4 clips ready · ${bytesLabel(total)} · ${format}`;
  for (const item of clipData.clips) {
    const card = document.querySelector(`[data-highlight-id="${item.id}"]`);
    if (!card || card.querySelector(".clip-preview")) continue;
    const video = document.createElement("video");
    video.className = "clip-preview";
    if (item.height > item.width) video.classList.add("vertical");
    video.style.aspectRatio = `${item.width} / ${item.height}`;
    video.controls = true;
    video.preload = "metadata";
    video.setAttribute("aria-label", `Clip ${item.rank}: ${item.title}`);
    const url = `/api/jobs/${clipData.job_id}/clips/${item.id}/media`;
    video.src = `${url}?revision=${clipData.revision}`;
    video.addEventListener("error", () =>
      notify(
        "This clip could not play. Try downloading it or generating clips again.",
      ),
    );
    const info = document.createElement("p");
    info.className = "clip-file-info";
    info.textContent = `${item.width} × ${item.height} · ${formatTime(item.duration)} · ${bytesLabel(item.size_bytes)}`;
    if (item.framing?.fallback === "center_crop") info.textContent += " · No face found; centered crop";
    const download = document.createElement("a");
    download.className = "clip-download";
    download.href = `${url}?download=true&revision=${clipData.revision}`;
    download.textContent = `Download clip ${item.rank} · MP4`;
    download.setAttribute("download", "");
    card.append(video, info, download);
  }
}
function updateClipFormatHelp() {
  $("clip-format-help").textContent = {
    face_follow: "Gently follows one person; stacks two people only while both are on screen.",
    center_crop: "A fixed 9:16 crop from the center of the video.",
    original: "Keep the source dimensions and framing.",
  }[$("clip-mode").value];
}
$("clip-mode").addEventListener("change", updateClipFormatHelp);
$("generate-clips").addEventListener("click", async () => {
  const id = state.active,
    version = loadVersion;
  $("generate-clips").disabled = true;
  try {
    const job = await api(`/api/jobs/${id}/clips`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: $("clip-mode").value }),
    });
    if (version !== loadVersion) return;
    resetClips();
    state.jobs = state.jobs.map((j) => (j.id === id ? job : j));
    renderJob(job);
  } catch (e) {
    if (version === loadVersion) {
      $("clip-error").textContent = e.message;
      $("clip-error").hidden = false;
      $("generate-clips").disabled = false;
    }
  }
});
$("delete-video").addEventListener("click", async () => {
  const job = state.jobs.find((j) => j.id === state.active);
  if (!job || jobBusy(job)) return;
  deleteTarget = job.id;
  $("delete-dialog-name").textContent = job.title;
  $("delete-dialog-size").textContent = storageSize
    ? `Stored files: ${bytesLabel(storageSize)}`
    : "";
  $("delete-error").hidden = true;
  $("confirm-delete").disabled = false;
  $("cancel-delete").disabled = false;
  $("delete-dialog").showModal();
  try {
    const data = await api(`/api/jobs/${job.id}/storage`);
    if (deleteTarget === job.id)
      $("delete-dialog-size").textContent =
        `Stored files: ${bytesLabel(data.size_bytes)}`;
  } catch {
    /* Backend handles missing jobs. */
  }
});
$("cancel-delete").addEventListener("click", () => {
  $("delete-dialog").close();
  deleteTarget = null;
});
$("delete-dialog").addEventListener("cancel", (e) => {
  if ($("confirm-delete").disabled) e.preventDefault();
});
$("confirm-delete").addEventListener("click", async () => {
  const id = deleteTarget;
  if (!id) return;
  $("confirm-delete").disabled = true;
  $("cancel-delete").disabled = true;
  // Release media readers so file storage can be reclaimed immediately.
  for (const video of document.querySelectorAll("video")) {
    video.pause();
    video.removeAttribute("src");
    video.load();
  }
  try {
    const response = await api(`/api/jobs/${id}`, { method: "DELETE" });
    $("delete-dialog").close();
    deleteTarget = null;
    state.jobs = state.jobs.filter((j) => j.id !== id);
    if (state.active === id) newTranscript();
    renderLibrary();
    notify(
      `Video and files deleted · ${bytesLabel(response.size_bytes)} removed.`,
    );
  } catch (e) {
    $("delete-error").textContent = e.message;
    $("delete-error").hidden = false;
    $("confirm-delete").disabled = false;
    $("cancel-delete").disabled = false;
    // Restore the current workspace if deletion was rejected, e.g. another tab started a job.
    if (state.active === id) {
      selectJob(id);
    }
  }
});
