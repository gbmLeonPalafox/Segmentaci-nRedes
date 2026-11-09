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
  highlightFormDirty = false;
  if (llmSettings) fillHighlightOptions(llmSettings.highlights);
  highlightFormDirty = false;
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
let promptCatalog = [];

let promptVariant = null;
let personalizationHistory = [];
let personalizationBusy = false;
let personalizationRequest = 0;
let pendingHighlightOptions = null;

function selectedPrompt() {
  return promptCatalog.find((p) => p.id === $("clip-area").value);
}
function selectedRevision() {
  return selectedPrompt()?.revisions.find((r) => r.version === Number($("prompt-version").value));
}
function adaptationInputs() {
  return {
    area: $("clip-area").value,
    master_version: Number($("prompt-version").value) || null,
    audience: $("audience-context").value.trim(),
    min_seconds: Number($("highlight-min").value),
    max_seconds: Number($("highlight-max").value),
    prior_learning: $("prior-learning").value.trim(),
    channel: $("audience-channel").value.trim(),
    campaign_context: $("campaign-context").value.trim(),
  };
}
function inputFingerprint() { return JSON.stringify(adaptationInputs()); }
function metricsText(metrics) {
  const labels = {views: "Reproducciones", interactions: "Interacciones", inquiries: "Consultas"};
  return Object.entries(metrics || {}).map(([key, value]) => `${labels[key] || key}: ${value}`).join(" · ") || "Sin métricas";
}
function invalidatePersonalization() {
  pendingHighlightOptions = null;
  personalizationRequest++;
  promptVariant = null;
  $("personalization-history").value = "";
  $("prompt-explanation").replaceChildren();
  $("prompt-evidence").replaceChildren();
  $("prompt-status").textContent = "Se usará el maestro con el contexto de audiencia. Puedes proponer o guardar una adaptación.";
  highlightFormDirty = true;
}
function fillVersions(version = null) {
  const profile = selectedPrompt();
  $("prompt-version").replaceChildren(...(profile?.revisions || []).map((r) => new Option(`v${r.version}`, r.version)));
  if (profile) $("prompt-version").value = version || profile.revisions.at(-1).version;
}
function applyPrompt(resetDuration = true) {
  const revision = selectedRevision();
  invalidatePersonalization();
  if (!revision) {
    $("prompt-status").textContent = "Cargando el prompt maestro…";
    return;
  }
  $("master-prompt").textContent = revision.objective;
  $("highlight-objective").value = revision.objective;
  if (resetDuration) {
    $("highlight-min").value = revision.min_seconds;
    $("highlight-max").value = revision.max_seconds;
  }
}
function showVariant(variant) {
  promptVariant = variant;
  $("highlight-objective").value = variant.objective;
  $("master-prompt").textContent = variant.master.objective;
  $("prompt-status").textContent = variant.reviewed_at
    ? `Personalización v${variant.version} revisada y lista para usar.`
    : "Propuesta pendiente de revisión. Lee o edita el prompt y pulsa Guardar y usar.";
  if (selectedPrompt()?.revisions.at(-1).version > variant.master.version) {
    $("prompt-status").textContent += ` Usa el maestro anterior v${variant.master.version}; puedes generar una nueva adaptación con el maestro actual.`;
  }
  const explanation = $("prompt-explanation");
  explanation.replaceChildren();
  for (const [title, entries] of [["Cambios respecto al maestro", variant.changes], ["Supuestos y límites", variant.assumptions]]) {
    const heading = document.createElement("strong"); heading.textContent = title;
    const list = document.createElement("ul");
    for (const entry of entries || []) { const row = document.createElement("li"); row.textContent = entry; list.append(row); }
    explanation.append(heading, list);
  }
  const evidence = $("prompt-evidence"); evidence.replaceChildren();
  const heading = document.createElement("strong"); heading.textContent = "Antecedentes consultados"; evidence.append(heading);
  if (!variant.evidence.length) {
    const empty = document.createElement("p"); empty.textContent = "Sin resultados relevantes registrados. La propuesta se basa en el maestro y el contexto proporcionado."; evidence.append(empty);
  }
  for (const item of variant.evidence) {
    const row = document.createElement("p");
    const relevance = item.relevance === "same_audience" ? "Misma audiencia" : "Audiencia similar · referencia indirecta";
    const source = item.evidence_label === "synthetic_hypothesis" ? "Hipótesis sintética" : "Resultado real reportado";
    const outcomes = {successful: "Funcionó", unsuccessful: "No funcionó", mixed: "Mixto"};
    row.textContent = `${relevance} · ${source} · ${outcomes[item.evaluation]} · ${item.channel} · ${item.published_at}\n${item.clip.title}: ${item.feedback || "Sin comentario"}\n${metricsText(item.metrics)}`;
    evidence.append(row);
  }
}
async function loadPersonalizationHistory() {
  personalizationHistory = await api("/api/personalizations");
  $("personalization-history").replaceChildren(new Option("Selecciona una versión revisada", ""),
    ...personalizationHistory.filter((p) => p.reviewed_at).map((p) => new Option(
      `${p.area === "acquisition" ? "Adquisición" : "Activación"} · ${p.audience.slice(0, 55)} · v${p.version}`, p.id,
    )));
}
$("clip-area").addEventListener("change", () => { fillVersions(); applyPrompt(); });
$("prompt-version").addEventListener("change", () => applyPrompt(false));
for (const id of ["audience-context", "highlight-min", "highlight-max", "prior-learning", "audience-channel", "campaign-context"]) {
  $(id).addEventListener("input", () => {
    invalidatePersonalization();
    if (selectedRevision()) $("highlight-objective").value = selectedRevision().objective;
  });
}
$("highlight-objective").addEventListener("input", () => {
  if (promptVariant?.reviewed_at) promptVariant = null;
  $("prompt-status").textContent = "Hay cambios sin guardar. Revisa y guarda la personalización para usarlos.";
});
$("personalization-history").addEventListener("change", () => {
  const variant = personalizationHistory.find((p) => p.id === $("personalization-history").value);
  if (!variant) return;
  personalizationRequest++;
  $("clip-area").value = variant.area;
  fillVersions(variant.master.version);
  $("audience-context").value = variant.audience;
  $("highlight-min").value = variant.min_seconds;
  $("highlight-max").value = variant.max_seconds;
  $("prior-learning").value = variant.prior_learning;
  $("audience-channel").value = variant.channel;
  $("campaign-context").value = variant.campaign_context || "";
  showVariant(variant);
  highlightFormDirty = true;
});
$("adapt-prompt").addEventListener("click", async () => {
  if (!$("highlights-form").reportValidity()) return;
  const inputs = adaptationInputs();
  if (inputs.audience.length < 3) { notify("Describe la audiencia para adaptar el prompt."); return; }
  if (inputs.min_seconds > inputs.max_seconds) { notify("Revisa el rango de duración."); return; }
  const request = ++personalizationRequest, fingerprint = inputFingerprint(), jobId = state.active;
  personalizationBusy = true;
  $("adapt-prompt").disabled = true;
  $("prompt-status").textContent = "Preparando una propuesta con los antecedentes relevantes…";
  const job = state.jobs.find((j) => j.id === state.active); if (job) renderHighlightState(job);
  try {
    const draft = await api("/api/personalizations", {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(inputs),
    });
    if (request !== personalizationRequest || fingerprint !== inputFingerprint() || jobId !== state.active) return;
    fillVersions(draft.master.version);
    showVariant(draft);
  } catch (error) { if (request === personalizationRequest) { $("prompt-status").textContent = error.message; notify(error.message); } }
  finally {
    personalizationBusy = false;
    $("adapt-prompt").disabled = false;
    const job = state.jobs.find((j) => j.id === state.active); if (job) renderHighlightState(job);
  }
});
$("save-prompt").addEventListener("click", async () => {
  if (!$("highlights-form").reportValidity()) return;
  const inputs = adaptationInputs(), objective = $("highlight-objective").value.trim();
  if (inputs.audience.length < 3) { notify("Describe la audiencia para guardar una personalización."); return; }
  const fingerprint = inputFingerprint(), jobId = state.active;
  $("save-prompt").disabled = true;
  try {
    const saved = await api(promptVariant && !promptVariant.reviewed_at
      ? `/api/personalizations/${promptVariant.id}/approve` : "/api/personalizations/manual", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify(promptVariant && !promptVariant.reviewed_at ? {objective} : {...inputs, objective}),
    });
    if (fingerprint !== inputFingerprint() || jobId !== state.active) return;
    showVariant(saved);
    await loadPersonalizationHistory();
    $("personalization-history").value = saved.id;
    notify("Personalización guardada y lista para usar.");
  } catch (error) { notify(error.message); }
  finally { $("save-prompt").disabled = false; }
});
$("publish-master").addEventListener("click", async () => {
  if (!$("highlights-form").reportValidity() || !selectedRevision()) return;
  try {
    const saved = await api(`/api/prompts/${selectedPrompt().id}/revisions`, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({base_version: Number($("prompt-version").value), objective: $("highlight-objective").value.trim(),
        min_seconds: Number($("highlight-min").value), max_seconds: Number($("highlight-max").value)}),
    });
    promptCatalog = promptCatalog.map((p) => p.id === saved.id ? saved : p);
    fillVersions(); applyPrompt(false); notify("Nueva versión del maestro guardada.");
  } catch (error) { notify(error.message); }
});

function resetHighlights() {
  highlightData = null;
  highlightRequest = null;
  highlightedSegmentIds = new Set();
  highlightEnd = null;
  invalidatePersonalization();
  $("prompt-manager").open = false;
  $("prior-learning").value = "";
  $("audience-channel").value = "";
  $("campaign-context").value = "";
  $("highlight-results").hidden = true;
  $("highlight-error").hidden = true;
  $("highlight-progress").hidden = true;
}
function fillHighlightOptions(options) {
  if (!promptCatalog.length) {
    pendingHighlightOptions = options;
    $("clip-area").value = options.prompt_id?.startsWith("activation") ? "activation" : "acquisition";
    $("audience-context").value = options.audience_context || "";
    $("campaign-context").value = options.campaign_context || "";
    $("highlight-count").value = options.max_count;
    $("highlight-min").value = options.min_seconds;
    $("highlight-max").value = options.max_seconds;
    return;
  }
  const profile = promptCatalog.find((p) => p.id === options.prompt_id);
  const area = profile?.area || (options.prompt_id?.startsWith("activation") ? "activation" : "acquisition");
  $("clip-area").value = area;
  fillVersions(options.personalization_id && profile?.status === "master" ? options.prompt_version : null);
  applyPrompt();
  $("audience-context").value = options.audience_context || "";
  $("campaign-context").value = options.campaign_context || "";
  $("highlight-count").value = options.max_count;
  if (options.prompt_id) {
    $("highlight-min").value = options.min_seconds;
    $("highlight-max").value = options.max_seconds;
  }
  if (options.personalization_id) {
    const job = state.jobs.find((j) => j.id === state.active);
    const variant = job?.highlights_prompt?.personalization;
    if (variant) {
      $("prior-learning").value = variant.prior_learning;
      $("audience-channel").value = variant.channel;
      showVariant(variant);
    }
  }
}
async function refreshLLMSettings() {
  try {
    const firstCatalogRead = !promptCatalog.length;
    llmSettings = await api("/api/settings");
    try {
      promptCatalog = await api("/api/prompts");
    } catch {
      $("prompt-status").textContent = "No se pudo cargar el catálogo de perfiles. Reinicia el servidor y recarga la página; si continúa, revisa la conexión local.";
      throw new Error("Prompt catalog unavailable");
    }
    settingsLastRead = Date.now();
    if (pendingHighlightOptions) {
      const pending = pendingHighlightOptions;
      pendingHighlightOptions = null;
      fillHighlightOptions(pending);
    } else if (!highlightFormDirty) {
      fillHighlightOptions(llmSettings.highlights);
      highlightFormDirty = false;
    } else if (firstCatalogRead) {
      const job = state.jobs.find((j) => j.id === state.active);
      if (job?.highlights_options?.prompt_id && !$("audience-context").value) {
        fillHighlightOptions(job.highlights_options);
      } else if ($("clip-area").value) {
        fillVersions();
        applyPrompt();
      }
    }
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
  for (const field of $("highlights-form").querySelectorAll("input, textarea, select, button"))
    field.disabled = busy;
  $("prompt-version").disabled = busy || !selectedPrompt();
  $("adapt-prompt").disabled = busy || personalizationBusy || !llmSettings?.llm.configured;
  $("generate-highlights").disabled =
    busy || renderingClips || personalizationBusy || !selectedPrompt() || !llmSettings?.llm.configured;
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
    `${data.model}${data.prompt ? ` · ${data.prompt.name} v${data.prompt.version}` : ""} · ranked by editorial importance · ${data.usage.requests} API request${data.usage.requests === 1 ? "" : "s"}`;
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
  attachResultForms(data);
  attachClipPreviews();
  const current = state.jobs.find((j) => j.id === state.active);
  if (current) renderClipState(current);
}

async function attachResultForms(data) {
  const version = loadVersion;
  try {
    const context = await api(`/api/jobs/${data.job_id}/results`);
    if (version !== loadVersion || highlightData !== data || context.selection_created_at !== (data.created_at || null)) return;
    for (const clip of data.highlights) {
      const card = document.querySelector(`[data-highlight-id="${clip.id}"]`);
      if (!card) continue;
      card.querySelector(".clip-feedback")?.remove();
      const details = document.createElement("details"); details.className = "clip-feedback";
      const summary = document.createElement("summary"); summary.textContent = "Registrar publicación y resultado";
      const form = document.createElement("form");
      const fields = {};
      function field(name, labelText, type = "text", choices = null) {
        const label = document.createElement("label"); label.textContent = labelText;
        const input = document.createElement(choices ? "select" : type === "textarea" ? "textarea" : "input");
        if (choices) for (const [value, text] of choices) input.append(new Option(text, value));
        else if (type !== "textarea") input.type = type;
        input.name = name;
        if (type === "number") { input.min = "0"; input.step = "any"; }
        if (type === "text" || type === "textarea") input.maxLength = name === "channel" ? 100 : 4000;
        label.append(input); form.append(label); fields[name] = input;
        return input;
      }
      field("area", "Objetivo", "text", [["acquisition", "Adquisición · nuevos clientes"], ["activation", "Activación · mayor interacción"]]).value = data.prompt?.area || "acquisition";
      field("audience", "Audiencia de esta publicación").value = data.options.audience_context || "";
      field("channel", "Canal").value = data.prompt?.personalization?.channel || "";
      const today = new Date(); today.setMinutes(today.getMinutes() - today.getTimezoneOffset());
      field("published_at", "Fecha de publicación", "date").value = today.toISOString().slice(0, 10);
      for (const name of ["audience", "channel", "published_at"]) fields[name].required = true;
      fields.audience.minLength = 3;
      field("evaluation", "Resultado", "text", [["pending", "Publicado · aún sin evaluar"], ["successful", "Funcionó"], ["unsuccessful", "No funcionó"], ["mixed", "Resultado mixto"]]);
      field("evidence_source", "Origen del resultado", "text", [["user_report", "Observación del usuario"], ["measured", "Métricas registradas"], ["synthetic", "Hipótesis de audiencia sintética"]]);
      field("views", "Reproducciones (opcional)", "number");
      field("interactions", "Interacciones (opcional)", "number");
      field("inquiries", "Consultas de clientes (opcional)", "number");
      field("feedback", "¿Qué funcionó o qué falló? (opcional)", "textarea");
      const save = document.createElement("button"); save.type = "submit"; save.className = "secondary-button"; save.textContent = "Guardar resultado de esta publicación";
      const status = document.createElement("p"); status.className = "prompt-help"; status.setAttribute("role", "status");
      form.append(save, status);
      const history = document.createElement("div"); history.className = "prompt-help";
      const records = context.records.filter((r) => r.clip_id === clip.id && r.selection_revision === context.selection_revision);
      let editingResultId = null;
      function showResult(record) {
        const row = document.createElement("p");
        const labels = {pending: "Sin evaluar", successful: "Funcionó", unsuccessful: "No funcionó", mixed: "Mixto"};
        row.textContent = `${record.published_at} · ${record.channel} · ${record.audience} · ${labels[record.evaluation]}${record.evidence_source === "synthetic" ? " · Hipótesis sintética" : ""}\n${record.feedback || ""}\n${metricsText(record.metrics)}`;
        row.dataset.resultId = record.id;
        const edit = document.createElement("button"); edit.type = "button"; edit.className = "secondary-button";
        edit.textContent = "Editar o evaluar publicación";
        edit.addEventListener("click", () => {
          editingResultId = record.id;
          for (const key of ["area", "audience", "channel", "published_at", "evaluation", "evidence_source", "feedback"]) fields[key].value = record[key] || "";
          for (const key of ["views", "interactions", "inquiries"]) fields[key].value = record.metrics?.[key] ?? "";
          save.textContent = "Actualizar resultado de esta publicación";
        });
        row.append(edit);
        history.querySelector(`[data-result-id="${record.id}"]`)?.remove();
        history.prepend(row);
      }
      for (const record of records.filter((r) => r.stage !== "selected").reverse()) showResult(record);
      const count = document.createElement("p"); count.className = "prompt-help";
      count.textContent = "Selección guardada. Cada publicación puede tener su propio registro; seleccionar un clip no demuestra éxito.";
      const exportLink = document.createElement("a"); exportLink.href = `/api/jobs/${data.job_id}/results/export`; exportLink.textContent = "Exportar historial del video (JSON)";
      form.addEventListener("submit", async (event) => {
        event.preventDefault(); save.disabled = true;
        const metrics = {};
        for (const key of ["views", "interactions", "inquiries"]) if (fields[key].value !== "") metrics[key] = Number(fields[key].value);
        const body = {selection_revision: context.selection_revision, metrics};
        for (const key of ["area", "audience", "channel", "published_at", "evaluation", "evidence_source", "feedback"]) body[key] = fields[key].value.trim();
        body.stage = body.evaluation === "pending" ? "published" : "evaluated";
        try {
          const result = await api(`/api/jobs/${data.job_id}/clips/${clip.id}/results${editingResultId ? `/${editingResultId}` : ""}`, {
            method: editingResultId ? "PUT" : "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body),
          });
          if (version !== loadVersion || highlightData !== data) return;
          showResult(result); status.textContent = "Resultado guardado para futuras personalizaciones.";
          for (const key of ["feedback", "views", "interactions", "inquiries"]) fields[key].value = "";
          fields.evaluation.value = "pending";
          editingResultId = null; save.textContent = "Guardar resultado de esta publicación";
        } catch (error) { status.textContent = error.message; }
        finally { save.disabled = false; }
      });
      details.append(summary, count, history, form, exportLink); card.append(details);
    }
    const old = context.records.filter((r) => r.stage !== "selected" && r.selection_revision !== context.selection_revision);
    document.querySelector(".previous-results")?.remove();
    if (old.length) {
      const history = document.createElement("details"); history.className = "previous-results clip-feedback";
      const summary = document.createElement("summary"); summary.textContent = "Resultados de selecciones anteriores"; history.append(summary);
      for (const r of old) { const row = document.createElement("p"); row.textContent = `${r.clip.title} · ${r.audience} · ${r.channel} · ${r.published_at} · ${r.evaluation} · ${r.feedback}`; history.append(row); }
      $("highlight-cards").append(history);
    }
  } catch (error) { if (version === loadVersion) notify(`No se pudo cargar el historial de resultados: ${error.message}`); }
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
    prompt_id: selectedPrompt()?.id || null,
    prompt_version: selectedPrompt() ? Number($("prompt-version").value) : null,
    personalization_id: promptVariant?.reviewed_at ? promptVariant.id : null,
    audience_kind: "unspecified",
    campaign_context: $("campaign-context").value.trim(),
    audience_context: $("audience-context").value.trim(),
    max_count: Number($("highlight-count").value),
    min_seconds: Number($("highlight-min").value),
    max_seconds: Number($("highlight-max").value),
    objective: $("highlight-objective").value.trim(),
  };
  const revision = selectedRevision();
  if ((promptVariant && !promptVariant.reviewed_at) ||
      (!promptVariant && revision && options.objective !== revision.objective)) {
    $("prompt-manager").open = true;
    notify("Revisa y guarda la personalización antes de generar clips.");
    return;
  }
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
loadPersonalizationHistory().catch(() => {});

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
    ".clip-preview, .clip-download, .clip-file-info, .clip-regenerate, .clip-edit-subtitles, .clip-caption-note",
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
  $("clip-subtitles").disabled = busy;
  if ($("clip-mode").dataset.job !== job.id) {
    $("clip-mode").dataset.job = job.id;
    $("clip-mode").value = job.clips_mode || "face_follow";
    $("clip-subtitles").checked = job.clips_subtitles || false;
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
  $("clip-error").hidden = !job.clips_error;
  $("clip-error").textContent = job.clips_error || "";
  if (job.clips_status === "stale" || job.highlights_status !== "completed") {
    if (clipData) resetClips();
    return;
  }
  const key = `${job.id}:${job.clips_revision || ""}`;
  for (const button of document.querySelectorAll(".clip-regenerate")) {
    button.disabled = busy;
    button.textContent = busy && button.dataset.clipId === job.clips_target_id
      ? "Regenerating this clip…" : "Regenerate this clip";
  }
  for (const button of document.querySelectorAll(".clip-edit-subtitles")) button.disabled = busy;
  if ((job.clips_status === "completed" || (busy && job.clips_target_id)) && clipLoadKey !== key)
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
    `${clipData.clips.length} MP4 clips ready · ${bytesLabel(total)} · ${format} · ${clipData.subtitles ? "Dynamic subtitles" : "No subtitles"}`;
  for (const item of clipData.clips) {
    const card = document.querySelector(`[data-highlight-id="${item.id}"]`);
    if (!card) continue;
    const revision = item.revision || clipData.revision;
    const existing = card.querySelector(".clip-preview");
    let note = card.querySelector(".clip-caption-note");
    if (!note) { note = document.createElement("p"); note.className = "clip-caption-note"; card.append(note); }
    note.textContent = item.subtitle_edits_pending ? "Subtitle corrections saved. Regenerate this clip to apply them." : "";
    note.hidden = !item.subtitle_edits_pending;
    if (existing?.dataset.revision === revision) continue;
    for (const element of card.querySelectorAll(".clip-preview, .clip-file-info, .clip-download, .clip-regenerate, .clip-edit-subtitles")) {
      if (element.tagName === "VIDEO") { element.pause(); element.removeAttribute("src"); element.load(); }
      element.remove();
    }
    const video = document.createElement("video");
    video.className = "clip-preview";
    if (item.height > item.width) video.classList.add("vertical");
    video.style.aspectRatio = `${item.width} / ${item.height}`;
    video.controls = true;
    video.preload = "metadata";
    video.setAttribute("aria-label", `Clip ${item.rank}: ${item.title}`);
    const url = `/api/jobs/${clipData.job_id}/clips/${item.id}/media`;
    video.dataset.revision = revision;
    video.src = `${url}?revision=${revision}`;
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
    download.href = `${url}?download=true&revision=${revision}`;
    download.textContent = `Download clip ${item.rank} · MP4`;
    download.setAttribute("download", "");
    const regenerate = document.createElement("button");
    regenerate.type = "button";
    regenerate.className = "clip-regenerate";
    regenerate.dataset.clipId = item.id;
    regenerate.textContent = "Regenerate this clip";
    regenerate.title = "Reuses this batch’s format and subtitle settings. Other clips stay unchanged.";
    regenerate.disabled = ["queued", "running"].includes(state.jobs.find(j => j.id === state.active)?.clips_status);
    regenerate.addEventListener("click", () => regenerateOneClip(item.id));
    card.append(video, info, download, regenerate);
    if (clipData.subtitles) {
      const edit = document.createElement("button");
      edit.type = "button";
      edit.className = "clip-edit-subtitles";
      edit.textContent = "Edit subtitles";
      edit.disabled = regenerate.disabled;
      edit.addEventListener("click", () => openSubtitleEditor(item));
      card.append(edit);
    }
  }
}
async function regenerateOneClip(clipId) {
  const id = state.active, version = loadVersion;
  for (const button of document.querySelectorAll(".clip-regenerate")) button.disabled = true;
  $("generate-clips").disabled = true;
  try {
    const job = await api(`/api/jobs/${id}/clips/${clipId}/regenerate`, { method: "POST" });
    if (version !== loadVersion) return;
    state.jobs = state.jobs.map(j => j.id === id ? job : j);
    renderJob(job);
    return true;
  } catch (e) {
    if (version !== loadVersion) return;
    renderClipState(state.jobs.find(j => j.id === id));
    $("clip-error").textContent = e.message;
    $("clip-error").hidden = false;
    return false;
  }
}

let subtitleEditor = null;
let subtitleSaving = false;

async function openSubtitleEditor(item) {
  const id = state.active, version = loadVersion;
  try {
    const document = await api(`/api/jobs/${id}/clips/${item.id}/subtitles`);
    if (version !== loadVersion) return;
    subtitleEditor = { id, clipId: item.id, start: item.source_start, version, document };
    $("subtitle-editor-name").textContent = document.title;
    $("subtitle-editor-error").hidden = true;
    $("subtitle-editor-status").textContent = document.pending
      ? "Saved corrections are waiting for regeneration. The preview still shows the current MP4."
      : "The preview shows the current MP4. Save and regenerate to apply your changes.";
    const words = $("subtitle-editor-words");
    words.replaceChildren();
    document.words.forEach((word, index) => {
      if (index) words.append(window.document.createTextNode(" "));
      const input = window.document.createElement("span");
      input.className = "subtitle-edit-word";
      input.id = `subtitle-word-${index}`;
      input.contentEditable = "plaintext-only";
      input.setAttribute("role", "textbox");
      input.setAttribute("aria-label", `Word ${index + 1}: ${word.word}`);
      input.setAttribute("aria-multiline", "false");
      input.tabIndex = 0;
      input.spellcheck = true;
      input.lang = "es";
      input.textContent = word.text;
      input.classList.toggle("corrected", word.text !== word.word);
      input.classList.toggle("removed", !word.text.trim());
      input.addEventListener("focus", () => {
        subtitleEditor.selectedIndex = index;
        const relative = Math.max(0, word.start - item.source_start);
        $("subtitle-editor-word-help").textContent = `Original: ${word.word} · ${formatTime(relative)}.${String(Math.floor((relative % 1) * 100)).padStart(2, "0")}`;
      });
      input.addEventListener("keydown", event => {
        if (event.key === "Enter") event.preventDefault();
      });
      input.addEventListener("input", () => {
        input.classList.toggle("corrected", input.textContent.trim() !== word.word);
        input.classList.toggle("removed", !input.textContent.trim());
        $("subtitle-editor-status").textContent = "Unsaved changes. Save and regenerate to update the MP4.";
      });
      words.append(input);
    });
    $("subtitle-editor-word-help").textContent = "Select a word to see its original spelling or listen to it.";
    $("subtitle-editor-video").src = `/api/jobs/${id}/clips/${item.id}/media?revision=${item.revision || clipData.revision}`;
    $("subtitle-editor").showModal();
  } catch (e) { notify(e.message); }
}

function closeSubtitleEditor() {
  $("subtitle-editor-video").pause();
  $("subtitle-editor-video").removeAttribute("src");
  $("subtitle-editor-video").load();
  $("subtitle-editor").close();
  subtitleEditor = null;
}
$("subtitle-editor-close").addEventListener("click", closeSubtitleEditor);
$("subtitle-editor-play-word").addEventListener("click", () => {
  if (!subtitleEditor) return;
  const word = subtitleEditor.document.words[subtitleEditor.selectedIndex || 0];
  const video = $("subtitle-editor-video");
  video.currentTime = Math.max(0, word.start - subtitleEditor.start);
  video.play().catch(() => {});
});
$("subtitle-editor").addEventListener("cancel", e => {
  e.preventDefault();
  if (!subtitleSaving) closeSubtitleEditor();
});
$("subtitle-editor-restore").addEventListener("click", () => {
  const inputs = $("subtitle-editor-words").querySelectorAll(".subtitle-edit-word");
  inputs.forEach((input, index) => {
    input.textContent = subtitleEditor.document.words[index].word;
    input.classList.remove("corrected");
    input.classList.remove("removed");
  });
  $("subtitle-editor-status").textContent = "Original text restored in the editor. Save to keep this change.";
});
async function saveSubtitleEditor(regenerate) {
  const editor = subtitleEditor;
  if (!editor || subtitleSaving || editor.version !== loadVersion) return;
  const inputs = [...$("subtitle-editor-words").querySelectorAll(".subtitle-edit-word")];
  const invalid = inputs.find(input => input.textContent.trim().length > 120);
  if (invalid) {
    $("subtitle-editor-error").textContent = "Each correction can contain at most 120 characters.";
    $("subtitle-editor-error").hidden = false;
    invalid.focus();
    return;
  }
  if (!inputs.some(input => input.textContent.trim())) {
    $("subtitle-editor-error").textContent = "Keep at least one subtitle word. Leave individual words empty to remove them.";
    $("subtitle-editor-error").hidden = false;
    return;
  }
  subtitleSaving = true;
  $("subtitle-editor-error").hidden = true;
  for (const element of $("subtitle-editor").querySelectorAll("button")) element.disabled = true;
  for (const input of inputs) input.contentEditable = "false";
  try {
    const document = await api(`/api/jobs/${editor.id}/clips/${editor.clipId}/subtitles`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ revision: editor.document.revision, texts: inputs.map(input => input.textContent) }),
    });
    editor.document = document;
    $("subtitle-editor-status").textContent = document.pending
      ? "Corrections saved. Regenerate this clip to apply them to the MP4." : "Corrections saved.";
    const job = state.jobs.find(j => j.id === editor.id);
    if (editor.version === loadVersion) await loadClips(job, `${job.id}:${job.clips_revision || ""}`);
    if (regenerate) {
      $("subtitle-editor-video").pause();
      if (await regenerateOneClip(editor.clipId)) closeSubtitleEditor();
      else throw new Error("Corrections were saved, but regeneration could not start. Close the editor and retry.");
    }
  } catch (e) {
    $("subtitle-editor-error").textContent = e.message;
    $("subtitle-editor-error").hidden = false;
  } finally {
    subtitleSaving = false;
    for (const element of $("subtitle-editor").querySelectorAll("button")) element.disabled = false;
    for (const input of inputs) input.contentEditable = "plaintext-only";
  }
}
$("subtitle-editor-save").addEventListener("click", () => saveSubtitleEditor(false));
$("subtitle-editor-render").addEventListener("click", () => saveSubtitleEditor(true));
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
      body: JSON.stringify({ mode: $("clip-mode").value, subtitles: $("clip-subtitles").checked }),
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
