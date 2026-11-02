
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const state = {
  config: null,
  word: "krishna",
  refSet: "Phone recordings",
  baseline: null,

  audioBlob: null,
  audioName: "",
  audioUrl: null,
  exampleLabel: null,

  recording: false,
  recorder: null,
  stream: null,
  chunks: [],
  recordingStarted: 0,
  recordingTimer: null,

  calibrationBlob: null,
  calibrationName: "",
  calibrationUrl: null,
  calibrationPreview: null,

  currentResult: null,
  currentAudioUrl: null,
  currentAudioBlob: null,
  activeMode: "record"
};

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("visible");

  clearTimeout(showToast.timeout);
  showToast.timeout = setTimeout(() => {
    toast.classList.remove("visible");
  }, 2800);
}

function showError(selector, message) {
  const element = $(selector);
  element.textContent = message;
  element.hidden = false;
}

function clearError(selector) {
  const element = $(selector);
  element.hidden = true;
  element.textContent = "";
}

async function api(url, options = {}) {
  const response = await fetch(url, options);

  let data;

  try {
    data = await response.json();
  } catch {
    data = {};
  }

  if (!response.ok) {
    throw new Error(
      data.detail || `Request failed (${response.status})`
    );
  }

  return data;
}

function formatDuration(seconds) {
  const total = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(total / 60);
  const remaining = total % 60;

  return `${String(minutes).padStart(2, "0")}:${String(remaining).padStart(2, "0")}`;
}

function formatFileSize(bytes) {
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }

  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function extensionForMime(mime) {
  if (mime.includes("mp4")) return "mp4";
  if (mime.includes("ogg")) return "ogg";
  if (mime.includes("mpeg")) return "mp3";
  if (mime.includes("wav")) return "wav";
  return "webm";
}

function createRecordingUrl(blob) {
  return URL.createObjectURL(blob);
}

function revokeUrl(url) {
  if (url) URL.revokeObjectURL(url);
}

function getSelectedWord() {
  return state.config?.words.find(
    (item) => item.id === state.word
  );
}

// --------------------------------------------------
// INITIALIZATION
// --------------------------------------------------

async function initialize() {
  try {
    const health = await api("/api/health");

    if (!health.ok) {
      throw new Error("Backend health check failed.");
    }

    await loadConfig();

    setupTabs();
    setupModes();
    setupWordSelector();
    setupReferenceSelector();
    setupAssessmentRecording();
    setupUpload();
    setupExamples();
    setupAssessment();
    setupLogging();
    setupCalibration();
    setupReset();

    await loadBaseline();

  } catch (error) {
    console.error(error);

    document.body.insertAdjacentHTML(
      "afterbegin",
      `<div style="
        position:fixed;top:0;left:0;right:0;z-index:999;
        background:#f7eae7;color:#a54e43;padding:13px 20px;
        font:12px DM Sans,sans-serif;text-align:center">
        Could not initialize the application: ${escapeHTML(error.message)}
      </div>`
    );
  }
}

async function loadConfig() {
  state.config = await api("/api/config");

  const wordSelect = $("#word-select");

  wordSelect.innerHTML = state.config.words.map((word) => `
    <option value="${escapeHTML(word.id)}">
      ${escapeHTML(word.display)} · ${escapeHTML(word.transliteration)}
    </option>
  `).join("");

  if (!state.config.words.some((word) => word.id === state.word)) {
    state.word = state.config.words[0]?.id || "";
  }

  wordSelect.value = state.word;
  updateWordDisplay();

  populateExamples();
  updateReferenceOptions();
}

function setupWordSelector() {
  $("#word-select").addEventListener("change", async (event) => {
    state.word = event.target.value;

    state.calibrationPreview = null;
    $("#takes-section").hidden = true;
    $("#cal-success").hidden = true;

    updateWordDisplay();
    populateExamples();
    updateReferenceOptions();

    await loadBaseline();
  });
}

function updateWordDisplay() {
  const word = getSelectedWord();

  if (!word) return;

  $("#word-script").textContent = word.display;
  $("#word-transliteration").textContent = word.transliteration;
}

function updateReferenceOptions() {
  const word = getSelectedWord();
  const select = $("#reference-select");

  if (!word) return;

  const options = word.reference_sets || ["Phone recordings"];

  select.innerHTML = options.map((option) => `
    <option value="${escapeHTML(option)}">
      ${escapeHTML(option)}
    </option>
  `).join("");

  if (!options.includes(state.refSet)) {
    state.refSet = "Phone recordings";
  }

  select.value = state.refSet;

  $("#calibration-count").textContent =
    `${getSavedLaptopCount()} SAVED`;
}

function getSavedLaptopCount() {
  const word = getSelectedWord();

  return word?.reference_sets.includes("Laptop microphone")
    ? "AVAILABLE"
    : "0";
}

function setupReferenceSelector() {
  $("#reference-select").addEventListener("change", async (event) => {
    state.refSet = event.target.value;
    await loadBaseline();
  });
}

async function loadBaseline() {
  if (!state.word) return;

  try {
    const params = new URLSearchParams({
      word: state.word,
      ref_set: state.refSet
    });

    const baseline = await api(`/api/baseline?${params}`);

    state.baseline = baseline;

    renderBaseline(baseline);

  } catch (error) {
    console.error(error);
    showToast(error.message);
  }
}

function renderBaseline(baseline) {
  $("#baseline-center").textContent =
    Number(baseline.center).toFixed(2);

  $("#baseline-count").textContent =
    baseline.reference_count;

  const threshold = $("#threshold-input");

  threshold.min = "0.5";
  threshold.max = String(
    Math.max(8, baseline.threshold * 2)
  );

  threshold.value = baseline.threshold.toFixed(2);

  updateThresholdLabel();

  $("#reference-distances").innerHTML =
    baseline.take_distances.map((item) => `
      <div class="reference-distance">
        <span>${escapeHTML(item.take)}</span>
        <strong>${Number(item.distance).toFixed(3)}</strong>
      </div>
    `).join("");
}

function updateThresholdLabel() {
  $("#threshold-value").textContent =
    Number($("#threshold-input").value).toFixed(2);
}

$("#threshold-input").addEventListener(
  "input",
  updateThresholdLabel
);

// --------------------------------------------------
// TABS
// --------------------------------------------------

function setupTabs() {
  $$(".tab-button").forEach((button) => {
    button.addEventListener("click", () => {
      const tab = button.dataset.tab;

      $$(".tab-button").forEach((item) => {
        item.classList.toggle("active", item === button);
      });

      $("#assess-panel").classList.toggle(
        "active",
        tab === "assess"
      );

      $("#calibrate-panel").classList.toggle(
        "active",
        tab === "calibrate"
      );
    });
  });
}

// --------------------------------------------------
// INPUT MODES
// --------------------------------------------------

function setupModes() {
  $$(".mode-button").forEach((button) => {
    button.addEventListener("click", () => {
      setMode(button.dataset.mode);
    });
  });
}

function setMode(mode) {
  state.activeMode = mode;

  $$(".mode-button").forEach((button) => {
    button.classList.toggle(
      "active",
      button.dataset.mode === mode
    );
  });

  $$(".mode-panel").forEach((panel) => {
    panel.classList.toggle(
      "active",
      panel.id === `${mode}-mode`
    );
  });

  if (mode !== "record" && state.recording) {
    stopRecording();
  }

  if (mode === "example" && state.refSet !== "Phone recordings") {
    $("#reference-select").value = "Phone recordings";
    state.refSet = "Phone recordings";
    loadBaseline();
  }

  updateAssessButton();
}

// --------------------------------------------------
// AUDIO RECORDING
// --------------------------------------------------

function setupAssessmentRecording() {
  $("#record-button").addEventListener("click", async () => {
    if (state.recording) {
      stopRecording();
    } else {
      await startRecording("assessment");
    }
  });
}

async function startRecording(purpose) {
  if (!navigator.mediaDevices?.getUserMedia) {
    showToast("Microphone access requires HTTPS or localhost.");
    return;
  }

  if (!window.MediaRecorder) {
    showToast("This browser does not support audio recording.");
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false
      }
    });

    const types = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/mp4"
    ];

    const mimeType = types.find((type) =>
      MediaRecorder.isTypeSupported(type)
    );

    const recorder = new MediaRecorder(
      stream,
      mimeType ? { mimeType } : undefined
    );

    state.stream = stream;
    state.recorder = recorder;
    state.chunks = [];
    state.recording = true;
    state.recordingStarted = Date.now();

    recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) {
        state.chunks.push(event.data);
      }
    });

    recorder.addEventListener("stop", () => {
      const blob = new Blob(state.chunks, {
        type: recorder.mimeType || "audio/webm"
      });

      stream.getTracks().forEach((track) => track.stop());

      state.stream = null;
      state.recorder = null;
      state.recording = false;

      if (purpose === "assessment") {
        setAssessmentAudio(
          blob,
          `recording.${extensionForMime(blob.type)}`
        );
        setRecordingUI(false);
      } else {
        setCalibrationAudio(
          blob,
          `calibration.${extensionForMime(blob.type)}`
        );
        setCalibrationRecordingUI(false);
      }
    });

    recorder.start();

    if (purpose === "assessment") {
      setRecordingUI(true);
      startTimer("#record-timer");
    } else {
      setCalibrationRecordingUI(true);
      startTimer("#cal-audio-meta");
    }

  } catch (error) {
    console.error(error);
    state.recording = false;

    showToast(
      error.name === "NotAllowedError"
        ? "Microphone permission was denied."
        : `Could not start recording: ${error.message}`
    );
  }
}

function stopRecording() {
  if (state.recorder && state.recorder.state !== "inactive") {
    state.recorder.stop();
  }

  clearInterval(state.recordingTimer);
  state.recordingTimer = null;
}

function startTimer(selector) {
  clearInterval(state.recordingTimer);

  const update = () => {
    const elapsed = (Date.now() - state.recordingStarted) / 1000;

    if (selector === "#record-timer") {
      $(selector).textContent = formatDuration(elapsed);
    } else {
      $(selector).textContent =
        `Recording · ${formatDuration(elapsed)}`;
    }

    if (elapsed >= 30) {
      stopRecording();
      showToast("Recording stopped at 30 seconds.");
    }
  };

  update();
  state.recordingTimer = setInterval(update, 250);
}

function setRecordingUI(active) {
  $("#record-orbit").classList.toggle("recording", active);

  $("#record-icon").textContent = active ? "■" : "◎";

  $("#record-title").textContent = active
    ? "Listening to you..."
    : "Ready when you are";

  $("#record-description").textContent = active
    ? "Say the word clearly. Press stop when finished."
    : "Find a quiet place. Say the word once, clearly.";

  $("#record-button-text").textContent = active
    ? "Stop recording"
    : "Start recording";

  $("#record-button-icon").textContent = active ? "■" : "●";
}

// --------------------------------------------------
// ASSESSMENT AUDIO
// --------------------------------------------------

function setupUpload() {
  $("#audio-file").addEventListener("change", (event) => {
    const file = event.target.files[0];

    if (!file) return;

    if (file.size > 25 * 1024 * 1024) {
      showError("#assess-error", "File exceeds the 25 MB limit.");
      event.target.value = "";
      return;
    }

    setAssessmentAudio(file, file.name);
  });

  $("#remove-audio").addEventListener("click", () => {
    clearAssessmentAudio();
  });
}

function setAssessmentAudio(blob, name) {
  revokeUrl(state.audioUrl);

  state.audioBlob = blob;
  state.audioName = name;
  state.exampleLabel = null;

  state.audioUrl = createRecordingUrl(blob);

  $("#selected-audio").hidden = false;
  $("#selected-audio-name").textContent = name;
  $("#selected-audio-meta").textContent =
    formatFileSize(blob.size);

  $("#audio-preview").src = state.audioUrl;

  $("#example-audio").hidden = true;

  clearError("#assess-error");
  updateAssessButton();
}

function clearAssessmentAudio() {
  revokeUrl(state.audioUrl);

  state.audioBlob = null;
  state.audioName = "";
  state.audioUrl = null;
  state.exampleLabel = null;

  $("#selected-audio").hidden = true;
  $("#audio-preview").removeAttribute("src");
  $("#audio-file").value = "";
  $("#example-select").value = "";
  $("#example-audio").hidden = true;

  updateAssessButton();
}

function populateExamples() {
  const word = getSelectedWord();
  const select = $("#example-select");

  select.innerHTML = `
    <option value="">Select an example...</option>
    ${(word?.examples || []).map((label) => `
      <option value="${escapeHTML(label)}">
        ${escapeHTML(label)}
      </option>
    `).join("")}
  `;
}

function setupExamples() {
  $("#example-select").addEventListener("change", async (event) => {
    const label = event.target.value;

    revokeUrl(state.audioUrl);

    state.audioBlob = null;
    state.audioUrl = null;
    state.audioName = "";
    state.exampleLabel = label || null;

    $("#selected-audio").hidden = true;
    $("#example-audio").hidden = !label;

    if (label) {
      const params = new URLSearchParams({
        word: state.word
      });

      $("#example-audio").src =
        `/api/examples/${encodeURIComponent(label)}/audio?${params}`;
    } else {
      $("#example-audio").removeAttribute("src");
    }

    updateAssessButton();
  });
}

function updateAssessButton() {
  const hasAudio = Boolean(
    state.audioBlob || state.exampleLabel
  );

  $("#assess-button").disabled = !hasAudio;
}

// --------------------------------------------------
// RUN ASSESSMENT
// --------------------------------------------------

function setupAssessment() {
  $("#assess-button").addEventListener("click", runAssessment);
}

async function runAssessment() {
  clearError("#assess-error");

  const button = $("#assess-button");

  button.disabled = true;
  button.innerHTML = "Analyzing audio...";

  try {
    const form = new FormData();

    form.append("word", state.word);
    form.append("ref_set", state.refSet);

    form.append(
      "threshold",
      $("#threshold-input").value
    );

    if (state.exampleLabel) {
      form.append("example_label", state.exampleLabel);
    } else if (state.audioBlob) {
      const extension = extensionForMime(
        state.audioBlob.type
      );

      form.append(
        "file",
        state.audioBlob,
        state.audioName || `recording.${extension}`
      );
    } else {
      throw new Error("Record or select an audio file first.");
    }

    const result = await api("/api/assess", {
      method: "POST",
      body: form
    });

    state.currentResult = result;
    state.currentAudioBlob = state.audioBlob;

    if (state.exampleLabel) {
      const params = new URLSearchParams({
        word: state.word
      });

      state.currentAudioUrl =
        `/api/examples/${encodeURIComponent(state.exampleLabel)}/audio?${params}`;
    } else {
      state.currentAudioUrl = state.audioUrl;
    }

    renderResults(result);

    $("#results-section").hidden = false;

    $("#results-section").scrollIntoView({
      behavior: "smooth",
      block: "start"
    });

  } catch (error) {
    showError("#assess-error", error.message);

  } finally {
    button.disabled = !(
      state.audioBlob || state.exampleLabel
    );

    button.innerHTML =
      'Analyze pronunciation <span>→</span>';
  }
}

// --------------------------------------------------
// RESULTS
// --------------------------------------------------

function renderResults(result) {
  $("#result-score").textContent =
    `${Math.round(result.score)}%`;

  $("#result-distance").textContent =
    Number(result.distance).toFixed(3);

  $("#result-center").textContent =
    `Typical correct distance ${Number(result.center).toFixed(2)}`;

  $("#result-verdict").textContent =
    result.matches ? "Matches" : "Differs";

  $("#result-verdict-detail").textContent =
    `Threshold ${Number(result.threshold).toFixed(2)}`;

  $("#result-duration").textContent =
    `${Number(result.duration).toFixed(2)} seconds`;

  const banner = $("#verdict-banner");

  banner.classList.toggle("differs", !result.matches);

  $("#verdict-symbol").textContent =
    result.matches ? "✓" : "!";

  $("#verdict-title").textContent =
    result.matches
      ? "Within normal variation"
      : "Outside normal variation";

  $("#verdict-description").textContent =
    result.matches
      ? "Your recording is within the normal variation of the correct references."
      : "Your recording differs from the normal variation of the correct references.";

  $("#spectrogram-image").src = result.spectrogram;

  $("#localization").hidden = !result.localized;
  $("#no-localization").hidden = result.localized;

  if (result.localized && result.window) {
    const [start, end] = result.window;

    $("#localization-time").textContent =
      `${Number(start).toFixed(2)}–${Number(end).toFixed(2)} s`;

    $("#localization-position").textContent =
      `${result.where} of the word · relative to the reference timeline`;

    $("#localization-peak").textContent =
      `${Number(result.peak).toFixed(1)} SD`;
  }

  $("#cues-list").innerHTML = (result.cues || [])
    .map((cue) => `
      <span class="cue-pill">${escapeHTML(cue)}</span>
    `).join("");

  $("#result-audio").src =
    state.currentAudioUrl || "";

  $("#result-disclaimer").textContent =
    result.disclaimer || "";
}

// --------------------------------------------------
// ATTEMPT LOGGING
// --------------------------------------------------

function setupLogging() {
  $("#log-button").addEventListener("click", saveAttempt);
}

async function saveAttempt() {
  if (!state.currentResult || !state.baseline) {
    return;
  }

  const button = $("#log-button");

  button.disabled = true;
  button.textContent = "Saving...";

  const result = state.currentResult;

  const payload = {
    ref_set: result.ref_set,
    said: $("#log-label").value,
    distance: Number(result.distance),
    score: Number(result.score),
    verdict: result.matches ? "Matches" : "Differs",
    peak_sd: Math.max(0, Number(result.peak || 0)),
    window: result.window
      ? `${result.window[0].toFixed(2)}-${result.window[1].toFixed(2)}`
      : "",
    mu: Number(result.center),
    sigma: Number(state.baseline.sigma),
    thr: Number(result.threshold)
  };

  try {
    const response = await api("/api/log", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    $("#log-status").textContent =
      `Saved successfully. Total attempts: ${response.total_attempts}`;

    showToast("Attempt saved.");

  } catch (error) {
    $("#log-status").textContent = error.message;
    $("#log-status").style.color = "var(--red)";

  } finally {
    button.disabled = false;
    button.textContent = "Save attempt";
  }
}

// --------------------------------------------------
// CALIBRATION
// --------------------------------------------------

function setupCalibration() {
  $("#cal-record-button").addEventListener("click", async () => {
    if (state.recording) {
      stopRecording();
    } else {
      await startRecording("calibration");
    }
  });

  $("#cal-file").addEventListener("change", (event) => {
    const file = event.target.files[0];

    if (!file) return;

    if (file.size > 25 * 1024 * 1024) {
      showError("#cal-error", "File exceeds the 25 MB limit.");
      event.target.value = "";
      return;
    }

    setCalibrationAudio(file, file.name);
  });

  $("#cal-remove-audio").addEventListener("click", () => {
    clearCalibrationAudio();
  });

  $("#gap-input").addEventListener("input", () => {
    $("#gap-value").textContent =
      Number($("#gap-input").value).toFixed(2);
  });

  $("#preview-calibration").addEventListener(
    "click",
    previewCalibration
  );

  $("#select-all-takes").addEventListener("click", () => {
    const boxes = $$(".take-checkbox");
    const allChecked = boxes.every((box) => box.checked);

    boxes.forEach((box) => {
      box.checked = !allChecked;
    });

    updateTakeSelection();
  });

  $("#save-calibration").addEventListener(
    "click",
    saveCalibration
  );
}

function setCalibrationRecordingUI(active) {
  $("#cal-record-button").innerHTML = active
    ? '<span>■</span><span>Stop recording</span>'
    : '<span>●</span><span>Record calibration</span>';
}

function setCalibrationAudio(blob, name) {
  revokeUrl(state.calibrationUrl);

  state.calibrationBlob = blob;
  state.calibrationName = name;
  state.calibrationUrl = createRecordingUrl(blob);

  state.calibrationPreview = null;

  $("#cal-audio-preview").hidden = false;
  $("#cal-audio-name").textContent = name;
  $("#cal-audio-meta").textContent =
    formatFileSize(blob.size);

  $("#cal-audio-player").src = state.calibrationUrl;

  $("#preview-calibration").disabled = false;
  $("#takes-section").hidden = true;

  clearError("#cal-error");
}

function clearCalibrationAudio() {
  revokeUrl(state.calibrationUrl);

  state.calibrationBlob = null;
  state.calibrationName = "";
  state.calibrationUrl = null;
  state.calibrationPreview = null;

  $("#cal-audio-preview").hidden = true;
  $("#cal-audio-player").removeAttribute("src");
  $("#cal-file").value = "";
  $("#preview-calibration").disabled = true;
  $("#takes-section").hidden = true;
}

function calibrationForm() {
  if (!state.calibrationBlob) {
    throw new Error("Record or upload a calibration clip first.");
  }

  const form = new FormData();

  form.append("word", state.word);
  form.append("gap", $("#gap-input").value);

  form.append(
    "file",
    state.calibrationBlob,
    state.calibrationName || "calibration.webm"
  );

  return form;
}

async function previewCalibration() {
  clearError("#cal-error");

  const button = $("#preview-calibration");

  button.disabled = true;
  button.textContent = "Detecting takes...";

  try {
    const result = await api(
      "/api/calibration/preview",
      {
        method: "POST",
        body: calibrationForm()
      }
    );

    state.calibrationPreview = result;

    renderTakes(result);

    $("#takes-section").hidden = false;

    $("#takes-section").scrollIntoView({
      behavior: "smooth",
      block: "start"
    });

  } catch (error) {
    showError("#cal-error", error.message);

  } finally {
    button.disabled = !state.calibrationBlob;
    button.innerHTML = 'Detect takes <span>→</span>';
  }
}

function renderTakes(result) {
  $("#takes-summary").textContent =
    `Detected ${result.count} take(s). Select at least four clean recordings.`;

  $("#takes-list").innerHTML = result.takes.map((take) => `
    <label class="take-item">
      <input
        class="take-checkbox"
        type="checkbox"
        value="${take.index}"
        checked
      />
      <span class="take-info">
        <strong>Take ${String(take.index + 1).padStart(2, "0")}</strong>
        <small>
          ${Number(take.start).toFixed(2)}–${Number(take.end).toFixed(2)} s
        </small>
      </span>
      <span class="take-duration">
        ${Number(take.duration).toFixed(2)} s
      </span>
    </label>
  `).join("");

  $$(".take-checkbox").forEach((box) => {
    box.addEventListener("change", updateTakeSelection);
  });

  updateTakeSelection();
}

function updateTakeSelection() {
  const selected = $$(".take-checkbox")
    .filter((box) => box.checked);

  const count = selected.length;

  $("#selected-takes-count").textContent =
    `${count} selected · 4 required`;

  $("#save-calibration").disabled =
    count < 4 || !state.calibrationBlob;

  $("#select-all-takes").textContent =
    count === $$(".take-checkbox").length
      ? "Deselect all"
      : "Select all";
}

async function saveCalibration() {
  clearError("#cal-error");

  const selected = $$(".take-checkbox")
    .filter((box) => box.checked)
    .map((box) => Number(box.value));

  if (selected.length < 4) {
    showError("#cal-error", "Select at least four takes.");
    return;
  }

  const button = $("#save-calibration");

  button.disabled = true;
  button.textContent = "Saving references...";

  try {
    const form = calibrationForm();

    form.append(
      "selected_indices",
      JSON.stringify(selected)
    );

    const result = await api(
      "/api/calibration/save",
      {
        method: "POST",
        body: form
      }
    );

    $("#cal-success").textContent =
      `${result.message} Saved ${result.saved} reference recordings.`;

    $("#cal-success").hidden = false;

    showToast("Calibration references saved.");

    await loadConfig();

    // Switch to the newly available laptop reference set.
    if (
      getSelectedWord()?.reference_sets.includes("Laptop microphone")
    ) {
      state.refSet = "Laptop microphone";
      $("#reference-select").value = state.refSet;

      await loadBaseline();
    }

    $("#calibration-count").textContent = "AVAILABLE";

  } catch (error) {
    showError("#cal-error", error.message);

  } finally {
    button.disabled = false;
    button.innerHTML = 'Save references <span>→</span>';
    updateTakeSelection();
  }
}

// --------------------------------------------------
// RESET
// --------------------------------------------------

function setupReset() {
  $("#new-attempt-button").addEventListener("click", () => {
    $("#results-section").hidden = true;
    $("#log-status").textContent = "";
    $("#log-status").style.color = "";

    clearAssessmentAudio();

    window.scrollTo({
      top: 0,
      behavior: "smooth"
    });
  });
}

// --------------------------------------------------
// UTILITIES
// --------------------------------------------------

function escapeHTML(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  })[char]);
}

window.addEventListener("beforeunload", () => {
  clearInterval(state.recordingTimer);

  state.stream?.getTracks().forEach((track) => {
    track.stop();
  });

  revokeUrl(state.audioUrl);
  revokeUrl(state.calibrationUrl);
});

initialize();
