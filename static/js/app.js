(() => {
  const ALLOWED = ["wav", "flac", "ogg", "mp3"];
  const MAX_BYTES = 10 * 1024 * 1024;

  const $ = (id) => document.getElementById(id);
  const dropzone = $("dropzone");
  const input = $("file-input");
  const filePanel = $("file-panel");
  const player = $("player");
  const canvas = $("waveform");
  const predictBtn = $("predict-btn");
  const clearBtn = $("clear-btn");
  const modelSelect = $("model-select");
  const errorBox = $("error");
  const views = { empty: $("result-empty"), loading: $("result-loading"), result: $("result") };

  const recorderEl = $("recorder");
  const recordBtn = $("record-btn");
  const tabUpload = $("tab-upload");
  const tabRecord = $("tab-record");
  const MAX_REC_SECONDS = 3;

  let currentFile = null;
  let objectUrl = null;
  let mode = "upload"; // or "record"
  let rec = null;

  const showView = (name) => Object.entries(views).forEach(([k, el]) => (el.hidden = k !== name));
  const showError = (msg) => { errorBox.textContent = msg; errorBox.hidden = false; };
  const hideError = () => { errorBox.hidden = true; errorBox.textContent = ""; };
  const fmtSize = (b) => (b < 1024 * 1024 ? `${(b / 1024).toFixed(1)} KB` : `${(b / 1048576).toFixed(2)} MB`);
  const modelReady = () => modelSelect && !modelSelect.disabled && modelSelect.value &&
    !modelSelect.selectedOptions[0]?.disabled;

  function setFile(file) {
    hideError();
    if (!file) return;
    const ext = file.name.includes(".") ? file.name.split(".").pop().toLowerCase() : "";
    if (!ALLOWED.includes(ext)) {
      showError(`Unsupported file type ".${ext || "?"}". Please upload ${ALLOWED.map((e) => "." + e).join(", ")}.`);
      return;
    }
    if (file.size > MAX_BYTES) {
      showError(`File is too large (${fmtSize(file.size)}). The limit is 10 MB.`);
      return;
    }
    if (file.size === 0) {
      showError("The selected file is empty.");
      return;
    }

    currentFile = file;
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);
    player.src = objectUrl;
    $("file-name").textContent = file.name;
    $("file-size").textContent = fmtSize(file.size);
    dropzone.hidden = true;
    recorderEl.hidden = true;
    $("change-btn").textContent = mode === "record" ? "Record again" : "Change";
    filePanel.hidden = false;
    predictBtn.disabled = !modelReady();
    clearBtn.disabled = false;
    showView("empty");
    drawWaveform(file);
  }

  async function drawWaveform(file) {
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * dpr; canvas.height = h * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);
    canvas.hidden = false;
    try {
      const AC = window.AudioContext || window.webkitAudioContext;
      const ac = new AC();
      const buf = await ac.decodeAudioData(await file.arrayBuffer());
      ac.close();
      const data = buf.getChannelData(0);
      const step = Math.max(1, Math.floor(data.length / w));
      const grad = ctx.createLinearGradient(0, 0, w, 0);
      grad.addColorStop(0, "#4f46e5"); grad.addColorStop(1, "#06b6d4");
      ctx.fillStyle = grad;
      const mid = h / 2;
      let peak = 0;
      for (let i = 0; i < data.length; i++) peak = Math.max(peak, Math.abs(data[i]));
      const scale = peak > 0 ? 1 / peak : 1;
      for (let x = 0; x < w; x += 2) {
        let max = 0;
        for (let j = 0; j < step; j++) max = Math.max(max, Math.abs(data[x * step + j] || 0));
        const bar = Math.max(1, max * scale * (mid - 4));
        ctx.fillRect(x, mid - bar, 1.4, bar * 2);
      }
    } catch {
      canvas.hidden = true; // browser cannot decode this format; playback/prediction still work
    }
  }

  function showSource(m) {
    tabUpload.classList.toggle("active", m === "upload");
    tabRecord.classList.toggle("active", m === "record");
    tabUpload.setAttribute("aria-selected", m === "upload");
    tabRecord.setAttribute("aria-selected", m === "record");
    dropzone.hidden = m !== "upload";
    recorderEl.hidden = m !== "record";
  }

  // ---- Microphone recording -------------------------------------------------
  // The browser records raw PCM and saves it as a WAV file. That file is then
  // uploaded exactly like a chosen file, so the server applies the same
  // preprocessing and model to both.
  async function startRecording() {
    hideError();
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showError("Recording is not available here. Open the site at http://127.0.0.1:5000 in Chrome, Safari or Firefox.");
      return;
    }
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
    } catch (e) {
      showError(e && e.name === "NotAllowedError"
        ? "Microphone access was blocked. Allow the microphone for this site in your browser and try again."
        : "No microphone was found. Connect one, or upload a file instead.");
      return;
    }
    const AC = window.AudioContext || window.webkitAudioContext;
    const ctx = new AC();
    await ctx.resume();
    const source = ctx.createMediaStreamSource(stream);
    const proc = ctx.createScriptProcessor(4096, 1, 1);
    const chunks = [];
    proc.onaudioprocess = (e) => {
      const d = e.inputBuffer.getChannelData(0);
      chunks.push(new Float32Array(d));
      let peak = 0;
      for (let i = 0; i < d.length; i++) peak = Math.max(peak, Math.abs(d[i]));
      $("rec-level").style.width = Math.min(100, peak * 140) + "%";
    };
    source.connect(proc);
    proc.connect(ctx.destination); // required for the node to run; it outputs silence

    const started = performance.now();
    const timer = setInterval(() => {
      const t = (performance.now() - started) / 1000;
      $("rec-sub").textContent = `${t.toFixed(1)} s / ${MAX_REC_SECONDS}.0 s`;
      if (t >= MAX_REC_SECONDS) stopRecording();
    }, 100);

    rec = { stream, ctx, source, proc, chunks, timer };
    recorderEl.classList.add("recording");
    recordBtn.setAttribute("aria-label", "Stop recording");
    $("rec-label").textContent = "Listening… say one number now";
    tabUpload.disabled = tabRecord.disabled = true;
  }

  function stopRecording() {
    if (!rec) return;
    const { stream, ctx, source, proc, chunks, timer } = rec;
    rec = null;
    clearInterval(timer);
    proc.onaudioprocess = null;
    source.disconnect(); proc.disconnect();
    stream.getTracks().forEach((t) => t.stop());
    const sr = ctx.sampleRate;
    ctx.close();
    recorderEl.classList.remove("recording");
    recordBtn.setAttribute("aria-label", "Start recording");
    $("rec-label").textContent = "Click the button and say one number";
    $("rec-sub").textContent = `Recording stops automatically after ${MAX_REC_SECONDS} seconds`;
    $("rec-level").style.width = "0";
    tabUpload.disabled = tabRecord.disabled = false;

    let total = 0;
    chunks.forEach((c) => (total += c.length));
    const samples = new Float32Array(total);
    let off = 0;
    chunks.forEach((c) => { samples.set(c, off); off += c.length; });

    const trimmed = trimSilence(samples, sr);
    if (!trimmed) {
      showError("No speech was heard. Check your microphone and speak a little louder.");
      return;
    }
    const file = new File([encodeWav(trimmed, sr)], `recording-${Date.now()}.wav`, { type: "audio/wav" });
    setFile(file);
  }

  // The training clips are short, tightly cut recordings and the LSTM only looks
  // at the first second, so leading/trailing silence is cut off the recording
  // (keeping a short margin) before it is sent. The server pipeline is unchanged.
  function trimSilence(x, sr) {
    const frame = Math.round(sr * 0.01);
    const rms = [];
    for (let i = 0; i + frame <= x.length; i += frame) {
      let s = 0;
      for (let j = i; j < i + frame; j++) s += x[j] * x[j];
      rms.push(Math.sqrt(s / frame));
    }
    const max = Math.max(0, ...rms);
    if (max < 0.005) return null;
    const thr = Math.max(0.08 * max, 0.003);
    let first = rms.findIndex((v) => v > thr);
    let last = rms.length - 1 - [...rms].reverse().findIndex((v) => v > thr);
    const start = Math.max(0, first * frame - Math.round(0.15 * sr));
    const end = Math.min(x.length, (last + 1) * frame + Math.round(0.25 * sr));
    return x.subarray(start, end);
  }

  function encodeWav(x, sr) {
    const buf = new ArrayBuffer(44 + x.length * 2);
    const v = new DataView(buf);
    const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    str(0, "RIFF"); v.setUint32(4, 36 + x.length * 2, true); str(8, "WAVE");
    str(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    str(36, "data"); v.setUint32(40, x.length * 2, true);
    for (let i = 0; i < x.length; i++) {
      const s = Math.max(-1, Math.min(1, x[i]));
      v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return new Blob([buf], { type: "audio/wav" });
  }

  function reset() {
    currentFile = null;
    input.value = "";
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = null;
    player.removeAttribute("src");
    player.load();
    filePanel.hidden = true;
    showSource(mode);
    predictBtn.disabled = true;
    clearBtn.disabled = true;
    hideError();
    showView("empty");
  }

  function renderResult(r) {
    $("result-digit").textContent = r.digit;
    const hasProba = r.confidence !== null && r.confidence !== undefined;
    $("confidence-block").hidden = !hasProba;
    $("topk-block").hidden = !hasProba;
    $("no-proba").hidden = hasProba;

    if (hasProba) {
      const pct = (r.confidence * 100).toFixed(1) + "%";
      $("confidence-value").textContent = pct;
      const bar = $("confidence-bar");
      bar.style.width = "0";
      requestAnimationFrame(() => (bar.style.width = (r.confidence * 100).toFixed(1) + "%"));

      const list = $("topk");
      list.innerHTML = "";
      r.top.forEach((t) => {
        const li = document.createElement("li");
        const p = (t.probability * 100).toFixed(1);
        li.innerHTML = `<span class="tk-digit"></span><span class="tk-bar"><span style="width:0"></span></span><span class="tk-val">${p}%</span>`;
        li.querySelector(".tk-digit").textContent = t.digit;
        list.appendChild(li);
        requestAnimationFrame(() => (li.querySelector(".tk-bar span").style.width = Math.max(p, 0.5) + "%"));
      });
    }
    $("meta-model").textContent = r.model.name;
    $("meta-duration").textContent = `${r.audio.duration} s`;
    $("meta-sr").textContent = `${r.audio.sample_rate.toLocaleString()} Hz`;
    showView("result");
  }

  async function predict() {
    if (!currentFile) return;
    hideError();
    predictBtn.disabled = true;
    predictBtn.classList.add("loading");
    predictBtn.querySelector(".btn-label").textContent = "Predicting…";
    showView("loading");

    const form = new FormData();
    form.append("audio", currentFile);
    if (modelSelect) form.append("model", modelSelect.value);

    try {
      const res = await fetch("/api/predict", { method: "POST", body: form });
      let body;
      try { body = await res.json(); } catch { body = {}; }
      if (!res.ok) throw new Error(body.error || `Server error (${res.status}).`);
      renderResult(body);
    } catch (e) {
      showView("empty");
      showError(e instanceof TypeError ? "Could not reach the server. Is the app still running?" : e.message);
    } finally {
      predictBtn.classList.remove("loading");
      predictBtn.querySelector(".btn-label").textContent = "Predict Digit";
      predictBtn.disabled = !currentFile || !modelReady();
    }
  }

  input.addEventListener("change", () => setFile(input.files[0]));
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
  });
  ["dragenter", "dragover"].forEach((t) =>
    dropzone.addEventListener(t, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((t) =>
    dropzone.addEventListener(t, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); }));
  dropzone.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));
  // Stop the browser from opening a file dropped outside the zone
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  $("change-btn").addEventListener("click", () => {
    if (mode === "record") { reset(); startRecording(); } else input.click();
  });
  tabUpload.addEventListener("click", () => { if (!rec) { mode = "upload"; reset(); } });
  tabRecord.addEventListener("click", () => { if (!rec) { mode = "record"; reset(); } });
  recordBtn.addEventListener("click", () => (rec ? stopRecording() : startRecording()));
  predictBtn.addEventListener("click", predict);
  clearBtn.addEventListener("click", reset);
  modelSelect?.addEventListener("change", () => { predictBtn.disabled = !currentFile || !modelReady(); });
})();
