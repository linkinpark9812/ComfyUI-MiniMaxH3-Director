// In-node live preview for MiniMax H3 Preview Override.
// Server-backed step history is independent of live animation playback.
// Only run identities and links live in the browser; archives survive clearing the node.

const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

const NODE_TYPE = "MiniMaxH3PreviewOverrideCS";
const IDLE_TEXT = "waiting for sample…";
const CONTROL_STYLE = {
  background: "#292929", color: "#ddd", border: "1px solid #555", borderRadius: "3px",
  padding: "4px 6px", font: "inherit", cursor: "pointer",
};

function element(tag, styles = {}, text = "") {
  const el = document.createElement(tag);
  Object.assign(el.style, styles);
  el.textContent = text;
  return el;
}

function archiveURL(panel, kind, step, download = false) {
  const query = new URLSearchParams({ folder: panel.folder, run_id: panel.runId });
  if (step != null) query.set("step", String(step));
  if (download) query.set("download", "1");
  return `/minimax_director/preview_archive/${kind}?${query}`;
}

function buildPanel() {
  const root = element("div", {
    display: "flex", flexDirection: "column", gap: "6px", minHeight: "0",
    boxSizing: "border-box", width: "100%", height: "100%", overflow: "hidden",
    color: "#ccc", fontSize: "11px",
  });
  const body = element("div", { display: "flex", gap: "8px", flex: "1", minHeight: "0" });
  const frame = element("div", {
    position: "relative", minWidth: "0", flex: "1", minHeight: "0",
    background: "#141414", border: "1px solid #3a3a3a", borderRadius: "6px",
    display: "flex", alignItems: "center", justifyContent: "center", overflow: "hidden",
  });
  const img = element("img", { width: "100%", height: "100%", objectFit: "contain", display: "none" });
  img.alt = "Sampling preview";
  const video = element("video", { width: "100%", height: "100%", objectFit: "contain", display: "none" });
  video.muted = true;
  video.autoplay = true;
  video.loop = true;
  video.playsInline = true;
  video.controls = true;
  video.setAttribute("aria-label", "Synchronized step preview");
  const idle = element("div", { color: "#999", padding: "8px" }, IDLE_TEXT);
  frame.append(img, video, idle);
  const sidebar = element("div", {
    display: "flex", flexDirection: "column", gap: "4px", width: "230px",
    maxWidth: "45%", minHeight: "0", flexShrink: "0",
  });
  const live = element("button", CONTROL_STYLE, "Live");
  live.type = "button";
  live.title = "Resume the latest live preview";
  const sound = element("button", CONTROL_STYLE, "Unmute");
  sound.type = "button";
  sound.disabled = true;
  sound.title = "Connect audio_vae to hear saved steps. Playback starts muted.";
  const list = element("div", { overflowY: "auto", flex: "1", minHeight: "0" });
  list.setAttribute("aria-label", "Saved sampling steps");
  const zip = element("a", { ...CONTROL_STYLE, color: "#8bcaff", textAlign: "center",
    textDecoration: "none" }, "Download all (.zip)");
  zip.setAttribute("role", "button");
  zip.addEventListener("click", (event) => {
    if (zip.getAttribute("aria-disabled") === "true") event.preventDefault();
  });
  zip.addEventListener("keydown", (event) => {
    if (event.key === " ") {
      event.preventDefault();
      zip.click();
    }
  });
  sidebar.append(live, sound, list, zip);
  body.append(frame, sidebar);
  const status = element("div", { display: "flex", gap: "8px", justifyContent: "space-between",
    fontFamily: "monospace", fontSize: "10px", flexWrap: "wrap" });
  const left = element("span");
  const right = element("span");
  status.append(left, right);
  const summary = element("div", { overflowWrap: "anywhere" });
  summary.setAttribute("aria-live", "polite");
  root.append(body, status, summary);
  // Interacting with a control must not drag the graph node or zoom the canvas.
  for (const event of ["pointerdown", "mousedown", "dblclick", "wheel"]) {
    sidebar.addEventListener(event, (e) => e.stopPropagation());
  }
  const panel = { root, img, video, sound, idle, left, right, summary, live, list, zip,
    muted: true, disposed: false };
  sound.addEventListener("click", () => {
    panel.muted = !panel.muted;
    video.muted = panel.muted;
    sound.textContent = panel.muted ? "Unmute" : "Mute";
    sound.setAttribute("aria-pressed", String(!panel.muted));
    void video.play().catch(() => { sound.title = "Use the player's Play control to start playback"; });
  });
  video.addEventListener("volumechange", () => {
    panel.muted = video.muted;
    sound.textContent = panel.muted ? "Unmute" : "Mute";
    sound.setAttribute("aria-pressed", String(!panel.muted));
  });
  video.addEventListener("error", () => {
    if (!video.getAttribute("src")) return;
    video.style.display = "none";
    img.style.display = "block";
    sound.disabled = true;
    panel.error = "AV playback unavailable; WebP and WAV downloads are still available";
    updateSummary(panel);
  });
  for (const event of ["pointerdown", "mousedown", "dblclick", "wheel"]) {
    video.addEventListener(event, (e) => e.stopPropagation());
  }
  live.addEventListener("click", () => {
    panel.selected = null;
    if (panel.latestDisplay) showPreview(panel, panel.latestDisplay);
    live.setAttribute("aria-pressed", "true");
  });
  resetPanel(panel);
  return panel;
}

function setZip(panel, ready, label = "Download all (.zip)") {
  panel.zip.textContent = label;
  panel.zip.setAttribute("aria-disabled", String(!ready));
  panel.zip.tabIndex = ready ? 0 : -1;
  panel.zip.style.opacity = ready ? "1" : "0.5";
  if (ready) {
    panel.zip.href = api.apiURL(archiveURL(panel, "zip"));
    panel.zip.download = `${panel.folder}.zip`;
  } else {
    panel.zip.removeAttribute("href");
    panel.zip.removeAttribute("download");
  }
}

function resetPanel(panel, promptId = null) {
  clearTimeout(panel.retryTimer);
  panel.controller?.abort();
  Object.assign(panel, { promptId, runId: null, folder: null, sequence: 0,
    artifacts: new Map(), selected: null, latestDisplay: null, terminal: false,
    error: null, retries: 0, controller: null, retryTimer: null, target: "node" });
  panel.list.replaceChildren();
  panel.img.removeAttribute("src");
  panel.img.style.display = "none";
  stopAV(panel);
  panel.idle.style.display = "block";
  panel.idle.textContent = IDLE_TEXT;
  panel.left.textContent = promptId ? "waiting…" : "idle";
  panel.right.textContent = "";
  panel.summary.textContent = "No saved steps";
  panel.live.setAttribute("aria-pressed", "true");
  setZip(panel, false);
}

function stopAV(panel) {
  panel.video.pause();
  panel.video.removeAttribute("src");
  panel.video.load();
  panel.video.style.display = "none";
  panel.sound.disabled = true;
}

function showPreview(panel, data) {
  const artifact = data.artifact || data;
  const src = data.webp ? `data:image/webp;base64,${data.webp}`
    : api.apiURL(archiveURL(panel, "step", artifact.step));
  if (panel.img.getAttribute("src") !== src) panel.img.src = src;
  panel.img.style.display = "block";
  if (artifact.audio) {
    const avSrc = api.apiURL(archiveURL(panel, "playback", artifact.step));
    if (panel.video.getAttribute("src") !== avSrc) {
      panel.video.pause();
      panel.video.muted = panel.muted;
      panel.video.src = avSrc;
    }
    panel.img.style.display = "none";
    panel.video.style.display = "block";
    panel.sound.disabled = false;
    panel.sound.title = "Audio and video share one playback clock at the clip's real speed";
    void panel.video.play().catch(() => {
      panel.sound.title = "Use the player's Play control to start audio/video playback";
    });
  } else {
    stopAV(panel);
  }
  panel.idle.style.display = "none";
  const rate = Number(artifact.fps || 0);
  const source = Number(artifact.source_fps || rate);
  const fps = Math.abs(source - rate) > 0.05
    ? `${rate.toFixed(1)}fps of ${source.toFixed(0)} · true speed`
    : `${rate.toFixed(1)}fps${artifact.playback === "source fps" ? " · source" : ""}`;
  panel.left.textContent = `step ${artifact.step}/${artifact.total_steps} · ${artifact.frames}f @${fps}`;
  if (artifact.audio) {
    panel.left.textContent = `step ${artifact.step}/${artifact.total_steps} · AV · ${artifact.audio.duration.toFixed(2)}s · real time`;
  }
  panel.right.textContent = `render ${((artifact.ms || 0) / 1000).toFixed(2)}s · ${(artifact.mode || "").split(" ")[0]}`;
}

function addArtifact(panel, artifact) {
  if (!Number.isInteger(artifact.step)) return;
  const previous = panel.artifacts.get(artifact.step);
  if (previous && JSON.stringify(previous) === JSON.stringify(artifact)) return;
  // Reconnect/polling may see the WebP/WAV while MP4 encoding is still running.
  // Update its controls when the export arrives without restarting playback.
  if (previous) {
    [...panel.list.children].find((row) => Number(row.dataset.step) === artifact.step)?.remove();
  }
  panel.artifacts.set(artifact.step, artifact);
  const row = element("div", { display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
    gap: "4px", alignItems: "center", padding: "3px 0" });
  row.dataset.step = String(artifact.step);
  const select = element("button", { ...CONTROL_STYLE, gridColumn: "1 / -1", minWidth: "0",
    textAlign: "left", overflow: "hidden", textOverflow: "ellipsis" }, artifact.filename);
  select.type = "button";
  select.title = `Inspect step ${artifact.step}`;
  select.addEventListener("click", () => {
    panel.selected = artifact.step;
    panel.live.setAttribute("aria-pressed", "false");
    showPreview(panel, artifact);
  });
  const download = element("a", { ...CONTROL_STYLE, color: "#8bcaff", textDecoration: "none" }, "WebP");
  download.href = api.apiURL(archiveURL(panel, "step", artifact.step, true));
  download.download = artifact.filename;
  download.title = `Download ${artifact.filename}`;
  download.setAttribute("aria-label", download.title);
  const wav = element(artifact.audio ? "a" : "button", {
    ...CONTROL_STYLE, color: "#8bcaff", textDecoration: "none",
  }, "WAV");
  if (artifact.audio) {
    wav.href = api.apiURL(archiveURL(panel, "audio", artifact.step, true));
    wav.download = artifact.audio.filename;
    wav.title = `Download ${artifact.audio.filename}`;
    wav.setAttribute("aria-label", wav.title);
  } else {
    wav.disabled = true;
    wav.title = artifact.audio_error || "No audio saved for this step; connect audio_vae before running";
    wav.style.opacity = "0.4";
  }
  const mp4 = element(artifact.mp4 ? "a" : "button", {
    ...CONTROL_STYLE, color: "#8bcaff", textDecoration: "none",
  }, "MP4");
  if (artifact.mp4) {
    mp4.href = api.apiURL(archiveURL(panel, "mp4", artifact.step, true));
    mp4.download = artifact.mp4.filename;
    mp4.title = `Download ${artifact.mp4.filename}`;
    mp4.setAttribute("aria-label", mp4.title);
  } else {
    mp4.disabled = true;
    mp4.title = artifact.mp4_error || "No MP4 saved for this step; connect audio_vae before running";
    mp4.style.opacity = "0.4";
  }
  row.append(select, download, wav, mp4);
  const following = [...panel.list.children].find((r) => Number(r.dataset.step) > artifact.step);
  panel.list.insertBefore(row, following || null);
}

function updateSummary(panel, data = {}) {
  if (data.error) panel.error = data.error;
  const labels = { running: "Sampling", completed: "Completed", interrupted: "Canceled", failed: "Failed" };
  const status = labels[data.sampling_status] || (panel.terminal ? "Finished" : "Sampling");
  const failures = data.capture_failures?.length || 0;
  panel.summary.textContent = `${panel.folder || "Preview"} · ${status} · ${panel.artifacts.size} steps saved` +
    (failures ? ` · ${failures} capture failures` : "") + (panel.error ? ` · ${panel.error}` : "");
}

function applyManifest(panel, data) {
  panel.error = data.error || null;
  for (const artifact of data.artifacts || []) addArtifact(panel, artifact);
  panel.target = data.settings?.preview_target || panel.target;
  if (data.sampling_status && data.sampling_status !== "running") {
    panel.terminal = true;
    clearTimeout(panel.retryTimer);
    setZip(panel, data.archive_state === "ready",
      data.archive_state === "ready" ? "Download all (.zip)" : "ZIP unavailable");
    const latest = [...panel.artifacts.values()].sort((a, b) => b.step - a.step)[0];
    if (latest && panel.target !== "sampler (VHS)") {
      panel.latestDisplay = latest;
      if (panel.selected == null) showPreview(panel, latest);
    }
  }
  updateSummary(panel, data);
}

async function reconcile(panel) {
  if (!panel.runId || panel.disposed) return;
  const runId = panel.runId;
  panel.controller?.abort();
  const controller = new AbortController();
  panel.controller = controller;
  try {
    const response = await api.fetchApi(archiveURL(panel, "manifest"), { signal: controller.signal });
    if (!response.ok) throw new Error(`Archive status unavailable (${response.status})`);
    const data = await response.json();
    if (panel.disposed || panel.runId !== runId || controller.signal.aborted) return;
    // A response requested while recording may arrive after the finished event.
    if (!panel.terminal || data.sampling_status !== "running") applyManifest(panel, data);
  } catch (error) {
    if (error.name === "AbortError" || panel.disposed || panel.runId !== runId) return;
    panel.error = error.message;
    updateSummary(panel);
  }
  if (!panel.terminal && panel.runId === runId && !panel.disposed && panel.retries++ < 30) {
    clearTimeout(panel.retryTimer);
    panel.retryTimer = setTimeout(() => reconcile(panel), 2000);
  }
}

function eventPanel(data) {
  const node = app.graph?._nodes?.find((n) => String(n.id) === String(data.node_id));
  const panel = node?._mmxPreview;
  if (!panel || panel.disposed || !data.run_id) return null;
  if (panel.promptId && panel.promptId !== data.prompt_id) return null;
  if (panel.runId !== data.run_id) {
    // IDs begin with a UTC timestamp. A late event from an older invocation cannot
    // replace the current invocation when one patched model feeds multiple samplers.
    if (panel.runId && data.run_id < panel.runId) return null;
    resetPanel(panel, data.prompt_id);
    panel.runId = data.run_id;
    panel.folder = data.folder_name;
    if (data.kind !== "started") void reconcile(panel);
  }
  if (data.sequence <= panel.sequence) return null;
  if (panel.sequence && data.sequence > panel.sequence + 1) void reconcile(panel);
  panel.sequence = data.sequence;
  return panel;
}

app.registerExtension({
  name: "MiniMaxH3PreviewOverrideCS",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      if (onNodeCreated) onNodeCreated.apply(this, arguments);

      const panel = buildPanel();
      this._mmxPreview = panel;

      // Height is declared through the DOM-widget layout API, not through computeSize.
      //
      // computeSize is the wrong hook here: LiteGraph evaluates it to get the node's
      // minimum height, so any height derived from node.size[1] ratchets — the minimum is
      // measured against the size you are trying to leave, and the node can only ever
      // grow. Core's multiline textarea sidesteps that by declaring a minHeight and no
      // maxHeight, which lets the layout engine hand it the leftover space instead. Same
      // deal here: one floor, no ceiling, resizable in both directions.
      const MIN_PREVIEW_H = 140;
      const widget = this.addDOMWidget("minimax_preview_ui", "minimax_preview_ui", panel.root, {
        getValue: () => "",
        setValue: () => {},
        getMinHeight: () => MIN_PREVIEW_H,
      });
      widget.serialize = false;

      if (this.size[0] < 560) this.size[0] = 560;
      if (this.size[1] < 520) this.size[1] = 520;
    };

    // Heal combo widgets whose saved value is not one of their options.
    //
    // ComfyUI stores widgets_values positionally, so a workflow saved while the widget
    // list had a different shape hands every value after the change to the wrong input —
    // and a combo then refuses to run with "The value true is not available". The values
    // themselves are unrecoverable at that point; what matters is that the node comes back
    // usable instead of blocking the whole workflow. Same idea as js/minimax_titles.js
    // healing stale titles.
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const r = onConfigure?.apply(this, arguments);
      for (const w of this.widgets || []) {
        const opts = w.options?.values;
        if (!Array.isArray(opts) || opts.length === 0) continue;
        if (opts.includes(w.value)) continue;
        const fallback = w.options?.default ?? opts[0];
        console.warn(
          `[MiniMaxDirector] ${NODE_TYPE}: saved value ${JSON.stringify(w.value)} is not a ` +
          `valid '${w.name}' — this workflow was saved against a different widget layout. ` +
          `Falling back to ${JSON.stringify(fallback)}; check the node's settings.`);
        w.value = fallback;
      }

      // A widget added since this workflow was saved pushes the preview panel down by one
      // row, and the panel has a floor it will not shrink below — so on a node saved at
      // its old minimum the panel ends up hanging out of the body. Measured: with the
      // rows this node has now, anything under computeSize() overflows by exactly the
      // difference. Grow to that minimum, never shrink: the height above it is the
      // user's choice.
      const min = this.computeSize()[1];
      if (this.size[1] < min) {
        this.size[1] = min;
        // the draw loop only re-arranges widgets for nodes carrying this flag; without it
        // the body would redraw at the new height while the overlay keeps the old one
        this._widgetSlotsDirty = true;
        this.setDirtyCanvas?.(true, true);
      }
      return r;
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      if (this._mmxPreview) {
        resetPanel(this._mmxPreview);
        this._mmxPreview.disposed = true;
      }
      this._mmxPreview = null;
      return onRemoved?.apply(this, arguments);
    };
  },

  async setup() {
    api.addEventListener("minimax_h3_preview_archive", ({ detail: data }) => {
      if (!data) return;
      const panel = eventPanel(data);
      if (!panel) return;
      if (data.kind === "started") {
        applyManifest(panel, data);
      } else if (data.kind === "step_saved") {
        addArtifact(panel, data.artifact);
        updateSummary(panel);
      } else if (data.kind === "finalizing") {
        setZip(panel, false, "Preparing ZIP…");
      } else if (data.kind === "finished") {
        applyManifest(panel, data);
      } else if (data.kind === "archive_error") {
        updateSummary(panel, data);
      }
    });
    api.addEventListener("minimax_h3_preview", ({ detail: data }) => {
      if (!data) return;
      const panel = eventPanel(data);
      if (!panel) return;
      panel.latestDisplay = data;
      if (panel.selected == null) showPreview(panel, data);
    });
    api.addEventListener("execution_start", ({ detail }) => {
      for (const node of app.graph?._nodes || []) {
        if (node._mmxPreview) resetPanel(node._mmxPreview, detail?.prompt_id);
      }
    });
    const terminal = ({ detail }) => {
      for (const node of app.graph?._nodes || []) {
        const panel = node._mmxPreview;
        if (!panel || (detail?.prompt_id && panel.promptId !== detail.prompt_id)) continue;
        if (panel.runId) {
          if (!panel.terminal) void reconcile(panel);
        } else {
          panel.left.textContent = "idle";
          panel.summary.textContent = "No new previews generated";
        }
      }
    };
    for (const type of ["execution_success", "execution_interrupted", "execution_error"]) {
      api.addEventListener(type, terminal);
    }
    api.addEventListener("reconnected", () => {
      for (const node of app.graph?._nodes || []) {
        const panel = node._mmxPreview;
        if (panel?.runId && !panel.terminal) {
          panel.retries = 0;
          void reconcile(panel);
        }
      }
    });
  },
});
