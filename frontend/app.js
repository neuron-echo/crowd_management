/* ─────────────────────────────────────────────────────────────
   CrowdGuard AI — Live Dashboard JS
   WebSocket → real-time stats push every 500ms
   MJPEG   → live + density streams per camera
───────────────────────────────────────────────────────────── */

const CAMERA_META = {
  cam_01: { label: "Camera 01", gateLabel: "Gate North (G1)" },
  cam_02: { label: "Camera 02", gateLabel: "Gate East (G2)"  },
  cam_03: { label: "Camera 03", gateLabel: "Gate West (G3)"  },
};

// ── State ──────────────────────────────────────────────────────
let latestSnap = null;
let ws = null;
let reconnectTimer = null;

// ── Helpers ────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const el = (tag, cls, html = "") => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html) e.innerHTML = html;
  return e;
};

function fmt(n, dec = 1) {
  if (n === null || n === undefined || isNaN(n)) return "—";
  return Number(n).toFixed(dec);
}

function fmtSeconds(s) {
  const m = Math.floor(s / 60);
  const ss = Math.floor(s % 60);
  return `${String(m).padStart(2, "0")}:${String(ss).padStart(2, "0")}`;
}

function congColor(level) {
  return { LOW: "#10b981", MODERATE: "#f59e0b", HIGH: "#f97316", CRITICAL: "#ef4444" }[level] || "#64748b";
}

function actionColor(action) {
  return { DIVERT: "#f97316", MONITOR: "#f59e0b", NO_ACTION: "#10b981" }[action] || "#64748b";
}

function dirArrow(deg) {
  const dirs = ["→","↗","↑","↖","←","↙","↓","↘"];
  return dirs[Math.round(((deg % 360) + 360) % 360 / 45) % 8];
}

// ── Clock ──────────────────────────────────────────────────────
function tickClock() {
  const now = new Date();
  $("clockDisplay").textContent = now.toLocaleTimeString("en-GB", { hour12: false });
}
setInterval(tickClock, 1000);
tickClock();

// ── Build DOM for each camera ──────────────────────────────────
function buildCameraPanel(camId) {
  const meta = CAMERA_META[camId] || { label: camId, gateLabel: "Gate" };

  const panel = el("div", "camera-panel");
  panel.id = `panel-${camId}`;
  panel.innerHTML = `
    <!-- Header -->
    <div class="panel-header">
      <div class="panel-cam-info">
        <span class="panel-cam-id">${meta.label}</span>
        <span class="panel-gate-name" id="gateName-${camId}">${meta.gateLabel}</span>
      </div>
      <div class="panel-badges">
        <span class="trend-badge" id="trend-${camId}">—</span>
        <span class="congestion-badge cong-LOW" id="congBadge-${camId}">LOW</span>
      </div>
    </div>

    <!-- Stream tabs -->
    <div class="stream-tabs">
      <button class="tab-btn active" id="tabLive-${camId}"
        onclick="switchTab('${camId}','live')">📹 Live Video</button>
      <button class="tab-btn" id="tabDensity-${camId}"
        onclick="switchTab('${camId}','density')">🌡️ Density Map</button>
    </div>

    <!-- Video -->
    <div class="stream-container">
      <img class="stream-img" id="streamLive-${camId}"
           src="/stream/${camId}" alt="Live feed" />
      <img class="stream-img hidden" id="streamDensity-${camId}"
           src="" alt="Density map" data-src="/density/${camId}" />
      <div class="decision-overlay">
        <span class="decision-action-badge action-INITIALIZING" id="decBadge-${camId}">
          Initializing…
        </span>
        <span class="frame-info mono" id="frameInfo-${camId}">—</span>
      </div>
    </div>

    <!-- Stats grid -->
    <div class="stats-grid">
      <div class="stat-cell">
        <span class="stat-cell-label">Crowd Count</span>
        <span class="stat-cell-value" id="count-${camId}">—</span>
        <span class="stat-cell-unit">people</span>
      </div>
      <div class="stat-cell">
        <span class="stat-cell-label">Velocity</span>
        <span class="stat-cell-value" id="vel-${camId}">—</span>
        <span class="stat-cell-unit">px/s <span id="dir-${camId}"></span></span>
      </div>
      <div class="stat-cell">
        <span class="stat-cell-label">Tracks</span>
        <span class="stat-cell-value" id="tracks-${camId}">—</span>
        <span class="stat-cell-unit">persons</span>
      </div>
      <div class="stat-cell">
        <span class="stat-cell-label">Congestion</span>
        <span class="stat-cell-value" id="congScore-${camId}" style="font-size:16px">—</span>
        <span class="stat-cell-unit" id="congLevel-${camId}">score</span>
      </div>
    </div>

    <!-- Congestion bar -->
    <div class="congestion-bar-row">
      <div class="cong-bar-header">
        <span>Congestion Level</span>
        <span id="congPct-${camId}">0%</span>
      </div>
      <div class="cong-bar-track">
        <div class="cong-bar-fill" id="congBar-${camId}" style="width:0%;background:#10b981"></div>
      </div>
    </div>

    <!-- Sparkline -->
    <div class="sparkline-row">
      <span class="sparkline-label">Crowd Count History (last 30 samples)</span>
      <canvas class="sparkline-canvas" id="spark-${camId}" height="36"></canvas>
    </div>
  `;
  return panel;
}

function switchTab(camId, tab) {
  const liveImg    = $(`streamLive-${camId}`);
  const densityImg = $(`streamDensity-${camId}`);
  const tabLive    = $(`tabLive-${camId}`);
  const tabDensity = $(`tabDensity-${camId}`);

  if (tab === "live") {
    liveImg.classList.remove("hidden");
    densityImg.classList.add("hidden");
    tabLive.classList.add("active");
    tabDensity.classList.remove("active");
  } else {
    liveImg.classList.add("hidden");
    densityImg.classList.remove("hidden");
    // Lazy-load density stream
    if (!densityImg.src || densityImg.src === window.location.href) {
      densityImg.src = densityImg.dataset.src;
    }
    tabLive.classList.remove("active");
    tabDensity.classList.add("active");
  }
}

// ── Build decision cards ───────────────────────────────────────
function buildDecisionCard(camId) {
  const card = el("div", "decision-card");
  card.id = `dec-card-${camId}`;
  card.innerHTML = `
    <div class="decision-card-header">
      <span class="decision-card-title" id="decTitle-${camId}">Gate — Decision</span>
      <span class="decision-big-action big-INITIALIZING" id="decAction-${camId}">—</span>
    </div>
    <div class="decision-target" id="decTarget-${camId}"></div>
    <div class="confidence-row">
      <div class="confidence-header">
        <span>Confidence</span>
        <span id="decConf-${camId}">0%</span>
      </div>
      <div class="confidence-track">
        <div class="confidence-fill" id="confBar-${camId}" style="width:0%"></div>
      </div>
    </div>
    <div class="decision-reason" id="decReason-${camId}">Waiting for data…</div>
  `;
  return card;
}

// ── Telemetry cards ────────────────────────────────────────────
function buildTelemetry() {
  const row = $("telemetryRow");
  const items = [
    { id: "tele-device",   icon: "⚡", label: "Compute Device",    val: "—" },
    { id: "tele-csrnet",   icon: "🧠", label: "CSRNet Latency",     val: "— ms" },
    { id: "tele-bytetrack",icon: "👁️", label: "ByteTrack Latency",  val: "— ms" },
    { id: "tele-uptime",   icon: "⏱️", label: "Pipeline Uptime",    val: "00:00" },
    { id: "tele-evals",    icon: "📊", label: "Total Evaluations",  val: "0" },
    { id: "tele-motion",   icon: "🏃", label: "Motion Status",      val: "—" },
  ];
  items.forEach(it => {
    const card = el("div", "tele-card");
    card.innerHTML = `
      <span class="tele-icon">${it.icon}</span>
      <div class="tele-body">
        <span class="tele-label">${it.label}</span>
        <span class="tele-value mono" id="${it.id}">${it.val}</span>
      </div>`;
    row.appendChild(card);
  });
}

// ── Sparkline renderer ─────────────────────────────────────────
function drawSparkline(camId, data) {
  const canvas = $(`spark-${camId}`);
  if (!canvas || !data || data.length < 2) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.offsetWidth, H = canvas.offsetHeight || 36;
  canvas.width = W * dpr; canvas.height = H * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);

  const min = Math.min(...data) * 0.9;
  const max = Math.max(...data) * 1.1 || 1;
  const step = W / (data.length - 1);

  const points = data.map((v, i) => ({
    x: i * step,
    y: H - ((v - min) / (max - min)) * H * 0.85 - H * 0.05,
  }));

  // Fill gradient
  const grad = ctx.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0, "rgba(99,102,241,0.3)");
  grad.addColorStop(1, "rgba(99,102,241,0)");
  ctx.beginPath();
  ctx.moveTo(points[0].x, H);
  points.forEach(p => ctx.lineTo(p.x, p.y));
  ctx.lineTo(points[points.length - 1].x, H);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Line
  ctx.beginPath();
  points.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
  ctx.strokeStyle = "#818cf8";
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.stroke();

  // Last dot
  const last = points[points.length - 1];
  ctx.beginPath();
  ctx.arc(last.x, last.y, 3, 0, Math.PI * 2);
  ctx.fillStyle = "#a5b4fc";
  ctx.fill();
}

// ── Update camera panel from snapshot ─────────────────────────
function updateCamera(camId, cam) {
  const level  = cam.congestion_level || "LOW";
  const action = cam.action || "INITIALIZING";
  const score  = cam.congestion_score || 0;
  const trend  = cam.trend || "STABLE";

  // Congestion badge
  const congBadge = $(`congBadge-${camId}`);
  if (congBadge) {
    congBadge.textContent = level;
    congBadge.className = `congestion-badge cong-${level}`;
  }

  // Trend
  const trendEl = $(`trend-${camId}`);
  if (trendEl) {
    const arrows = { INCREASING: "↑", STABLE: "→", DECREASING: "↓" };
    trendEl.textContent = arrows[trend] || "→";
    trendEl.className = `trend-badge trend-${trend}`;
  }

  // Gate name
  const gateEl = $(`gateName-${camId}`);
  if (gateEl) gateEl.textContent = `Gate ${cam.gate_id}`;

  // Decision overlay badge
  const decBadge = $(`decBadge-${camId}`);
  if (decBadge) {
    let txt = action;
    if (action === "DIVERT" && cam.target_gate) txt = `DIVERT → ${cam.target_gate}`;
    decBadge.textContent = txt;
    decBadge.className = `decision-action-badge action-${action}`;
  }

  // Frame info
  const fi = $(`frameInfo-${camId}`);
  if (fi) fi.textContent = `#${cam.frame_id} | t=${cam.video_timestamp.toFixed(2)}s`;

  // Stats — animate changes
  function setVal(id, val, dec = 0) {
    const e = $(id);
    if (!e) return;
    const newTxt = dec === 0 ? String(Math.round(val)) : fmt(val, dec);
    if (e.textContent !== newTxt) {
      e.textContent = newTxt;
      e.classList.remove("updated");
      void e.offsetWidth;
      e.classList.add("updated");
    }
  }
  setVal(`count-${camId}`, cam.crowd_count, 1);
  setVal(`vel-${camId}`, cam.velocity, 1);
  setVal(`tracks-${camId}`, cam.track_count, 0);
  setVal(`congScore-${camId}`, cam.congestion_score * 100, 0);

  const dirEl = $(`dir-${camId}`);
  if (dirEl) dirEl.textContent = dirArrow(cam.direction || 0);

  const congLvl = $(`congLevel-${camId}`);
  if (congLvl) congLvl.textContent = level;

  // Congestion bar
  const bar = $(`congBar-${camId}`);
  const pct = $(`congPct-${camId}`);
  const pctVal = Math.round(score * 100);
  if (bar) { bar.style.width = `${pctVal}%`; bar.style.background = congColor(level); }
  if (pct) pct.textContent = `${pctVal}%`;

  // Sparkline
  drawSparkline(camId, cam.count_history || []);

  // Decision card
  const decCard = $(`dec-card-${camId}`);
  if (decCard) decCard.className = `decision-card action-${action}`;
  const decTitle = $(`decTitle-${camId}`);
  if (decTitle) decTitle.textContent = `Gate ${cam.gate_id} — Decision`;
  const decActionEl = $(`decAction-${camId}`);
  if (decActionEl) {
    decActionEl.textContent = action.replace("_", " ");
    decActionEl.className = `decision-big-action big-${action}`;
  }
  const decTargetEl = $(`decTarget-${camId}`);
  if (decTargetEl) {
    if (action === "DIVERT" && cam.target_gate) {
      decTargetEl.innerHTML = `Redirect flow to <span>${cam.target_gate}</span>`;
    } else {
      decTargetEl.textContent = action === "MONITOR" ? "Monitoring — no redirection" : "Normal operations";
    }
  }
  const confPct = Math.round((cam.confidence || 0) * 100);
  const confBar = $(`confBar-${camId}`);
  if (confBar) confBar.style.width = `${confPct}%`;
  const confLbl = $(`decConf-${camId}`);
  if (confLbl) confLbl.textContent = `${confPct}%`;
  const reasonEl = $(`decReason-${camId}`);
  if (reasonEl) reasonEl.textContent = cam.reason || "No reason available.";
}

// ── Update global telemetry ────────────────────────────────────
function updateTelemetry(snap) {
  const cameras = Object.values(snap.cameras || {});

  const avgCsrnet = cameras.reduce((s, c) => s + (c.csrnet_latency_ms || 0), 0) / (cameras.length || 1);
  const avgBt     = cameras.reduce((s, c) => s + (c.bytetrack_latency_ms || 0), 0) / (cameras.length || 1);
  const motion    = cameras.map(c => c.motion_status).join(" / ") || "—";

  const set = (id, val) => { const e = $(id); if (e) e.textContent = val; };
  set("tele-device",    (snap.device || "cpu").toUpperCase());
  set("tele-csrnet",    avgCsrnet > 0 ? `${Math.round(avgCsrnet)} ms` : "—");
  set("tele-bytetrack", avgBt > 0 ? `${Math.round(avgBt)} ms` : "—");
  set("tele-uptime",    fmtSeconds(snap.uptime_s || 0));
  set("tele-evals",     snap.total_evaluations || 0);
  set("tele-motion",    motion);

  // Navbar pills
  set("deviceBadge", (snap.device || "cpu").toUpperCase());
  set("uptimeBadge", fmtSeconds(snap.uptime_s || 0));
  set("evalBadge",   snap.total_evaluations || 0);

  // Pipeline status
  const badge = $("pipelineBadge");
  const statusTxt = $("pipelineStatus");
  if (snap.pipeline_running) {
    badge.classList.add("running");
    if (statusTxt) statusTxt.textContent = "Pipeline Running";
  } else {
    badge.classList.remove("running");
    if (statusTxt) statusTxt.textContent = "Initializing…";
  }
}

// ── Bootstrap DOM on first data ────────────────────────────────
let domBuilt = false;

function bootstrapDOM(snap) {
  if (domBuilt) return;
  domBuilt = true;

  const grid = $("camerasGrid");
  const decBar = $("decisionBar");

  const camIds = Object.keys(snap.cameras || {}).sort();
  camIds.forEach((camId, idx) => {
    const panel = buildCameraPanel(camId);
    panel.style.animationDelay = `${idx * 0.08}s`;
    grid.appendChild(panel);

    const card = buildDecisionCard(camId);
    card.style.animationDelay = `${idx * 0.08 + 0.2}s`;
    decBar.appendChild(card);
  });

  buildTelemetry();
}

// ── WebSocket connection ───────────────────────────────────────
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    console.log("[WS] connected");
    clearTimeout(reconnectTimer);
    const badge = $("pipelineBadge");
    if (badge) badge.classList.add("running");
  };

  ws.onmessage = (e) => {
    try {
      const snap = JSON.parse(e.data);
      latestSnap = snap;

      bootstrapDOM(snap);
      updateTelemetry(snap);
      renderAdvisories(snap.advisories || []);

      Object.entries(snap.cameras || {}).forEach(([camId, cam]) => {
        updateCamera(camId, cam);
      });
    } catch (err) {
      console.error("[WS] parse error", err);
    }
  };

  ws.onclose = () => {
    console.warn("[WS] disconnected — reconnecting in 2s");
    const badge = $("pipelineBadge");
    const statusTxt = $("pipelineStatus");
    if (badge) badge.classList.remove("running");
    if (statusTxt) statusTxt.textContent = "Reconnecting…";
    reconnectTimer = setTimeout(connect, 2000);
  };

  ws.onerror = () => ws.close();
}

// ── Advisory Panel renderer ────────────────────────────────────
function renderAdvisories(advisories) {
  const list  = $("advisoryList");
  const count = $("advisoryCount");
  if (!list) return;

  if (!advisories || advisories.length === 0) {
    list.innerHTML = `<div style="padding:16px 20px;color:var(--text-muted);font-size:13px;">No active advisories.</div>`;
    if (count) count.textContent = "0 advisories";
    return;
  }

  if (count) {
    const emg = advisories.filter(a => a.severity === "EMERGENCY").length;
    count.textContent = `${advisories.length} active${emg ? ` · ${emg} EMERGENCY` : ""}`;
    count.style.background = emg ? "rgba(239,68,68,0.2)" : "rgba(99,102,241,0.2)";
    count.style.color       = emg ? "#f87171" : "#a5b4fc";
    count.style.borderColor = emg ? "rgba(239,68,68,0.4)" : "rgba(99,102,241,0.3)";
  }

  // Only re-render if content changed (avoid flicker)
  const key = advisories.map(a => a.rule_id + a.gate_id + a.severity).join("|");
  if (list.dataset.lastKey === key) return;
  list.dataset.lastKey = key;

  list.innerHTML = advisories.map(adv => {
    const ts = adv.timestamp
      ? new Date(adv.timestamp * 1000).toLocaleTimeString("en-GB", { hour12: false })
      : "—";
    return `
      <div class="advisory-card adv-${adv.severity}">
        <div class="advisory-stripe"></div>
        <div class="advisory-body">
          <div class="advisory-row-top">
            <span class="adv-severity-badge">${adv.severity}</span>
            <span class="adv-gate-badge">${adv.gate_id}</span>
            <span class="adv-title">${adv.title}</span>
          </div>
          <div class="adv-description">${adv.description}</div>
          <div class="adv-action-label">Recommended Action</div>
          <div class="adv-action">${adv.action}</div>
        </div>
        <div class="adv-timestamp">${ts}</div>
      </div>`;
  }).join("");
}

// ── Start ──────────────────────────────────────────────────────
connect();

// Redraw sparklines on window resize
window.addEventListener("resize", () => {
  if (!latestSnap) return;
  Object.entries(latestSnap.cameras || {}).forEach(([camId, cam]) => {
    drawSparkline(camId, cam.count_history || []);
  });
});
