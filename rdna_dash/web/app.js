/* RDNA Dash front end. Vanilla JS, no build step, no CDN scripts. */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 0) => (v == null || isNaN(v) ? "–" : Number(v).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d }));
const ago = (t) => { if (!t) return ""; const s = Math.max(0, Date.now() / 1000 - t); return s < 60 ? Math.round(s) + "s" : s < 3600 ? Math.round(s / 60) + " min" : (s / 3600).toFixed(1) + " h"; };

const api = {
  async get(p) { const r = await fetch(p, { cache: "no-store" }); if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText); return r.json(); },
  async post(p, body) {
    const r = await fetch(p, { method: "POST", headers: { "Content-Type": "application/json", "X-Rdna-Dash": "1" }, body: JSON.stringify(body || {}) });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.error || r.statusText);
    return j;
  },
};

/* ---------- glossary: plain words for newcomers */
const GLOSSARY = {
  tp: ["Tensor parallel (TP)", "Splits every layer of the model across several GPUs so they work on each token together. TP=4 means four cards share the work."],
  pp: ["Pipeline parallel (PP)", "Gives each GPU a slice of the model's layers, like an assembly line. Used here for three-card setups."],
  p2p: ["PCIe P2P", "Lets GPUs copy memory straight to each other instead of going through the CPU. Faster for multi-GPU when the motherboard supports it."],
  ple: ["PLE table", "Flash-Next's large n-gram lookup table. It lives in system RAM, not on the GPUs. INT4 is a compressed copy at the same speed."],
  kv: ["KV cache", "GPU memory that holds the conversation so far. More KV cache means longer prompts and more users at once."],
  ttft: ["Time to first token", "How long you wait before the reply starts. Mostly the time to read your prompt (prefill)."],
  prefill: ["Prefill", "Reading the prompt. Measured in tokens per second; this is where long documents feel slow."],
  decode: ["Decode", "Writing the answer, one token at a time. Tokens per second per user."],
  tps: ["Tokens per second", "A token is roughly ¾ of a word. 30 tok/s reads faster than most people."],
  mtp: ["MTP", "Multi-token prediction: the model drafts a couple of tokens ahead and checks them, which speeds up decode."],
  awq: ["AWQ INT4", "A 4-bit compressed version of the model. Uses a quarter of the memory with little quality loss."],
  ecc: ["ECC", "Error-correcting memory on Pro cards. Safer, but it reserves about 2 GB of each 32 GB card."],
  hip: ["HIP_VISIBLE_DEVICES", "Which GPUs vLLM is allowed to use, by index."],
  venv: ["Host venv", "A private Python environment on this machine where the fork is compiled. Newest kernels, longer setup."],
  container: ["Container", "A prebuilt image (Docker or Podman) with everything inside. Fastest to start; updated less often."],
  prefix: ["Prefix cache", "Reuses work when prompts start the same way (system prompts, long chats). Higher hit rate = faster."],
  spec: ["Speculative acceptance", "Share of drafted tokens the model kept. Higher means MTP is paying off."],
  recall: ["Passkey recall", "We hide a number in the middle of the long prompt and ask for it. Shows the model really reads long context."],
  hops: ["Hops", "PCIe bridges a transfer has to pass through between two cards. Fewer is better; a shared switch beats a trip through the CPU."],
  compose: ["compose.yml", "A file that describes the container. Docker Compose and Podman both read it, so you can also run it by hand."],
};
const term = (text, key) => `<span class="term" tabindex="0" data-tip="${key}">${text}</span>`;

/* ---------- the address other machines use for the model server */
function publicUrl() {
  const ss = S.server, ch = S.plan.choice;
  if (ss.attached) return ss.url;
  if (ch.host === "0.0.0.0") return `http://${location.hostname}:${ch.port}/v1`;
  return ss.url;
}
function shareNote() {
  const ch = S.plan.choice;
  if (S.server.attached || ch.host === "0.0.0.0") return "";
  return `<span class="muted">Only this machine can reach it.</span><button class="ghost small" data-act="shareNet">Share on my network</button>`;
}

/* ---------- app state */
let S = null;          // /api/state
let T = null;          // /api/telemetry
let view = (location.hash || "#setup").slice(1);
const ui = { openStep: null, logs: {}, offsets: {}, change: false, compare: new Set(), benchSel: null, chat: [], chatBusy: false, slammed: {} };
const prevNums = {};

/* ---------- helpers */
function patch(el, html) {
  if (!el || el._html === html) return;
  const scrolls = $$(".log", el).map((l) => [l.dataset.k, l.scrollTop, l.scrollHeight - l.clientHeight - l.scrollTop < 30]);
  // Keep what someone is typing: live re-renders must not reset inputs or steal focus.
  const inputs = $$("input[id]", el).map((i) => [i.id, i.value, document.activeElement === i, i.selectionStart, i.selectionEnd]);
  el.innerHTML = html;
  el._html = html;
  for (const [id, value, focused, a, b] of inputs) {
    const i = el.querySelector("#" + CSS.escape(id));
    if (!i) continue;
    i.value = value;
    if (focused) { i.focus(); try { i.setSelectionRange(a, b); } catch { /* number inputs have no selection */ } }
  }
  for (const [k, top, atEnd] of scrolls) {
    const l = k && el.querySelector(`.log[data-k="${k}"]`);
    if (l) l.scrollTop = atEnd ? l.scrollHeight : top;
  }
}
function toast(msg) {
  const t = $("#toast"); t.textContent = msg; t.hidden = false;
  clearTimeout(t._h); t._h = setTimeout(() => (t.hidden = true), 2600);
}
function pill(level, text) { return `<span class="pill ${level}">${esc(text)}</span>`; }
function cmdBlock(text) { return `<div class="cmd">${esc(text)}<button class="ghost small copy" data-copy="${esc(text)}">Copy</button></div>`; }
function slamOnce(key) { if (ui.slammed[key]) return ""; ui.slammed[key] = true; return " slam"; }
const stepLevel = { done: "ok", running: "run", "needs-you": "warn", failed: "fail", cancelled: "warn", todo: "idle", recommended: "rec" };
const stepText = { done: "Done", running: "Running", "needs-you": "Needs you", failed: "Failed", cancelled: "Cancelled", todo: "To do", recommended: "Recommended" };

function countTo(el, value, digits) {
  if (!el) return;
  const key = el.dataset.k;
  const from = prevNums[key];
  prevNums[key] = value;
  if (value == null || isNaN(value)) { el.firstChild.nodeValue = "–"; return; }
  if (from == null || isNaN(from) || matchMedia("(prefers-reduced-motion: reduce)").matches) { el.firstChild.nodeValue = fmt(value, digits); return; }
  const t0 = performance.now();
  const step = (t) => {
    const k = Math.min(1, (t - t0) / 450);
    const e = 1 - Math.pow(1 - k, 3);
    el.firstChild.nodeValue = fmt(from + (value - from) * e, digits);
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

/* ---------- charts: thin lines, recessive grid, crosshair tooltip */
const charts = new Map();
function lineChart(id, opts) {
  const W = 640, H = opts.height || 190, L = 46, R = 12, Tp = 10, B = 26;
  const all = opts.series.flatMap((s) => s.pts.filter((p) => p[1] != null));
  const cats = opts.cats;
  const xs = cats ? cats.map((_, i) => i) : all.map((p) => p[0]);
  const x0 = cats ? 0 : Math.min(...xs), x1 = cats ? Math.max(1, cats.length - 1) : Math.max(...xs);
  let ymax = Math.max(opts.minMax || 1, ...all.map((p) => p[1]));
  const nice = (v) => { const p = Math.pow(10, Math.floor(Math.log10(v))); const m = v / p; return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p; };
  ymax = nice(ymax * 1.08);
  const sx = (x) => L + ((x - x0) / (x1 - x0 || 1)) * (W - L - R);
  const sy = (y) => Tp + (1 - y / ymax) * (H - Tp - B);
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => ymax * f);
  let g = `<g class="grid">${ticks.map((t) => `<line x1="${L}" x2="${W - R}" y1="${sy(t)}" y2="${sy(t)}"/>`).join("")}</g>`;
  g += `<g class="axis">${ticks.map((t) => `<text x="${L - 8}" y="${sy(t) + 4}" text-anchor="end">${opts.yFmt ? opts.yFmt(t) : fmt(t)}</text>`).join("")}`;
  if (cats) g += cats.map((c, i) => `<text x="${sx(i)}" y="${H - 6}" text-anchor="middle">${esc(c)}</text>`).join("");
  else if (all.length) { const now = x1; [0, 0.5, 1].forEach((f) => { const x = x0 + (x1 - x0) * f; g += `<text x="${sx(x)}" y="${H - 6}" text-anchor="${f === 0 ? "start" : f === 1 ? "end" : "middle"}">${f === 1 ? "now" : "-" + Math.round((now - x) / 60 * 10) / 10 + " min"}</text>`; }); }
  g += `</g>`;
  for (const s of opts.series) {
    const pts = s.pts.filter((p) => p[1] != null).map((p) => [cats ? p[0] : p[0], p[1]]);
    if (!pts.length) continue;
    const d = pts.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join("");
    if (opts.area && pts.length > 1) g += `<path d="${d}L${sx(pts[pts.length - 1][0]).toFixed(1)},${sy(0)}L${sx(pts[0][0]).toFixed(1)},${sy(0)}Z" style="fill:${s.color};opacity:.12"/>`;
    g += `<path d="${d}" fill="none" style="stroke:${s.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
    if (cats || pts.length < 3) g += pts.map((p) => `<circle cx="${sx(p[0])}" cy="${sy(p[1])}" r="4" style="fill:${s.color};stroke:var(--bg)" stroke-width="2"/>`).join("");
  }
  g += `<line class="xh" x1="0" x2="0" y1="${Tp}" y2="${H - B}" stroke="var(--line-2)" visibility="hidden"/>`;
  charts.set(id, { opts, sx, x0, x1, L, R, W, cats });
  const legend = opts.series.length > 1 ? `<div class="legend">${opts.series.map((s) => `<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join("")}</div>` : "";
  return `${legend}<div class="chart" data-chart="${id}"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(opts.label || "chart")}">${g}<rect x="${L}" y="0" width="${W - L - R}" height="${H}" fill="transparent" class="hit"/></svg></div>`;
}
document.addEventListener("mousemove", (ev) => {
  const box = ev.target.closest && ev.target.closest("[data-chart]");
  const tip = $("#tip");
  if (!box) { if (tip.dataset.chart) { tip.hidden = true; delete tip.dataset.chart; $$(".xh").forEach((l) => l.setAttribute("visibility", "hidden")); } return; }
  const c = charts.get(box.dataset.chart); if (!c) return;
  const svg = $("svg", box); const r = svg.getBoundingClientRect();
  const vx = ((ev.clientX - r.left) / r.width) * c.W;
  let best = null, bx = 0;
  if (c.cats) { const i = Math.round(((vx - c.L) / (c.W - c.L - c.R)) * (c.cats.length - 1)); best = Math.max(0, Math.min(c.cats.length - 1, i)); bx = c.sx(best); }
  else { const xv = c.x0 + ((vx - c.L) / (c.W - c.L - c.R)) * (c.x1 - c.x0); let d = Infinity; for (const s of c.opts.series) for (const p of s.pts) if (Math.abs(p[0] - xv) < d) { d = Math.abs(p[0] - xv); best = p[0]; } bx = c.sx(best); }
  if (best == null) return;
  const xh = $(".xh", svg); xh.setAttribute("x1", bx); xh.setAttribute("x2", bx); xh.setAttribute("visibility", "visible");
  const rows = c.opts.series.map((s) => { const p = s.pts.find((q) => q[0] === best); return p && p[1] != null ? `<div><span style="color:${s.color}">●</span> ${esc(s.name)}: <b style="display:inline">${c.opts.yFmt ? c.opts.yFmt(p[1]) : fmt(p[1], 1)}</b></div>` : ""; }).join("");
  const head = c.cats ? c.cats[best] : new Date(best * 1000).toLocaleTimeString();
  tip.innerHTML = `<b>${esc(head)}</b>${rows || "no data"}`;
  tip.hidden = false; tip.dataset.chart = "1";
  tip.style.left = Math.min(window.innerWidth - 330, ev.clientX + 14) + "px"; tip.style.top = ev.clientY + 14 + "px";
});

/* ---------- glossary tooltips */
function showTip(el) {
  const k = el.dataset.tip; const g = GLOSSARY[k]; if (!g) return;
  const tip = $("#tip"); tip.innerHTML = `<b>${esc(g[0])}</b>${esc(g[1])}`; tip.hidden = false; delete tip.dataset.chart;
  const r = el.getBoundingClientRect();
  tip.style.left = Math.min(window.innerWidth - 330, r.left) + "px"; tip.style.top = r.bottom + 8 + "px";
}
document.addEventListener("mouseover", (e) => { const t = e.target.closest && e.target.closest("[data-tip]"); if (t) showTip(t); });
document.addEventListener("mouseout", (e) => { if (e.target.closest && e.target.closest("[data-tip]")) $("#tip").hidden = true; });
document.addEventListener("focusin", (e) => { if (e.target.dataset && e.target.dataset.tip) showTip(e.target); });
document.addEventListener("focusout", (e) => { if (e.target.dataset && e.target.dataset.tip) $("#tip").hidden = true; });

/* ---------- global clicks */
document.addEventListener("click", async (e) => {
  const c = e.target.closest("[data-copy]");
  if (c) { try { await navigator.clipboard.writeText(c.dataset.copy); toast("Copied"); } catch { toast("Select and copy it by hand"); } return; }
  const a = e.target.closest("[data-act]");
  if (!a) return;
  const act = a.dataset.act, arg = a.dataset.arg;
  try {
    a.disabled = true;
    await actions[act](arg, a);
  } catch (err) { toast(err.message || String(err)); }
  finally { a.disabled = false; }
});

const actions = {
  async autopilot() { await api.post("/api/autopilot"); toast("Setting everything up"); await refresh(); },
  async autopilotStop() { await api.post("/api/autopilot", { cancel: true }); await refresh(); },
  async recheck() { await api.post("/api/profile/refresh"); toast("Checked again"); await refresh(); },
  async run(id) { const r = await api.post(`/api/steps/${id}/run`); ui.openStep = id; if (r.job) { ui.logs[r.job.id] = ""; ui.offsets[r.job.id] = 0; } await refresh(); },
  async reset(id) { await api.post(`/api/steps/${id}/reset`); await refresh(); },
  async open(id) { ui.openStep = ui.openStep === id ? null : id; if (ui.openStep) await loadStepLog(id); renderSetup(); },
  async cancelJob(id) { await api.post(`/api/jobs/${id}/cancel`); await refresh(); },
  async toggleChange() { ui.change = !ui.change; renderSetup(true); },
  async alt(json) { await api.post("/api/prefs", JSON.parse(json)); toast("Plan updated"); await refresh(); },
  async savePrefs() {
    const f = $("#change-form"); const v = (n) => f.elements[n].value;
    await api.post("/api/prefs", { recipe: v("recipe"), path: v("path"), port: v("port") ? Number(v("port")) : null, max_len: v("max_len") ? Number(v("max_len")) : null,
      root: v("root") || null, ple: v("ple"), lan: f.elements.lan.checked, label: v("label") });
    ui.change = false; toast("Saved. The plan was re-tuned."); await refresh(); renderSetup(true);
  },
  async start() { await api.post("/api/server/start"); toast("Starting the server"); await refresh(); },
  async stop() { if (!confirm("Stop the model server?")) return; await api.post("/api/server/stop"); toast("Stopped"); await refresh(); },
  async restart() { await api.post("/api/server/restart"); toast("Restarting"); await refresh(); },
  async attach() { const u = $("#attach-url").value.trim(); await api.post("/api/attach", { url: u }); toast(u ? "Watching " + u : "Detached"); await refresh(); },
  async attachTo(url) { await api.post("/api/attach", { url }); toast("Watching " + url); await refresh(); },
  async detach() { await api.post("/api/attach", { url: "" }); await refresh(); },
  async smoke() { const r = await api.post("/api/smoke"); toast(r.ok ? "It answered" : "Failed: " + r.error); await refresh(); },
  async bench(suite) { await api.post("/api/bench/start", { suite: $('input[name="suite"]:checked').value, thinking_off: !$("#bench-think").checked }); await loadBench(); },
  async benchCancel() { await api.post("/api/bench/cancel"); },
  async benchDel(id) { if (!confirm("Delete this run?")) return; await api.post("/api/bench/delete", { id }); ui.compare.delete(id); await loadBench(); },
  async benchPick(id) { ui.benchSel = id; renderBench(); },
  async topoScan() { await loadTopo(true); toast("Rescanned"); },
  async bandwidth(tool) { const r = await api.post("/api/topology/bandwidth", { tool }); ui.topoJob = r.job.id; ui.logs[r.job.id] = ""; ui.offsets[r.job.id] = 0; toast("Measuring"); },
  async fleetAdd() { const f = $("#fleet-form"); await api.post("/api/fleet", { name: f.elements.name.value, url: f.elements.url.value, token: f.elements.token.value }); f.reset(); await loadFleet(); },
  async fleetRm(name) { await api.post("/api/fleet/remove", { name }); await loadFleet(); },
  async share() { const r = await api.get("/api/profile/share"); const t = JSON.stringify(r, null, 2); try { await navigator.clipboard.writeText(t); toast("Redacted report copied"); } catch { alert(t); } },
  async chatClear() { ui.chat = []; renderChat(); },
  async chatSend() { await sendChat(); },
};

/* ---------- data loading */
async function refresh() {
  try { S = await api.get("/api/state"); } catch (e) { $("#rail-status").innerHTML = pill("fail", "Dashboard offline"); return; }
  $("#demo-banner").hidden = !S.demo;
  let seen = false; try { seen = localStorage.getItem("rdna-open-ok") === "1"; } catch { /* private mode */ }
  $("#open-banner").hidden = !S.open_access || seen;
  renderRail();
  if (view === "setup") renderSetup();
  if (view === "dash") renderDash();
  if (view === "logs") renderLogs();
  if (ui.openStep) loadStepLog(ui.openStep);
}
async function loadTelemetry() {
  try { T = await api.get("/api/telemetry"); } catch { return; }
  if (view === "dash") renderDash();
}
async function loadStepLog(id) {
  const st = S && S.steps[id]; if (!st || !st.job) return;
  try {
    const off = ui.offsets[st.job] || 0;
    const r = await api.get(`/api/jobs/${st.job}?offset=${off}`);
    ui.logs[st.job] = (ui.logs[st.job] || "") + r.text; ui.offsets[st.job] = r.offset; ui.problems = r.problems;
    renderSetup();
  } catch {
    const r = await api.get(`/api/steps/log?step=${id}`).catch(() => null);
    if (r) { ui.logs[st.job] = r.text; ui.problems = r.problems; renderSetup(); }
  }
}

/* ---------- rail */
function renderRail() {
  $$(".rail a").forEach((a) => a.classList.toggle("active", a.dataset.view === view));
  const s = S.server.state;
  const txt = { up: "Serving", starting: "Starting", crashed: "Server stopped with an error", stopped: "Server off" }[s];
  const lvl = { up: "ok", starting: "run", crashed: "fail", stopped: "idle" }[s];
  patch($("#rail-status"), `${pill(lvl, txt)}<div class="muted" style="margin-top:8px">${esc(S.label)}</div>`);
}

/* ======================================================================= SETUP */
function renderSetup(force) {
  const el = $("#view-setup"); if (!S) return;
  if (!el.dataset.built || force) {
    el.innerHTML = `<div id="s-hero"></div><div id="s-needs"></div><div class="sec"><h2>The steps<span class="right"><button class="ghost small" data-act="recheck">Re-check this box</button></span></h2><div id="s-steps"></div></div>
      <div class="sec"><h2>Your tuned setup<span class="right"><button class="ghost small" data-act="toggleChange">${ui.change ? "Close" : "Change"}</button></span></h2><div id="s-change"></div><div id="s-plan"></div></div>
      <div class="sec"><h2>Make it faster (optional)</h2><div id="s-tune"></div></div>`;
    el.dataset.built = "1";
    if (ui.change) $("#s-change").innerHTML = changeForm();
  }
  const P = S.plan, sm = P.summary, ch = P.choice, ap = S.autopilot;
  const up = S.server.state === "up";
  let title;
  if (up) title = `You're <span class="key">serving.</span>`;
  else if (ap.status === "running") title = `Setting up <span class="key">${esc(S.label)}</span>`;
  else if (P.verdict === "blocked") title = `A few things <span class="key">first.</span>`;
  else title = `Let's get <span class="key">${esc(S.label)}</span> serving.`;
  const facts = [["GPUs", sm.gpus], ["Memory", fmt(sm.ram_gb) + " GB RAM"], ["ROCm", sm.rocm], ["System", sm.os]];
  const apLine = ap.status === "idle" ? "" :
    `<div class="row" style="margin-top:16px">${pill({ running: "run", done: "ok", "needs-you": "warn", failed: "fail", cancelled: "warn" }[ap.status] || "idle", { running: "Working", done: "Finished", "needs-you": "Waiting on you", failed: "Stopped on an error", cancelled: "Stopped" }[ap.status] || ap.status)}<span>${esc(ap.message || "")}</span>${ap.status === "running" ? `<button class="ghost small" data-act="autopilotStop">Stop</button>` : ""}</div>`;
  const busy = ap.status === "running";
  patch($("#s-hero"), `
    <h1 class="punch${slamOnce("setup")}">${title}</h1>
    <p class="lede">We looked at this machine and picked the setup that fits it best: <b>${esc(ch.recipe_label)}</b>,
      ${ch.path === "container" ? `run in a ${term("container", "container")}` : `built on the host in a ${term("venv", "venv")} for the newest kernels`},
      on ${ch.pp > 1 ? `${ch.pp} GPUs with ${term("PP", "pp")}` : ch.tp > 1 ? `${ch.tp} GPUs with ${term("TP", "tp")}` : "one GPU"}.
      Press the button and we'll do the rest. Anything that needs your password shows up as a command to paste.</p>
    <div class="facts">${facts.map(([k, v]) => `<div class="fact"><div class="k">${k}</div><div class="v">${esc(v)}</div></div>`).join("")}</div>
    <div class="row">
      ${up ? `<a class="btn go" href="#chat">Try it in chat</a><a class="btn alt" href="#dash">Open the dashboard</a>`
           : `<button class="btn go" data-act="autopilot" ${busy || P.verdict === "blocked" ? "disabled" : ""}>Set everything up</button>
              <span class="muted">${P.verdict === "blocked" ? "Fix the items below first, then Re-check." : "or go step by step below."}</span>`}
    </div>${apLine}`);

  const needs = P.checks.filter((c) => c.level === "fail" || c.level === "warn");
  patch($("#s-needs"), needs.length ? `<div class="sec"><h2>Needs you</h2>${needs.map(checkRow).join("")}</div>` : "");

  const cur = P.steps.find((s) => !["done", "recommended"].includes(S.steps[s.id].status));
  patch($("#s-steps"), `<ol class="steps">${P.steps.map((s) => stepRow(s, cur && cur.id === s.id)).join("")}</ol>`);

  patch($("#s-plan"), `<div class="kv">${P.settings.map((s) => `<div class="k">${glossKey(s.key)}</div><div class="v">${esc(s.key === "Endpoint" ? s.value.replace("this-box", location.hostname) : s.value)}<span class="why">${esc(s.why)}</span></div>`).join("")}</div>
    ${P.alternatives.length ? `<p class="eyebrow" style="margin-top:22px">Other good options</p><div class="row">${P.alternatives.map((a) => `<button class="ghost" data-act="alt" data-arg='${esc(JSON.stringify(a.prefs))}' title="${esc(a.why)}">${esc(a.label)}</button>`).join("")}</div>` : ""}
    <details style="margin-top:18px"><summary>Show the exact launch command</summary>${cmdBlock(P.launch.command)}${P.launch.compose_yaml ? `<p class="muted" style="margin:12px 0 0">${term("compose.yml", "compose")} written to ${esc(P.launch.compose_file)}</p>${cmdBlock(P.launch.compose_yaml)}` : ""}</details>
    ${S.registry_errors && S.registry_errors.length ? `<div class="problem"><div class="t">Some recipe files were skipped</div><div class="d">${S.registry_errors.map(esc).join("<br>")}</div></div>` : ""}`);

  const tune = P.checks.filter((c) => ["iommu"].includes(c.id));
  const multi = ch.tp * ch.pp > 1;
  patch($("#s-tune"), (tune.length ? tune.map(checkRow).join("") : `<p class="empty">Nothing to tune right now.</p>`) +
    (multi ? `<div class="check"><span>${pill("info", "P2P")}</span><div class="t">See how your GPUs talk to each other</div><div class="d">Map the PCIe paths, count the ${term("hops", "hops")}, and measure real GPU-to-GPU bandwidth. The result tunes ${term("P2P", "p2p")} for you.</div><div class="fix"><a class="ghost" href="#topo" style="display:inline-block;text-decoration:none">Open Topology</a></div></div>` : ""));
}
function glossKey(k) {
  const map = { Parallelism: "tp", P2P: "p2p", PLE: "ple", HIP_VISIBLE_DEVICES: "hip", "Install path": null };
  const g = map[k]; return g ? term(esc(k), g) : esc(k);
}
function checkRow(c) {
  return `<div class="check"><span>${pill(c.level, { ok: "OK", info: "Tip", rec: "Recommended", warn: "Check", fail: "Needs you" }[c.level])}</span>
    <div class="t">${esc(c.title)}</div><div class="d">${esc(c.detail)}${c.link ? ` <a href="${esc(c.link)}" target="_blank" rel="noopener">Read more</a>` : ""}</div>
    ${c.fix && c.fix.length ? `<div class="fix">${cmdBlock(c.fix.join("\n"))}</div>` : ""}
    ${c.attach ? `<div class="fix row"><button class="btn small" data-act="attachTo" data-arg="${esc(c.attach)}">Watch this server</button><button class="ghost small" data-act="toggleChange">Pick another port</button></div>` : ""}</div>`;
}
function stepRow(s, isCur) {
  const st = S.steps[s.id] || { status: "todo" };
  const status = st.status;
  const running = status === "running";
  const job = st.job;
  const busy = S.job && S.job.status === "running";
  let acts = "";
  if (s.kind === "run") {
    if (running) acts = `<button class="ghost small" data-act="cancelJob" data-arg="${job}">Cancel</button>`;
    else acts = `<button class="${isCur ? "btn" : "ghost"} small" data-act="run" data-arg="${s.id}" ${busy ? "disabled" : ""}>${status === "done" ? "Run again" : status === "failed" ? "Try again" : "Run this step"}</button>`;
    if (job) acts += `<button class="ghost small" data-act="open" data-arg="${s.id}">${ui.openStep === s.id ? "Hide details" : "Show details"}</button>`;
    if (s.optional && status !== "done") acts += `<span class="muted" style="font-size:13px">Optional</span>`;
  } else if (s.kind === "check") {
    acts = status === "needs-you" ? `<button class="ghost small" data-act="recheck">Re-check</button>` : "";
  } else if (s.kind === "tune") {
    acts = status === "recommended" ? `<button class="ghost small" data-act="recheck">Re-check after rebooting</button>` : "";
  } else if (s.kind === "write") {
    acts = `<button class="ghost small" data-act="run" data-arg="${s.id}">${status === "done" ? "Write again" : "Write it"}</button>`;
  } else if (s.kind === "serve") {
    const ss = S.server;
    acts = ss.state === "up" ? `<a class="ghost small" href="#dash" style="text-decoration:none">Watch it live</a>` : ss.state === "starting"
      ? `<span class="muted">${esc(ss.phase || "Booting")} · ${ago(ss.started)}</span>` : `<button class="${isCur ? "btn" : "ghost"} small" data-act="start">Start the server</button>`;
  } else if (s.kind === "smoke") {
    acts = `<button class="${isCur ? "btn" : "ghost"} small" data-act="smoke" ${S.server.state !== "up" ? "disabled" : ""}>Send a test message</button>`;
    if (S.smoke && S.smoke.ok) acts += `<span class="muted">Answered in ${fmt(S.smoke.e2e, 1)} s, first token after ${fmt(S.smoke.ttft, 2)} s: “${esc((S.smoke.text || "").slice(0, 120))}”</span>`;
    if (S.smoke && !S.smoke.ok) acts += `<span class="muted">${esc(S.smoke.error)}</span>`;
  }
  let extra = "";
  if (s.kind === "tune") extra += `<div class="tune-list">${s.checks.map(checkRow).join("")}</div>`;
  if (s.kind === "check" && s.blocked_by && s.blocked_by.length) extra = `<p class="muted">Waiting on: ${s.blocked_by.map(esc).join(", ")}. The commands are under “Needs you”.</p>`;
  if (s.download) {
    const pr = st.progress;
    const lim = (S.prefs && S.prefs.download_mbps) || 0;
    if (pr && pr.total) {
      const pct = Math.min(100, (100 * pr.done) / pr.total);
      extra += `<div class="big-progress"><i style="width:${pct}%"></i></div><p class="muted">${fmt(pct, 0)}% · ${fmt(pr.done / 1e9, 1)} of ${fmt(pr.total / 1e9, 1)} GB · ${fmt(pr.rate / 1e6, 0)} MB/s${pr.eta ? " · about " + ago(Date.now() / 1000 - pr.eta) + " left" : ""} · file ${pr.file} of ${pr.files}</p>`;
    }
    extra += `<div class="row dl-limit"><label for="dl-limit" class="muted">Max download speed</label><input id="dl-limit" class="in" type="number" min="0" step="1" value="${lim || ""}" placeholder="no limit"><span class="muted">MB/s</span><button class="ghost small" data-act="dlLimit">Apply</button><span class="muted">${running ? "Changes take effect within a second." : "Leave empty for no limit."}</span></div>`;
  }
  if (ui.openStep === s.id && job) {
    extra += `<div class="log" data-k="${s.id}">${esc(ui.logs[job] || "Loading…")}</div>`;
    if (status === "failed" && ui.problems && ui.problems.length) extra += ui.problems.map(problemRow).join("");
  }
  if (s.kind === "serve" && S.server.problems && S.server.problems.length) extra += S.server.problems.map(problemRow).join("");
  return `<li class="step ${status}${isCur ? " current" : ""}"><h3>${esc(s.title)} ${pill(stepLevel[status] || "idle", stepText[status] || status)}</h3>
    <div><p>${esc(s.summary)}</p><div class="acts">${acts}</div>${extra}</div></li>`;
}
function problemRow(p) {
  return `<div class="problem"><div class="t">${esc(p.title)}</div><div class="d">${esc(p.detail)}${p.link ? ` <a href="${esc(p.link)}" target="_blank" rel="noopener">How to fix</a>` : ""}</div><code>${esc(p.line)}</code></div>`;
}
function changeForm() {
  const P = S.plan, pr = S.prefs || {};
  const rec = P.registry.map((r) => `<option value="${esc(r.id)}" ${pr.recipe === r.id ? "selected" : ""} ${r.fits ? "" : ""}>${esc(r.label)}${r.fits ? "" : " (doesn't fit: " + esc(r.why) + ")"}${r.verify ? " · needs verify" : ""}</option>`).join("");
  return `<form id="change-form" class="form" onsubmit="return false" style="margin-bottom:22px">
    <div class="field"><label>Model and layout</label><select name="recipe"><option value="auto">Best fit (recommended)</option>${rec}</select></div>
    <div class="field"><label>How to install</label><select name="path">${[["auto", "Best fit (recommended)"], ["container", "Container (Docker / Podman)"], ["venv", "Build on the host"]].map(([v, l]) => `<option value="${v}" ${pr.path === v ? "selected" : ""}>${l}</option>`).join("")}</select></div>
    <div class="field"><label>${term("PLE table", "ple")} (Flash-Next)</label><select name="ple">${["auto", "int4", "int4-disk", "bf16"].map((v) => `<option ${pr.ple === v ? "selected" : ""}>${v}</option>`).join("")}</select></div>
    <div class="field"><label>Port</label><input name="port" type="number" min="1" max="65535" placeholder="${P.choice.port}" value="${pr.port || ""}"></div>
    <div class="field"><label>Max context (tokens)</label><input name="max_len" type="number" min="1024" step="1024" placeholder="model default" value="${pr.max_len || ""}"></div>
    <div class="field"><label>Install folder</label><input name="root" placeholder="${esc(P.paths.root)}" value="${esc(pr.root || "")}"><div class="hint">Venv, fork checkout, models and caches go here.</div></div>
    <div class="field"><label>Name for this box</label><input name="label" placeholder="${esc(S.label)}" maxlength="60"></div>
    <div class="field"><label>&nbsp;</label><label class="switch"><input name="lan" type="checkbox" ${pr.lan ? "checked" : ""}> Share the model on my network</label><div class="hint">Off: only this machine can reach it.</div></div>
    <div class="field"><label>&nbsp;</label><button class="btn" data-act="savePrefs">Save and re-tune</button></div></form>`;
}

/* ======================================================================= DASHBOARD */
function spark(vals, color, h = 34) {
  const v = vals.filter((x) => x != null && !isNaN(x));
  if (v.length < 2) return `<svg class="spark" viewBox="0 0 120 ${h}" preserveAspectRatio="none" aria-hidden="true"></svg>`;
  const max = Math.max(...v, 1e-9), n = vals.length;
  const pts = vals.map((x, i) => [(i / (n - 1)) * 120, h - 2 - ((x == null ? 0 : x) / max) * (h - 6)]);
  const d = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join("");
  return `<svg class="spark" viewBox="0 0 120 ${h}" preserveAspectRatio="none" aria-hidden="true">
    <path d="${d}L120,${h}L0,${h}Z" style="fill:${color};opacity:.14"/><path d="${d}" fill="none" style="stroke:${color}" stroke-width="1.6" vector-effect="non-scaling-stroke" stroke-linejoin="round"/></svg>`;
}
function ring(pct, color) {
  const r = 30, c = 2 * Math.PI * r, p = Math.max(0, Math.min(100, pct || 0));
  return `<svg class="ring" viewBox="0 0 76 76" aria-hidden="true"><circle cx="38" cy="38" r="${r}" style="stroke:var(--bg-3)" stroke-width="7" fill="none"/>
    <circle cx="38" cy="38" r="${r}" style="stroke:${color};transition:stroke-dashoffset .5s" stroke-width="7" fill="none" stroke-linecap="round"
      stroke-dasharray="${c.toFixed(1)}" stroke-dashoffset="${(c * (1 - p / 100)).toFixed(1)}" transform="rotate(-90 38 38)"/></svg>`;
}
const info = (key) => `<span class="q" tabindex="0" data-tip="${key}" aria-label="What is this?">i</span>`;

function renderDash() {
  const el = $("#view-dash"); if (!S) return;
  if (!el.dataset.built) {
    el.innerHTML = `<div id="d-hero"></div><div id="d-problems"></div><div id="d-tiles" class="tiles"></div>
      <div class="grid2"><div class="panel" id="d-c1"></div><div class="panel" id="d-c2"></div></div>
      <div class="panel-head"><h2>GPUs</h2><span id="d-gpu-sum" class="muted"></span></div><div id="d-gpus" class="gpu-grid"></div>
      <div class="grid2"><div class="panel" id="d-host"></div><div class="panel" id="d-connect"></div></div>`;
    el.dataset.built = "1";
  }
  const ss = S.server, lat = (T && T.latest) || {}, srv = lat.server || {}, hist = (T && T.server) || [];
  const tail = (k) => hist.slice(-60).map((s) => s[k]);
  // Bursty numbers (prefill, first token, hit rates) are 0 or empty between requests. Show the
  // last real value while idle, and a 1-minute average of the busy moments under it.
  const lastReal = (key) => { for (let i = hist.length - 1; i >= 0; i--) { const v = hist[i][key]; if (v != null && v > 0) return [v, hist[i].t]; } return [null, null]; };
  const busyAvg = (key) => { const now = Date.now() / 1000; const v = hist.filter((s) => s.t > now - 60 && s[key] != null && s[key] > 0).map((s) => s[key]); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
  const live = (key, d, unit) => {
    const cur = srv[key];
    if (cur != null && cur > 0) { const a = busyAvg(key); return { v: cur, note: a != null ? `1-min avg ${fmt(a, d)}${unit}` : null, idle: false }; }
    const [v, t] = lastReal(key);
    return v == null ? { v: cur, note: null, idle: false } : { v, note: `idle · last ${ago(t)} ago`, idle: true };
  };
  const word = { up: "Serving", starting: "Starting", crashed: "Stopped", stopped: "Off" }[ss.state];
  const dot = { up: "ok", starting: "run", crashed: "fail", stopped: "idle" }[ss.state];
  const model = (S.smoke && S.smoke.models && S.smoke.models[0]) || (ss.attached ? "The server" : S.plan.choice.served_name || S.plan.choice.recipe_label);
  const sub = ss.state === "up" ? `<b>${esc(model)}</b> is answering requests` : ss.state === "starting"
    ? `${esc(ss.phase || "Booting")} · ${ago(ss.started)} so far. First boot compiles kernels; later boots are quick.`
    : ss.state === "crashed" ? "The server exited. Known problems from its log are listed below." : "The model server is not running.";
  const btns = ss.attached ? `<button class="ghost" data-act="detach">Stop watching</button>` : ss.state === "up" || ss.state === "starting"
    ? `<button class="ghost" data-act="restart">Restart</button><button class="ghost" data-act="stop">Stop</button>`
    : `<button class="btn go" data-act="start">Start the server</button>`;
  const ch = S.plan.choice;
  const layout = ss.attached ? "A server you run yourself: this dashboard only watches it" : [ch.recipe_label, ch.pp > 1 ? `PP=${ch.pp}` : ch.tp > 1 ? `TP=${ch.tp}` : "1 GPU", ch.p2p == null ? null : `P2P ${ch.p2p ? "on" : "off"}`].filter(Boolean).join(" · ");
  const hero = $("#d-hero");
  if (!hero.dataset.built) {
    hero.innerHTML = `<div class="panel hero-panel"><div class="hero-l"><div id="dh-state"></div><h1 class="punch${slamOnce("dash")}" id="dh-word"></h1><p class="hero-sub" id="dh-sub"></p>
        <div class="row" id="dh-meta"></div></div>
      <div class="hero-r"><div class="eyebrow">Output speed ${info("tps")}</div><div class="hero-num"><span data-k="kpi-gen">–</span><small>tok/s</small></div><div class="hero-note" id="dh-gen-note"></div><div id="dh-spark"></div></div></div>
      <div id="dh-attach"></div>`;
    hero.dataset.built = "1";
  }
  patch($("#dh-state"), `<div class="row hero-top">${pill(dot, word)}<span class="muted">${esc(layout)}</span><span class="spacer"></span>${btns}</div>`);
  patch($("#dh-word"), ss.state === "up" ? `<span class="key">${word}</span>` : word);
  patch($("#dh-sub"), sub);
  const pub = publicUrl();
  patch($("#dh-meta"), `<span class="chip mono">${esc(pub)}</span><button class="ghost small" data-copy="${esc(pub)}">Copy URL</button>${shareNote()}`);
  const G = live("gen_tps", 1, " tok/s");
  countTo($('[data-k="kpi-gen"]'), G.v, 1);
  const gn = $('[data-k="kpi-gen"]'); if (gn) gn.parentElement.classList.toggle("idle", G.idle);
  patch($("#dh-gen-note"), G.note ? esc(G.note) : "");
  patch($("#dh-spark"), spark(tail("gen_tps"), "var(--key)", 44));
  patch($("#dh-attach"), ss.state !== "up" && !ss.attached ? `<details class="attach"><summary>Already running vLLM yourself? Watch that server instead</summary><div class="row" style="margin-top:10px"><input id="attach-url" class="in" placeholder="http://127.0.0.1:8000/v1"><button class="ghost" data-act="attach">Watch it</button></div></details>` : "");
  patch($("#d-problems"), ss.problems && ss.problems.length ? `<div class="panel warn-panel"><div class="eyebrow">Problems found in the log</div>${ss.problems.map(problemRow).join("")}</div>` : "");

  const ext = srv.ext_prefix_hit != null ? ` · ${fmt(srv.ext_prefix_hit)}% from offload/connector` : (lastReal("ext_prefix_hit")[0] != null ? ` · ${fmt(lastReal("ext_prefix_hit")[0])}% from offload/connector` : "");
  const kc = srv.kv_config || {};
  const kvNote = [kc.tokens ? `of ${fmt(kc.tokens)} tokens on GPU` : "of GPU cache in use",
    srv.kv_offload_usage != null ? `${fmt(srv.kv_offload_usage)}% offload in use`
      : kc.offload_on ? `offload on${kc.offload_gb ? " · " + fmt(kc.offload_gb) + " GB" : ""}` : kc.offload_backend ? "offload off" : null]
    .filter(Boolean).join(" · ");
  const L = { prompt: live("prompt_tps", 0, " tok/s"), ttft: live("ttft", 2, " s"), pc: live("prefix_hit", 0, "%"), sd: live("spec_accept", 0, "%") };
  const tiles = [
    ["prompt", "Prompt reading", "tok/s", L.prompt.v, 0, "prefill", "prompt_tps", "var(--s3)", L.prompt.note || "Prefill, all users", L.prompt.idle],
    ["run", "Active requests", "", srv.running, 0, null, "running", "var(--s1)", `${fmt(srv.waiting)} waiting in line`, false],
    ["ttft", "First token", "s", L.ttft.v, 2, "ttft", "ttft", "var(--s1)", L.ttft.note || "Average of recent requests", L.ttft.idle],
    ["kv", "KV cache", "%", srv.kv_usage, 0, "kv", "kv_usage", "var(--s3)", kvNote, false],
    ["pc", "Prefix cache hits", "%", L.pc.v, 0, "prefix", "prefix_hit", "var(--s1)", (L.pc.note || "Reused prompt work") + ext, L.pc.idle],
    ["sd", "Draft accepted", "%", L.sd.v, 0, "spec", "spec_accept", "var(--s3)", L.sd.note || "MTP drafts kept", L.sd.idle],
  ];
  const tl = $("#d-tiles");
  if (!tl.dataset.built) {
    tl.innerHTML = tiles.map(([id, l, u, , , g]) => `<div class="tile"><div class="tile-l">${l}${g ? " " + info(g) : ""}</div>
      <div class="tile-n"><span data-k="kpi-${id}">–</span><small>${u}</small></div><div class="tile-s" id="ts-${id}"></div><div class="tile-sp" id="tp-${id}"></div></div>`).join("");
    tl.dataset.built = "1";
  }
  for (const [id, , , v, d, , key, color, note, idle] of tiles) {
    countTo($(`[data-k="kpi-${id}"]`), v, d);
    const n = $(`[data-k="kpi-${id}"]`); if (n) n.parentElement.classList.toggle("idle", !!idle);
    patch($("#ts-" + id), esc(note));
    patch($("#tp-" + id), spark(tail(key), color));
  }

  patch($("#d-c1"), `<div class="panel-head"><h2>Output speed</h2><span class="muted">tok/s, last 5 min</span></div>` +
    lineChart("c-gen", { label: "Output tokens per second", area: true, height: 200, series: [{ name: "Output tok/s", color: "var(--s1)", pts: hist.map((s) => [s.t, s.gen_tps]) }], yFmt: (v) => fmt(v, v < 10 ? 1 : 0) }));
  patch($("#d-c2"), `<div class="panel-head"><h2>Prompt reading</h2><span class="muted">tok/s, last 5 min</span></div>` +
    lineChart("c-prompt", { label: "Prompt tokens per second", area: true, height: 200, series: [{ name: "Prompt tok/s", color: "var(--s3)", pts: hist.map((s) => [s.t, s.prompt_tps]) }] }));

  const gpus = lat.gpus || [];
  const names = Object.fromEntries((S.profile.gpus || []).map((g) => [g.index, g]));
  const used = (ch.hip_visible_devices || "").split(",").filter(Boolean).map(Number);
  const gh = (T && T.gpu) || [];
  const totPow = gpus.reduce((a, g) => a + (g.power || 0), 0);
  patch($("#d-gpu-sum"), gpus.length ? `${gpus.length} cards · ${fmt(totPow)} W total · hottest ${fmt(Math.max(...gpus.map((g) => g.temp_junction || g.temp_edge || 0)))}°C` : "");
  patch($("#d-gpus"), gpus.length ? gpus.map((g) => {
    const n = names[g.index] || {};
    const mem = g.vram_total ? (100 * g.vram_used) / g.vram_total : 0;
    const pw = g.power_cap ? (100 * (g.power || 0)) / g.power_cap : 0;
    const t = g.temp_junction || g.temp_edge;
    const hot = (t || 0) >= 95;
    const idle = used.length && !used.includes(g.index);
    const utilHist = gh.slice(-60).map((s) => ((s.gpus || []).find((x) => x.index === g.index) || {}).util);
    return `<div class="gpu-card${idle ? " idle" : ""}"><div class="gpu-top"><div><div class="gpu-name">GPU ${g.index}</div><div class="muted small-t">${esc(n.name || "")}</div></div>
        ${idle ? `<span class="pill idle">Not used</span>` : `<span class="pill ok">In use</span>`}</div>
      <div class="gpu-mid"><div class="ring-wrap">${ring(g.util, "var(--s1)")}<div class="ring-n">${fmt(g.util)}<small>%</small></div><div class="ring-l">busy</div></div>
        <div class="gpu-stats">
          <div class="stat"><div class="stat-h"><span>Memory</span><b>${fmt(g.vram_used, 1)} <span class="muted">/ ${fmt(g.vram_total, 0)} GB</span></b></div><div class="meter mem"><i style="width:${mem}%"></i></div></div>
          <div class="stat"><div class="stat-h"><span>Power</span><b>${fmt(g.power)} <span class="muted">/ ${fmt(g.power_cap)} W</span></b></div><div class="meter pow"><i style="width:${pw}%"></i></div></div>
          <div class="stat-row"><span><span class="muted">Temp</span> <b class="${hot ? "temp-hot" : ""}">${hot ? "! " : ""}${fmt(t)}°C</b></span><span><span class="muted">Mem</span> <b>${fmt(g.temp_mem)}°C</b></span><span><span class="muted">Clock</span> <b>${fmt(g.sclk)}</b></span></div>
        </div></div>
      <div class="gpu-spark">${spark(utilHist, "var(--s1)", 28)}</div></div>`;
  }).join("") : `<p class="empty">No GPU readings yet.</p>`);

  const h = lat.host || {};
  const ramPct = h.ram_total ? (100 * h.ram_used) / h.ram_total : 0;
  patch($("#d-host"), `<div class="panel-head"><h2>This machine</h2><span class="muted">${esc(S.label)}</span></div>
    <div class="stat"><div class="stat-h"><span>CPU</span><b>${fmt(h.cpu)}%</b></div><div class="meter"><i style="width:${h.cpu || 0}%"></i></div></div>
    <div class="stat"><div class="stat-h"><span>RAM${ch.ple ? ` (holds the PLE table ${info("ple")})` : ""}</span><b>${fmt(h.ram_used)} <span class="muted">/ ${fmt(h.ram_total)} GB</span></b></div><div class="meter mem"><i style="width:${ramPct}%"></i></div></div>
    <div class="stat-row big"><span><span class="muted">Requests served</span> <b>${fmt(srv.requests_done)}</b></span><span><span class="muted">Avg request</span> <b>${srv.e2e == null ? "–" : fmt(srv.e2e, 1) + " s"}</b></span></div>`);
  const base = publicUrl().replace(/\/$/, "");
  const snip = ui.snip || "curl";
  const code = snip === "curl"
    ? `curl ${base}/chat/completions \\\n  -H 'Content-Type: application/json' \\\n  -d '{"model":"${model}","messages":[{"role":"user","content":"Hello!"}]}'`
    : `from openai import OpenAI\nclient = OpenAI(base_url="${base}", api_key="not-used")\nr = client.chat.completions.create(\n    model="${model}",\n    messages=[{"role": "user", "content": "Hello!"}])\nprint(r.choices[0].message.content)`;
  patch($("#d-connect"), `<div class="panel-head"><h2>Connect an app</h2><div class="seg">${["curl", "python"].map((k) => `<button class="${snip === k ? "on" : ""}" data-act="snip" data-arg="${k}">${k === "curl" ? "curl" : "Python"}</button>`).join("")}</div></div>
    <p class="muted" style="margin:0 0 8px">Any OpenAI-compatible app works: give it this base URL and the model name <b>${esc(model)}</b>. No API key needed.</p>${cmdBlock(code)}`);
}
actions.snip = async (k) => { ui.snip = k; renderDash(); };
actions.shareNet = async () => {
  await api.post("/api/prefs", { lan: true });
  await refresh();
  if (S.server.state === "up" || S.server.state === "starting") {
    if (confirm("Restart the model server now so other machines can reach it?")) await api.post("/api/server/restart");
  }
  toast("The model server is shared on your network"); await refresh();
};
actions.p2p = async (mode) => {
  await api.post("/api/prefs", { p2p: mode });
  await refresh();
  toast(mode === "auto" ? "P2P: automatic" : "P2P " + mode + " for vLLM");
  if (S.server.state === "up" && confirm("Restart the model server now to apply the P2P setting?")) await api.post("/api/server/restart");
  renderTopo();
};
actions.dlLimit = async () => {
  const v = Number($("#dl-limit").value || 0);
  await api.post("/api/download/limit", { mbps: v });
  toast(v ? `Download limit ${v} MB/s` : "No download limit"); await refresh();
};

/* ======================================================================= CHAT */
function renderChat() {
  const el = $("#view-chat");
  if (!el.dataset.built) {
    el.innerHTML = `<h1 class="punch sm${slamOnce("chat")}">Say <span class="key">hello.</span></h1>
      <p class="lede">Talk to the model running on this box. Each reply shows how fast it started and how fast it wrote.</p>
      <div class="chat" id="chat-msgs"></div>
      <div class="composer"><textarea id="chat-in" class="in" placeholder="Ask anything. Enter sends, Shift+Enter adds a line."></textarea>
        <div style="display:flex;flex-direction:column;gap:8px"><button class="btn" data-act="chatSend">Send</button>
        <label class="ghost small" style="text-align:center;cursor:pointer">Image<input id="chat-img" type="file" accept="image/*" hidden></label></div></div>
      <div class="row"><label class="switch"><input id="chat-think" type="checkbox"> Let the model think first</label><span id="chat-img-name" class="muted"></span><button class="ghost small" data-act="chatClear">Clear</button></div>`;
    el.dataset.built = "1";
    $("#chat-in").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); } });
    $("#chat-img").addEventListener("change", (e) => {
      const f = e.target.files[0]; if (!f) return;
      const r = new FileReader(); r.onload = () => { ui.chatImg = r.result; $("#chat-img-name").textContent = "Image attached: " + f.name; }; r.readAsDataURL(f);
    });
  }
  patch($("#chat-msgs"), ui.chat.length ? ui.chat.map((m) => `<div class="msg ${m.role}"><div class="who">${m.role === "user" ? "You" : "Model"}</div>
    ${m.img ? `<img src="${m.img}" alt="" style="max-width:220px;border-radius:8px;display:block;margin-bottom:6px">` : ""}<div class="body">${esc(m.content)}${m.streaming ? "▍" : ""}</div>
    ${m.meta ? `<div class="meta">${m.meta}</div>` : ""}</div>`).join("") : `<p class="empty">${S && S.server.state === "up" ? "Ready when you are." : "Start the server first (Dashboard → Start)."}</p>`);
}
async function sendChat() {
  const box = $("#chat-in"); const text = box.value.trim();
  if (!text || ui.chatBusy) return;
  box.value = "";
  const content = ui.chatImg ? [{ type: "text", text }, { type: "image_url", image_url: { url: ui.chatImg } }] : text;
  ui.chat.push({ role: "user", content: text, img: ui.chatImg, api: content });
  ui.chatImg = null; $("#chat-img-name").textContent = "";
  const reply = { role: "assistant", content: "", streaming: true };
  ui.chat.push(reply); ui.chatBusy = true; renderChat();
  const t0 = performance.now(); let first = null, n = 0, usage = null;
  try {
    const r = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json", "X-Rdna-Dash": "1" },
      body: JSON.stringify({ messages: ui.chat.filter((m) => m !== reply).map((m) => ({ role: m.role, content: m.api || m.content })), thinking: $("#chat-think").checked, max_tokens: 1024 }) });
    if (!r.ok || !r.body) throw new Error((await r.text()).slice(0, 300));
    const rd = r.body.getReader(); const dec = new TextDecoder(); let buf = "";
    while (true) {
      const { value, done } = await rd.read(); if (done) break;
      buf += dec.decode(value, { stream: true });
      let i; while ((i = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
        if (!line.startsWith("data:")) continue;
        const d = line.slice(5).trim(); if (d === "[DONE]") continue;
        try {
          const o = JSON.parse(d); if (o.usage) usage = o.usage;
          for (const c of o.choices || []) { const p = (c.delta && (c.delta.content || c.delta.reasoning_content || c.delta.reasoning)) || ""; if (p) { if (first == null) first = performance.now(); n++; reply.content += p; } }
        } catch { /* partial line */ }
      }
      renderChat();
    }
    const tt = first ? (first - t0) / 1000 : null; const dur = first ? (performance.now() - first) / 1000 : 0;
    const toks = (usage && usage.completion_tokens) || n;
    reply.meta = `first token ${fmt(tt, 2)} s · ${fmt(dur > 0 ? (toks - 1) / dur : null, 1)} tok/s · ${toks} tokens`;
  } catch (e) { reply.content += "\n[" + (e.message || e) + "]"; }
  reply.streaming = false; ui.chatBusy = false; renderChat();
}

/* ======================================================================= BENCHMARK */
let B = null;
async function loadBench() { try { B = await api.get("/api/bench"); } catch { return; } renderBench(); }
function fingerprint(r) {
  const f = r.fingerprint || {};
  return [`${r.suite} suite v${r.suite_version}`, f.box, f.recipe, f.pp > 1 ? "PP=" + f.pp : f.tp ? "TP=" + f.tp : null,
    f.p2p == null ? null : "P2P " + (f.p2p ? "on" : "off"), f.ple ? "PLE " + f.ple : null, f.rocm ? "ROCm " + f.rocm : null,
    f.attached ? "external server" : f.path, new Date(r.started * 1000).toLocaleString(), Math.round(r.duration) + " s"].filter(Boolean).join(" · ");
}
function caseName(c) { return c.kind === "vision" ? `Vision ${c.image}px ×${c.concurrency}` : `${c.context / 1024}k ×${c.concurrency || 1}`; }
function renderBench() {
  const el = $("#view-bench"); if (!B) { el.innerHTML = `<p class="empty">Loading…</p>`; return; }
  if (!el.dataset.built) {
    el.innerHTML = `<h1 class="punch sm${slamOnce("bench")}">Benchmark, <span class="key">for real.</span></h1>
      <p class="lede">The same test for every model: prompts sized to exact token counts with the model's own tokenizer, no help from caching, a fixed answer length, and a hidden ${term("passkey", "recall")} to prove long context works. Runs are saved so you can compare models, recipes and boxes.</p>
      <div id="b-run"></div><div id="b-live"></div><div id="b-compare"></div><div id="b-detail"></div><div id="b-history"></div>`;
    el.dataset.built = "1";
    $("#b-run").innerHTML = `<div class="sec"><h2>Run a suite</h2><div class="form">${Object.entries(B.suites).map(([k, s], i) => `<label class="field" style="cursor:pointer"><span class="switch"><input type="radio" name="suite" value="${k}" ${i === 0 ? "checked" : ""}> <b>${esc(s.label)}</b></span>
      <span class="hint">Context ${s.contexts.map((c) => c / 1024 + "k").join(", ")} · ${s.concurrency.join("/")} users at once · vision ${s.vision.map((v) => v + "px").join(", ") || "off"} · ${s.output}-token answers</span></label>`).join("")}</div>
      <div class="row" style="margin-top:16px"><button class="btn go" data-act="bench">Start benchmark</button><label class="switch"><input id="bench-think" type="checkbox"> Leave thinking mode on</label></div></div>`;
  }
  const L = B.live;
  patch($("#b-live"), L && (L.status === "running" || ui.benchWatch) ? `<div class="sec"><h2>${L.status === "running" ? "Running" : "Last run"} · ${esc(L.suite)}${L.status === "running" ? `<span class="right"><button class="ghost small" data-act="benchCancel">Cancel</button></span>` : ""}</h2>
    <div class="big-progress"><i style="width:${L.progress.total ? (100 * L.progress.done) / L.progress.total : 0}%"></i></div>
    <p class="muted">${esc(L.progress.current || L.status)} · ${L.progress.done} of ${L.progress.total} requests</p><div class="log" data-k="bench">${esc(L.log.join("\n"))}</div></div>` : "");
  if (L && L.status === "running") ui.benchWatch = true;

  const runs = B.results;
  if (!ui.compare.size && runs.length) runs.slice(0, Math.min(2, runs.length)).forEach((r) => ui.compare.add(r.id));
  const sel = runs.filter((r) => ui.compare.has(r.id)).slice(0, 4);
  const colors = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)"];
  const runName = (r) => `${r.model} · ${(r.fingerprint && r.fingerprint.box) || ""} · ${new Date(r.started * 1000).toLocaleDateString()}`;
  let cmp = "";
  if (sel.length) {
    const ctxs = [...new Set(sel.flatMap((r) => r.cases.filter((c) => c.kind === "text" && c.summary).map((c) => c.context)))].sort((a, b) => a - b);
    const concs = [...new Set(sel.flatMap((r) => r.cases.filter((c) => c.kind === "text" && c.summary).map((c) => c.concurrency)))].sort((a, b) => a - b);
    const at = (r, ctx, conc, key) => { const c = r.cases.find((c) => c.kind === "text" && c.context === ctx && c.concurrency === conc && c.summary); return c ? c.summary[key] : null; };
    const minC = concs[0], midCtx = ctxs.includes(8192) ? 8192 : ctxs[0];
    const ser = (fn) => sel.map((r, i) => ({ name: runName(r), color: colors[i], pts: fn(r) }));
    cmp = `<div class="sec"><h2>Compare runs</h2><div class="cols">
      <div><p class="chart-title">${term("Prefill", "prefill")} speed by prompt length</p><p class="chart-sub">Prompt tokens per second, one user.</p>${lineChart("b1", { label: "prefill", cats: ctxs.map((c) => c / 1024 + "k"), series: ser((r) => ctxs.map((c, i) => [i, at(r, c, minC, "prefill_tps")])) })}</div>
      <div><p class="chart-title">${term("Time to first token", "ttft")} by prompt length</p><p class="chart-sub">Seconds, median, one user. Lower is better.</p>${lineChart("b2", { label: "ttft", cats: ctxs.map((c) => c / 1024 + "k"), yFmt: (v) => fmt(v, 1) + "s", series: ser((r) => ctxs.map((c, i) => [i, at(r, c, minC, "ttft_p50")])) })}</div>
      <div><p class="chart-title">${term("Decode", "decode")} speed per user</p><p class="chart-sub">Tokens per second each user sees, ${midCtx / 1024}k prompt.</p>${lineChart("b3", { label: "decode", cats: concs.map((c) => c + (c > 1 ? " users" : " user")), series: ser((r) => concs.map((c, i) => [i, at(r, midCtx, c, "decode_tps")])) })}</div>
      <div><p class="chart-title">Total output with many users</p><p class="chart-sub">All users together, ${midCtx / 1024}k prompt.</p>${lineChart("b4", { label: "aggregate", cats: concs.map((c) => c + (c > 1 ? " users" : " user")), series: ser((r) => concs.map((c, i) => [i, at(r, midCtx, c, "output_tps")])) })}</div>
    </div></div>`;
  }
  patch($("#b-compare"), cmp);
  const det = runs.find((r) => r.id === ui.benchSel) || runs[0];
  patch($("#b-detail"), det ? `<div class="sec"><h2>Run details · ${esc(det.model)}</h2><p class="muted">${esc(fingerprint(det))}</p><div class="scroll-x"><table class="t">
    <thead><tr><th>Case</th><th class="num">Prompt tok</th><th class="num">${term("TTFT", "ttft")} p50</th><th class="num">p95</th><th class="num">${term("Prefill", "prefill")} tok/s</th><th class="num">${term("Decode", "decode")} tok/s</th><th class="num">ITL p95</th><th class="num">All users tok/s</th><th class="num">${term("Recall", "recall")}</th><th class="num">Errors</th></tr></thead><tbody>
    ${det.cases.map((c) => { const s = c.summary || {}; return c.skip ? `<tr><td>${caseName(c)}</td><td colspan="9" class="muted">Skipped: ${esc(c.skip)}</td></tr>` :
      `<tr><td>${caseName(c)}</td><td class="num">${fmt(s.prompt_tokens)}</td><td class="num">${fmt(s.ttft_p50, 2)} s</td><td class="num">${fmt(s.ttft_p95, 2)} s</td><td class="num">${fmt(s.prefill_tps)}</td><td class="num">${fmt(s.decode_tps, 1)}</td><td class="num">${fmt(s.itl_p95_ms)} ms</td><td class="num">${fmt(s.output_tps, 1)}</td><td class="num">${s.recall == null ? "–" : fmt(s.recall * 100) + "%"}</td><td class="num">${s.errors || 0}</td></tr>`; }).join("")}
    </tbody></table></div></div>` : "");
  patch($("#b-history"), `<div class="sec"><h2>Saved runs</h2>${runs.length ? `<table class="t"><thead><tr><th>Compare</th><th>When</th><th>Model</th><th>Setup</th><th>Suite</th><th></th></tr></thead><tbody>
    ${runs.map((r) => `<tr><td><input type="checkbox" data-cmp="${r.id}" ${ui.compare.has(r.id) ? "checked" : ""} aria-label="Compare this run"></td><td>${new Date(r.started * 1000).toLocaleString()}</td><td>${esc(r.model)}</td>
      <td class="muted">${esc([r.fingerprint.box, r.fingerprint.recipe, r.fingerprint.tp > 1 ? "TP=" + r.fingerprint.tp : r.fingerprint.pp > 1 ? "PP=" + r.fingerprint.pp : null].filter(Boolean).join(" · "))}</td><td>${esc(r.suite)} ${r.status !== "done" ? pill("warn", r.status) : ""}</td>
      <td><button class="link" data-act="benchPick" data-arg="${r.id}">Details</button> · <button class="link" data-act="benchDel" data-arg="${r.id}">Delete</button></td></tr>`).join("")}</tbody></table><p class="muted">Pick up to four runs to compare. Runs only compare cleanly within the same suite version.</p>` : `<p class="empty">No runs yet. Start the Quick suite to get a baseline.</p>`}</div>`);
}
document.addEventListener("change", (e) => {
  const id = e.target.dataset && e.target.dataset.cmp; if (!id) return;
  if (e.target.checked) { if (ui.compare.size >= 4) { e.target.checked = false; toast("Compare up to four runs"); return; } ui.compare.add(id); } else ui.compare.delete(id);
  renderBench();
});

/* ======================================================================= TOPOLOGY */
let TP = null;
async function loadTopo(refresh) { try { TP = await api.get("/api/topology" + (refresh ? "?refresh=1" : "")); } catch (e) { TP = { error: e.message }; } renderTopo(); }
function seq(v, max) { const steps = ["var(--seq-1)", "var(--seq-2)", "var(--seq-3)", "var(--seq-4)", "var(--seq-5)"]; if (v == null) return "var(--seq-0)"; return steps[Math.min(4, Math.floor((v / (max || 1)) * 4.999))]; }
function renderTopo() {
  const el = $("#view-topo"); if (!TP) { el.innerHTML = `<p class="empty">Scanning…</p>`; return; }
  if (TP.error) { el.innerHTML = `<p class="empty">${esc(TP.error)}</p>`; return; }
  if (!el.dataset.built) {
    el.innerHTML = `<h1 class="punch sm${slamOnce("topo")}">Who talks <span class="key">to who.</span></h1>
      <p class="lede">How each GPU connects to the CPU and to the other GPUs: the PCIe switches in between, the link speed, and whether the driver allows direct ${term("P2P", "p2p")} copies. Then measure it.</p>
      <div class="row"><button class="ghost" data-act="topoScan">Rescan</button><button class="btn" data-act="bandwidth" data-arg="torch">Measure GPU-to-GPU bandwidth</button><span id="t-rbt"></span></div>
      <div id="t-p2p"></div><div id="t-job"></div><div id="t-map"></div><div id="t-pairs"></div><div id="t-bw"></div><div id="t-raw"></div>`;
    el.dataset.built = "1";
  }
  patch($("#t-rbt"), TP.tools && TP.tools.rocm_bandwidth_test ? `<button class="ghost" data-act="bandwidth" data-arg="rbt">Run rocm-bandwidth-test</button>` : `<span class="muted">rocm-bandwidth-test not installed (optional)</span>`);
  const job = ui.topoJob ? ui.logs[ui.topoJob] : null;
  patch($("#t-job"), job != null ? `<div class="sec"><h2>Measuring</h2><div class="log" data-k="topo">${esc(job || "Starting…")}</div></div>` : "");
  patch($("#t-p2p"), S ? p2pPanel(TP) : "");
  patch($("#t-map"), `<div class="sec"><h2>PCIe map</h2>${topoSvg(TP)}</div>`);
  const viaTxt = { switch: "Shared PCIe switch", "cpu-root-complex": "Through one CPU root complex",
    "cpu-fabric": "Same CPU, different root complex", "cpu-interconnect": "Across CPU sockets" };
  patch($("#t-pairs"), TP.pairs.length ? `<div class="sec"><h2>Every pair</h2><div class="scroll-x"><table class="t"><thead><tr><th>Pair</th><th>Path</th><th class="num">${term("Hops", "hops")}</th><th>Driver P2P link</th><th class="num">Measured</th></tr></thead><tbody>
    ${TP.pairs.map((p) => { const bw = TP.bandwidth && TP.bandwidth.gbs ? TP.bandwidth.gbs[p.a] && TP.bandwidth.gbs[p.a][p.b] : null;
      return `<tr><td><b>GPU ${p.a} ↔ GPU ${p.b}</b></td><td>${esc(viaTxt[p.via] || p.via)}${p.meet ? ` <span class="muted mono">${esc(p.meet)}</span>` : ""}</td><td class="num">${p.hops}</td>
      <td>${p.kfd_p2p == null ? `<span class="muted">unknown</span>` : p.kfd_p2p ? pill("ok", "Enabled") : pill("warn", "Not enabled")}</td><td class="num">${bw == null ? "–" : fmt(bw, 1) + " GB/s"}</td></tr>`; }).join("")}</tbody></table></div>
    <p class="muted">Pairs behind the same switch can usually copy directly. Pairs on different root complexes of the same CPU go through its I/O die. ${(TP.sockets || {}).count > 1 ? "Crossing CPU sockets is the slowest path; keep tensor-parallel groups on one socket when you can." : ""}</p></div>` : "");
  const bw = TP.bandwidth;
  if (bw && bw.gbs) {
    const n = bw.devices; const max = Math.max(...bw.gbs.flat().filter((x) => x != null), ...(bw.h2d || []));
    patch($("#t-bw"), `<div class="sec"><h2>Bandwidth, GB/s</h2><div class="scroll-x"><table class="t heat"><thead><tr><th>From \\ To</th>${Array.from({ length: n }, (_, j) => `<th class="num">GPU ${j}</th>`).join("")}<th class="num">From host</th></tr></thead><tbody>
      ${bw.gbs.map((row, i) => `<tr><td><b>GPU ${i}</b></td>${row.map((v) => `<td class="cell" style="background:${seq(v, max)};color:${v != null && v / max > 0.55 ? "#0b0b0b" : "var(--text)"}">${v == null ? "·" : fmt(v, 1)}</td>`).join("")}<td class="cell" style="background:${seq(bw.h2d[i], max)};color:${bw.h2d[i] / max > 0.55 ? "#0b0b0b" : "var(--text)"}">${fmt(bw.h2d[i], 1)}</td></tr>`).join("")}</tbody></table></div>
      <p class="muted">The more a cell stands out from the background, the faster that path. ${bw.peer.flat().every(Boolean) ? "Every pair can copy directly, so the plan keeps P2P on." : "Some pairs cannot copy directly, so the plan turns P2P off."}</p></div>`);
  } else patch($("#t-bw"), "");
  patch($("#t-raw"), (TP.smi ? `<details class="sec"><summary>amd-smi / rocm-smi topology (raw)</summary><div class="log tall">${esc(TP.smi)}</div></details>` : "") +
    (TP.rbt_output ? `<details class="sec"><summary>rocm-bandwidth-test output (raw)</summary><div class="log tall">${esc(TP.rbt_output)}</div></details>` : ""));
}
function p2pPanel(tp) {
  const ch = S.plan.choice, prof = S.profile, mode = (S.prefs && S.prefs.p2p) || "auto";
  if ((tp.gpus || []).length < 2) return "";
  const pairs = tp.pairs || [];
  const linked = pairs.filter((p) => p.kfd_p2p).length;
  const vram = Object.fromEntries((prof.gpus || []).map((g) => [g.index, g.vram_gb || 0]));
  const smallBar = (tp.gpus || []).filter((g) => g.large_bar && g.large_bar < (vram[g.index] || 0) * 0.9);
  const amd = prof.cpu.vendor === "amd", vm = !["none", "unknown", ""].includes(prof.os.virt);
  const bw = tp.bandwidth;
  const items = [
    [linked === pairs.length && pairs.length > 0 ? "ok" : linked ? "warn" : "fail", "Driver P2P links",
      linked === pairs.length ? `All ${pairs.length} GPU pairs have a direct link.` : linked ? `${linked} of ${pairs.length} pairs have a direct link.`
        : tp.kfd_p2p_supported ? "The driver created no direct links between any pair." : "This kernel does not report P2P links (no p2p_links in the KFD topology)."],
    [smallBar.length ? "fail" : "ok", "Large BAR (Resizable BAR / Above 4G)",
      smallBar.length ? `GPU ${smallBar.map((g) => g.index).join(", ")} expose only ${fmt(smallBar[0].large_bar, 0)} GB of VRAM. Turn on Above 4G decoding and Resizable BAR in the BIOS.` : "Every card exposes its full VRAM."],
    [prof.os.iommu_pt ? "ok" : amd ? "warn" : "info", "iommu=pt on the kernel command line",
      prof.os.iommu_pt ? "Set." : "Not set. The validated setups boot with iommu=pt (and ACS off)."],
    [amd ? "ok" : "warn", "CPU platform", amd ? "AMD: the wiki-validated P2P platform." : "Intel: P2P can enumerate but often does not speed up inference. A/B it."],
    [vm ? "fail" : "ok", "Bare metal", vm ? `Running in a VM (${esc(prof.os.virt)}): passthrough usually blocks peer access.` : "Not a VM."],
    [bw ? (bw.peer.flat().every(Boolean) ? "ok" : "warn") : "info", "Measured",
      bw ? (bw.peer.flat().every(Boolean) ? "PyTorch can copy directly between every pair." : "PyTorch cannot copy directly between some pairs.") : "Not measured yet. Use the button above."],
  ];
  const label = { ok: "OK", warn: "Check", fail: "Missing", info: "Tip" };
  const now = ch.p2p == null ? "not used (one GPU)" : ch.p2p ? "on" : "off (P2P=0, traffic goes through the CPU)";
  return `<div class="sec"><h2>P2P for vLLM</h2>
    <div class="row" style="margin-bottom:12px"><div class="seg">${[["auto", "Auto"], ["on", "On"], ["off", "Off"]].map(([k, l]) => `<button class="${mode === k ? "on" : ""}" data-act="p2p" data-arg="${k}">${l}</button>`).join("")}</div>
      <span>vLLM uses P2P: <b>${esc(now)}</b></span><span class="muted">${mode === "auto" ? "Auto follows the measurement, or a safe guess before one." : "Set by you."}</span></div>
    <p class="muted" style="margin:0 0 6px">The switch only tells vLLM whether to use direct GPU-to-GPU copies. Whether the cards <i>can</i> depends on the driver, kernel and BIOS, below. Forcing it on without driver links can hang multi-GPU serving.</p>
    ${items.map(([lvl, t, d]) => `<div class="check"><span>${pill(lvl, label[lvl])}</span><div class="t">${esc(t)}</div><div class="d">${d}</div></div>`).join("")}
    ${linked < pairs.length ? `<p style="margin-top:12px">To turn on driver P2P, follow the <a href="https://blivioniag.github.io/gfx1030-wiki/tuning/p2p.html" target="_blank" rel="noopener">wiki's P2P page</a> and the <a href="https://github.com/blivioniag/v620_toolbox/tree/master/pcie_p2p" target="_blank" rel="noopener">v620_toolbox pcie_p2p</a> scripts (kernel options, iommu=pt, ACS off, large BAR; validated on Fedora with an AMD CPU), reboot, then Rescan and measure here.</p>` : ""}</div>`;
}
function topoSvg(tp) {
  const gpus = tp.gpus; if (!gpus.length) return `<p class="empty">No GPUs to map.</p>`;
  const W = Math.max(640, gpus.length * 150), H = 400, colW = W / gpus.length;
  const gx = (i) => colW * i + colW / 2;
  const span = (idx) => [Math.min(...idx.map(gx)) - colW / 2 + 8, Math.max(...idx.map(gx)) + colW / 2 - 8];
  const nSock = (tp.sockets && tp.sockets.count) || 1;
  const sockOf = (g) => (g.socket == null ? 0 : g.socket);
  let s = "";
  // Sockets: one box per physical CPU, spanning its GPUs.
  const socks = [...new Set(gpus.map(sockOf))];
  socks.forEach((k) => {
    const idx = gpus.map((g, i) => (sockOf(g) === k ? i : -1)).filter((i) => i >= 0);
    const [x0, x1] = span(idx);
    const name = (tp.sockets && tp.sockets.model) || "CPU";
    s += `<rect x="${x0}" y="10" width="${x1 - x0}" height="132" rx="12" fill="var(--bg-3)" fill-opacity=".55" stroke="var(--line-2)"/>
      <text x="${x0 + 14}" y="32">${nSock > 1 ? "CPU socket " + k : "CPU"}</text><text class="sub" x="${x0 + 14}" y="48">${esc(name)}</text>`;
  });
  // Root complexes inside the socket.
  const roots = [...new Set(gpus.map((g) => g.root))];
  const rootX = {};
  roots.forEach((r) => {
    const idx = gpus.map((g, i) => (g.root === r ? i : -1)).filter((i) => i >= 0);
    const x = (gx(idx[0]) + gx(idx[idx.length - 1])) / 2;
    rootX[r] = x;
    s += `<rect x="${x - 66}" y="74" width="132" height="46" rx="8" fill="var(--bg-2)" stroke="var(--line-2)"/><text x="${x}" y="94" text-anchor="middle">Root complex</text><text class="sub" x="${x}" y="110" text-anchor="middle">${esc(r)}</text>`;
  });
  const sw = {};
  (tp.switches || []).forEach((x) => { const xs = x.gpus.map((i) => gx(gpus.findIndex((g) => g.index === i))); sw[x.addr] = (Math.min(...xs) + Math.max(...xs)) / 2; });
  const drawn = new Set();
  const gy = 290;
  gpus.forEach((g, i) => {
    const x = gx(i); const firstSw = g.bridges.find((b) => b.kind === "switch");
    if (firstSw && sw[firstSw.addr] != null) {
      const sx = sw[firstSw.addr];
      if (!drawn.has(firstSw.addr)) {
        drawn.add(firstSw.addr);
        s += `<line x1="${rootX[g.root]}" y1="120" x2="${sx}" y2="176" stroke="var(--line-2)" stroke-width="2"/><rect x="${sx - 70}" y="176" width="140" height="38" rx="8" fill="var(--bg-3)"/><text x="${sx}" y="193" text-anchor="middle">PCIe switch</text><text class="sub" x="${sx}" y="207" text-anchor="middle">${esc((tp.switches.find((q) => q.addr === firstSw.addr) || {}).vendor || "")}</text>`;
      }
      s += `<line x1="${sx}" y1="214" x2="${x}" y2="${gy}" stroke="var(--s1)" stroke-width="2"/>`;
    } else {
      s += `<line x1="${rootX[g.root]}" y1="120" x2="${x}" y2="${gy}" stroke="var(--s1)" stroke-width="2"/>`;
    }
    const l = g.link || {};
    const slow = l.max_gen && l.gen && (l.gen < l.max_gen || l.width < l.max_width);
    s += `<text class="sub" x="${x + 8}" y="${gy - 14}">${l.gen ? "Gen" + l.gen : ""}${l.width ? " x" + l.width : ""}${slow ? " (below max)" : ""}</text>`;
    s += `<rect x="${x - 62}" y="${gy}" width="124" height="62" rx="10" fill="var(--bg-2)" stroke="var(--line-2)"/><text x="${x}" y="${gy + 24}" text-anchor="middle">GPU ${g.index}</text><text class="sub" x="${x}" y="${gy + 40}" text-anchor="middle">${esc(g.pci)}</text><text class="sub" x="${x}" y="${gy + 54}" text-anchor="middle">BAR ${fmt(g.large_bar, g.large_bar < 1 ? 2 : 0)} GB</text>`;
  });
  return `<div class="topo scroll-x"><svg viewBox="0 0 ${W} ${H - 30}" style="min-width:${Math.min(W, 900)}px;width:100%" role="img" aria-label="PCIe topology">${s}</svg></div>
    <p class="muted">${nSock > 1 ? nSock + " CPU sockets. " : "One CPU. "}A CPU has several PCIe root complexes; cards on different ones talk through the CPU's I/O die. Blue lines are each card's slot link. A BAR much smaller than the card's VRAM means Resizable BAR / Above 4G is off in the BIOS, which blocks P2P.</p>`;
}

/* ======================================================================= FLEET */
let F = null;
async function loadFleet() { try { F = await api.get("/api/fleet"); } catch (e) { F = []; } renderFleet(); }
function renderFleet() {
  const el = $("#view-fleet");
  if (!el.dataset.built) {
    el.innerHTML = `<h1 class="punch sm${slamOnce("fleet")}">All your <span class="key">boxes.</span></h1>
      <p class="lede">Each GPU box runs its own dashboard. Add the others here to see them side by side. Their names and addresses stay on this machine.</p>
      <div id="f-list"></div>
      <div class="sec"><h2>Add a box</h2><form id="fleet-form" class="form" onsubmit="return false">
        <div class="field"><label>Name</label><input name="name" placeholder="gpu-box-2" required></div>
        <div class="field"><label>Dashboard URL</label><input name="url" placeholder="http://gpu-box-2.local:7860" required></div>
        <div class="field"><label>Access token</label><input name="token" placeholder="from that box's config" autocomplete="off"><div class="hint">Start that box with --host 0.0.0.0; it prints its token.</div></div>
        <div class="field"><label>&nbsp;</label><button class="btn" data-act="fleetAdd">Add</button></div></form></div>
      <div class="sec"><h2>Share this box's setup</h2><p class="muted">Copies a report with no hostname, username or tokens, ready to paste in an issue or on Discord.</p><button class="ghost" data-act="share">Copy redacted report</button></div>`;
    el.dataset.built = "1";
  }
  if (!F) return;
  const word = { up: "Serving", starting: "Starting", crashed: "Error", stopped: "Off" };
  const lvl = { up: "ok", starting: "run", crashed: "fail", stopped: "idle" };
  patch($("#f-list"), F.map((b) => `<div class="box-row"><div><div class="nm">${esc(b.name)}</div><div class="muted">${b.me ? "this box" : esc(b.url)}</div></div>
    <div>${b.reachable ? pill(lvl[b.state] || "idle", word[b.state] || b.state) : pill("fail", "Unreachable")}</div>
    <div><b>${esc(b.recipe || "")}</b><div class="muted">${esc(b.gpus || b.error || "")}</div></div>
    <div><span style="font:italic 900 26px var(--display)">${fmt(b.gen_tps, 1)}</span> <span class="muted">tok/s</span></div>
    <div class="muted">${(b.gpu || []).map((g) => `GPU${g.index} ${fmt(g.temp_junction || g.temp_edge)}°C ${fmt(g.util)}%`).join(" · ")}</div>
    <div>${b.me ? "" : `<button class="link" data-act="fleetRm" data-arg="${esc(b.name)}">Remove</button>`}</div></div>`).join(""));
}

/* ======================================================================= LOGS */
async function renderLogs() {
  const el = $("#view-logs");
  if (!el.dataset.built) {
    el.innerHTML = `<h1 class="punch sm${slamOnce("logs")}">Server <span class="key">log.</span></h1><p class="lede">The last lines from the model server. Known problems are explained at the top.</p><div id="l-prob"></div><div id="l-log"></div>`;
    el.dataset.built = "1";
  }
  try {
    const r = await api.get("/api/server/log?lines=600");
    patch($("#l-prob"), r.problems.length ? r.problems.map(problemRow).join("") : "");
    patch($("#l-log"), `<div class="log tall" data-k="server">${esc(r.text || "No log yet. Start the server first.")}</div>`);
  } catch (e) { patch($("#l-log"), `<p class="empty">${esc(e.message)}</p>`); }
}

/* ---------- routing and loops */
function go() {
  view = (location.hash || "#setup").slice(1);
  if (!$("#view-" + view)) view = "setup";
  $$(".view").forEach((v) => v.classList.toggle("active", v.dataset.view === view));
  if (S) renderRail();
  ({ setup: renderSetup, dash: () => { renderDash(); loadTelemetry(); }, chat: renderChat, bench: loadBench, topo: () => loadTopo(false), fleet: loadFleet, logs: renderLogs }[view] || (() => {}))();
}
window.addEventListener("hashchange", go);
$("#open-dismiss").addEventListener("click", () => { try { localStorage.setItem("rdna-open-ok", "1"); } catch { /* ignore */ } $("#open-banner").hidden = true; });
(function theme() {
  let t = null; try { t = localStorage.getItem("rdna-theme"); } catch { /* private mode */ }
  if (t) document.documentElement.dataset.theme = t;
  $("#theme").addEventListener("click", () => {
    const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
    const next = cur === "light" ? "dark" : "light"; document.documentElement.dataset.theme = next;
    try { localStorage.setItem("rdna-theme", next); } catch { /* ignore */ }
    charts.clear(); $$(".view").forEach((v) => { if (v.dataset.view === "dash") $$("[data-chart]", v).forEach((c) => (c.parentElement._html = null)); });
  });
})();

async function loop() {
  await refresh();
  setInterval(refresh, 2000);
  setInterval(() => { if (view === "dash") loadTelemetry(); }, 1000);
  setInterval(() => { if (view === "fleet") loadFleet(); }, 4000);
  setInterval(async () => {
    if (view === "bench" && B && B.live && B.live.status === "running") await loadBench();
    if (view === "topo" && ui.topoJob) {
      const id = ui.topoJob;
      try { const r = await api.get(`/api/jobs/${id}?offset=${ui.offsets[id] || 0}`); ui.logs[id] += r.text; ui.offsets[id] = r.offset; if (r.status !== "running") { ui.topoJob = null; await loadTopo(false); toast(r.status === "ok" ? "Measured" : "Measurement " + r.status); } else renderTopo(); } catch { ui.topoJob = null; }
    }
  }, 900);
  go();
}
loop();
