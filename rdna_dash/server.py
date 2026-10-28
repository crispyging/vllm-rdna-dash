"""HTTP server: the setup wizard, live dashboard, benchmark, topology and fleet API.

Stdlib only. Binds 127.0.0.1 by default; on any other address every request
needs the access token (cookie set by the one-time ?token= link, or a Bearer
header for fleet peers). POSTs also need an X-Rdna-Dash header, which a
cross-site form cannot send.
"""

import hashlib
import hmac
import http.client
import ipaddress
import json
import mimetypes
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__, bench, logdoctor, profile as profiler, store, topology, tuning
from .jobs import DemoRunner, Runner
from .serve import ServeManager
from .telemetry import GpuReader, HostReader, Sampler

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

LOGIN = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RDNA Dash sign in</title><style>
:root{color-scheme:dark;--bg:#0e0e10;--bg2:#141417;--line:#2a2a30;--text:#f2f1ee;--muted:#8b8a84;--key:#ffd21f;--brand:#ff3b30}
@media (prefers-color-scheme:light){:root{color-scheme:light;--bg:#fafaf8;--bg2:#f1f0ec;--line:#dddcd6;--text:#0b0b0b;--muted:#6b6a65;--key:#b88a00}}
body{margin:0;min-height:100vh;display:grid;place-items:center;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif;padding:16px}
form{width:min(420px,100%%);background:var(--bg2);border:1px solid var(--line);border-radius:18px;padding:28px}
h1{font:italic 900 34px/1 "Montserrat","Arial Black",sans-serif;text-transform:uppercase;margin:0 0 10px;text-shadow:3px 3px 0 var(--brand)}
h1 b{color:var(--key)}p{color:var(--muted);margin:0 0 18px}code{font-size:13px}
input{width:100%%;box-sizing:border-box;padding:11px 12px;border-radius:10px;border:1px solid var(--line);background:var(--bg);color:var(--text);font:14px ui-monospace,monospace}
button{margin-top:14px;width:100%%;padding:13px;border:0;border-radius:10px;background:var(--key);color:#0b0b0b;font:italic 900 16px "Montserrat","Arial Black",sans-serif;text-transform:uppercase;cursor:pointer}
.err{color:#ec835a;margin:10px 0 0}</style></head><body>
<form method="post" action="/login"><h1>RDNA<b>DASH</b></h1>
<p>Paste the access token. It is printed when the dashboard starts (<code>docker logs vllm-rdna-dash</code>), or set your own with <code>RDNA_DASH_TOKEN</code>.</p>
<input name="token" type="password" autocomplete="current-password" placeholder="access token" autofocus required>
%s<button type="submit">Sign in</button></form></body></html>"""


def _hash(script):
    return hashlib.sha1((script or "").encode()).hexdigest()[:12]


class App(object):
    def __init__(self, demo=False, bind="127.0.0.1"):
        self.demo = demo
        self.bind = bind
        self.cfg = store.load_config()
        self.lock = threading.RLock()
        self.autopilot = {"status": "idle", "step": None, "message": None}
        self.smoke = None
        self.bench = None
        self.topo = None
        if demo:
            from . import demo as d
            self.fake = d.FakeVllm()
            self.runner = DemoRunner(on_done=self._job_done)
            self.serve = d.DemoServe(self.fake)
            self.profile = d.demo_profile()
            gpu_reader, host_reader = d.DemoGpuReader(self.fake), d.DemoHostReader()
        else:
            self.runner = Runner(on_done=self._job_done)
            self.serve = ServeManager()
            self.profile = profiler.collect(self.cfg["prefs"].get("root"))
            gpu_reader, host_reader = GpuReader(self.profile["gpus"]), HostReader()
        self.rebuild()
        self.sampler = Sampler(gpu_reader, host_reader, self.server_url)
        self.sampler.start()

    # ------------------------------------------------------------------ plan
    def rebuild(self):
        st = store.load_state()
        prefs = dict(self.cfg.get("prefs") or {})
        if st.get("p2p") is not None and prefs.get("p2p_probe") is None:
            prefs["p2p_probe"] = st["p2p"]
        self.registry = tuning.load_registry()
        self.plan = tuning.build_plan(self.profile, prefs, self.registry)
        if self.demo:
            # Point the plan at the fake server so the dashboard and bench exercise real code paths.
            self.plan["choice"]["port"] = self.fake.port
        return self.plan

    def server_url(self):
        if self.cfg.get("attach_url"):
            return self.cfg["attach_url"]
        return "http://127.0.0.1:%d/v1" % self.plan["choice"]["port"]

    def label(self):
        return self.cfg.get("label") or self.profile.get("hostname") or "this box"

    # ------------------------------------------------------------------ steps
    def step_status(self):
        st = store.load_state()["steps"]
        out = {}
        for s in self.plan["steps"]:
            rec = st.get(s["id"]) or {}
            if s["kind"] == "check":
                status = "needs-you" if s.get("blocked_by") else "done"
            elif s["kind"] == "tune":
                status = "recommended" if s.get("pending") else "done"
            elif s["kind"] == "serve":
                status = "done" if self.server_state()["state"] == "up" else ("running" if self.server_state()["state"] == "starting" else "todo")
            elif s["kind"] == "smoke":
                status = "done" if self.smoke and self.smoke.get("ok") else ("failed" if self.smoke else "todo")
            elif s["kind"] == "write":
                status = "done" if rec.get("status") == "done" and os.path.exists(self.plan["paths"]["compose"]) or (self.demo and rec.get("status") == "done") else "todo"
            else:
                if rec.get("hash") and rec["hash"] != _hash(s.get("script")):
                    status = "todo"  # plan changed since this ran
                else:
                    status = rec.get("status") or "todo"
                if self.runner.current and self.runner.current.step == s["id"] and self.runner.current.status == "running":
                    status = "running"
            out[s["id"]] = {"status": status, "job": rec.get("job"), "at": rec.get("at")}
        return out

    def run_step(self, sid):
        step = next((s for s in self.plan["steps"] if s["id"] == sid), None)
        if not step:
            raise KeyError(sid)
        if step["kind"] == "run":
            job = self.runner.start(sid, step["script"])
            store.update_state(lambda st: st["steps"].__setitem__(sid, {"status": "running", "job": job.id,
                                                                         "hash": _hash(step["script"]), "at": time.time()}))
            return {"job": job.info()}
        if step["kind"] == "write":
            path = self.serve.write_compose(self.plan["launch"])
            store.update_state(lambda st: st["steps"].__setitem__(sid, {"status": "done", "at": time.time()}))
            return {"written": path}
        if step["kind"] == "serve":
            return self.serve.start(self.plan)
        if step["kind"] == "smoke":
            return self.run_smoke()
        if step["kind"] == "check":
            self.refresh_profile()
            return {"ok": True}
        raise KeyError(sid)

    def _job_done(self, job):
        rec = {"status": "done" if job.status == "ok" else job.status, "job": job.id, "at": time.time()}
        step = next((s for s in self.plan["steps"] if s["id"] == job.step), None)
        rec["hash"] = _hash(step["script"]) if step else None
        text, _ = job.read(0, 4 * 1024 * 1024)

        def upd(st):
            if job.step in ("bandwidth", "rbt"):
                st.setdefault("topology", {})[job.step] = {"status": job.status, "job": job.id, "at": time.time()}
            else:
                st["steps"][job.step] = rec
            m = re.search(r"^P2P_RESULT (.+)$", text, re.M)
            if m:
                st["p2p"] = bool(json.loads(m.group(1)).get("all"))
            m = re.search(r"^BW_RESULT (.+)$", text, re.M)
            if m:
                bw = json.loads(m.group(1))
                st.setdefault("topology", {})["bandwidth_result"] = bw
                st["p2p"] = all(all(r) for r in bw.get("peer", [[False]]))
            if job.step == "rbt":
                st.setdefault("topology", {})["rbt_output"] = text[-60000:]
        if self.demo and job.step == "bandwidth":
            from . import demo as d
            text += "\nBW_RESULT " + json.dumps(d.demo_bandwidth())
        store.update_state(upd)
        if "P2P_RESULT" in text or "BW_RESULT" in text:
            self.rebuild()

    def refresh_profile(self):
        if not self.demo:
            self.profile = profiler.collect(self.cfg["prefs"].get("root"))
        self.rebuild()

    # ------------------------------------------------------------------ autopilot
    def start_autopilot(self):
        with self.lock:
            if self.autopilot["status"] == "running":
                return self.autopilot
            self.autopilot = {"status": "running", "step": None, "message": "Starting", "started": time.time()}
        threading.Thread(target=self._autopilot, daemon=True).start()
        return self.autopilot

    def _autopilot(self):
        ap = self.autopilot
        try:
            self.refresh_profile()
            for s in self.plan["steps"]:
                if ap.get("cancel"):
                    ap.update(status="cancelled", message="Stopped by you.")
                    return
                status = self.step_status()[s["id"]]["status"]
                ap["step"] = s["id"]
                if status in ("done", "skipped") or s["kind"] == "tune":
                    continue  # tuning needs root and reboots: shown, never blocks setup
                # Optional probes (P2P) still run here: their result tunes the plan. A failure is skipped below.
                if s["kind"] == "check":
                    ap.update(status="needs-you", message="Some things need you first. Run the commands shown, then press Re-check.")
                    return
                ap["message"] = s["title"]
                if s["kind"] == "run":
                    job = self.run_step(s["id"])["job"]
                    while self.runner.jobs[job["id"]].status == "running":
                        if ap.get("cancel"):
                            self.runner.cancel(job["id"])
                        time.sleep(0.5)
                    if self.runner.jobs[job["id"]].status != "ok":
                        if s.get("optional"):
                            continue
                        ap.update(status="failed", message="%s failed. Open its log to see why." % s["title"])
                        return
                elif s["kind"] == "write":
                    self.run_step(s["id"])
                elif s["kind"] == "serve":
                    res = self.serve.start(self.plan)
                    if not res.get("ok"):
                        ap.update(status="failed", message="The server did not start: " + (res.get("message") or "")[-300:])
                        return
                    deadline = time.time() + 3 * 3600
                    while time.time() < deadline:
                        ss = self.server_state()
                        if ss["state"] == "up":
                            break
                        if ss["state"] in ("crashed", "stopped"):
                            ap.update(status="failed", message="The server stopped while starting. See the problems found in its log.")
                            return
                        ap["message"] = "Starting the server: %s" % (ss.get("phase") or "booting")
                        if ap.get("cancel"):
                            ap.update(status="cancelled", message="Stopped by you.")
                            return
                        time.sleep(2)
                elif s["kind"] == "smoke":
                    res = self.run_smoke()
                    if not res.get("ok"):
                        ap.update(status="failed", message="The first request failed: %s" % res.get("error"))
                        return
            ap.update(status="done", message="All set. Your model is serving.", step=None)
        except Exception as e:
            ap.update(status="failed", message=str(e))

    # ------------------------------------------------------------------ server
    def server_state(self):
        proc = {"process": "attached"} if self.cfg.get("attach_url") else self.serve.status(self.plan)
        latest = self.sampler.latest.get("server", {}) if hasattr(self, "sampler") else {}
        if latest.get("healthy"):
            state = "up"
        elif proc["process"] == "running" or (proc["process"] == "attached" and latest.get("reachable")):
            state = "starting"
        elif proc["process"] in ("exited", "dead"):
            state = "crashed"
        else:
            state = "stopped"
        out = {"state": state, "process": proc["process"], "started": proc.get("started"), "url": self.server_url(),
               "attached": bool(self.cfg.get("attach_url"))}
        if state in ("starting", "crashed"):
            log = self.serve.log(400) if not self.cfg.get("attach_url") else ""
            out["phase"] = logdoctor.phase(log)
            out["problems"] = logdoctor.diagnose(log)
        return out

    def run_smoke(self):
        c = bench.Client(self.server_url())
        res = {"ok": False, "at": time.time()}
        try:
            models = c.models()
            res["models"] = [m["id"] for m in models]
            c.model = models[0]["id"]
            res["max_model_len"] = models[0].get("max_model_len")
            r = c.stream_chat([{"role": "user", "content": "Say hello in one short sentence."}], 48,
                              {"chat_template_kwargs": {"enable_thinking": False}})
            if not r.get("ok"):
                res["error"] = r.get("error")
            else:
                res.update(ok=True, ttft=r["ttft"], e2e=r["e2e"], text=r["text"][:400],
                           tps=(r["completion_tokens"] - 1) / r["decode_time"] if r["decode_time"] > 0 else None)
        except Exception as e:
            res["error"] = str(e)
        self.smoke = res
        return res

    # ------------------------------------------------------------------ bench
    def start_bench(self, suite, thinking_off=True):
        if self.bench and self.bench.status == "running":
            raise RuntimeError("A benchmark is already running.")
        ch = self.plan["choice"]
        fp = {"box": self.plan["summary"]["gpus"], "recipe": ch["recipe_label"], "path": ch["path"], "tp": ch["tp"],
              "pp": ch["pp"], "p2p": ch["p2p"], "ple": ch["ple"], "rocm": self.plan["summary"]["rocm"],
              "attached": bool(self.cfg.get("attach_url")), "tool": __version__}
        self.bench = bench.Bench(self.server_url(), suite, thinking_off=thinking_off, fingerprint=fp)

        def go():
            res = self.bench.run()
            d = os.path.join(store.STATE_DIR, "bench")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, res["id"] + ".json"), "w") as f:
                json.dump(res, f, indent=1)

        threading.Thread(target=go, daemon=True).start()
        return {"ok": True}

    def bench_results(self):
        d = os.path.join(store.STATE_DIR, "bench")
        out = []
        if os.path.isdir(d):
            for name in sorted(os.listdir(d), reverse=True):
                if name.endswith(".json"):
                    try:
                        with open(os.path.join(d, name)) as f:
                            out.append(json.load(f))
                    except Exception:
                        pass
        return out

    # ------------------------------------------------------------------ topology
    def topology(self, refresh=False):
        if refresh or not self.topo:
            if self.demo:
                from . import demo as d
                self.topo = d.demo_topology()
            else:
                self.topo = topology.discover(self.profile["gpus"])
                self.topo["smi"] = topology.amd_smi_topology()
        st = store.load_state().get("topology", {})
        return dict(self.topo, bandwidth=st.get("bandwidth_result"), rbt_output=st.get("rbt_output"),
                    jobs={k: v for k, v in st.items() if k in ("bandwidth", "rbt")})

    def run_bandwidth(self, tool):
        ch, paths = self.plan["choice"], self.plan["paths"]
        if tool == "rbt":
            exe = (self.topo or self.topology())["tools"].get("rocm_bandwidth_test")
            if not exe:
                raise RuntimeError("rocm-bandwidth-test is not installed. Install it from your ROCm packages "
                                   "(rocm-bandwidth-test), or use the PyTorch test.")
            script = "%s 2>&1" % tuning.q(exe)
        else:
            py = os.path.join(paths["venv"], "bin", "python")
            if not self.demo and os.path.exists(py):
                script = "%s - <<'PY'\n%sPY" % (tuning.q(py), topology.BANDWIDTH_BENCH)
            else:
                eng = self.profile.get("container", {}).get("engine") or "docker"
                groups = "--group-add keep-groups" if self.profile.get("container", {}).get("rootless") else "--group-add video --group-add render"
                script = ("%s run --rm -i --device /dev/kfd --device /dev/dri %s --security-opt seccomp=unconfined "
                          "--security-opt label=disable --ipc host %s python3 - <<'PY'\n%sPY"
                          % (eng, groups, self.registry.get("image", tuning.load_registry().get("image")), topology.BANDWIDTH_BENCH))
        job = self.runner.start("bandwidth" if tool != "rbt" else "rbt", script)
        return {"job": job.info()}

    # ------------------------------------------------------------------ fleet
    def summary(self):
        lat = self.sampler.latest
        ss = self.server_state()
        srv = lat.get("server", {})
        return {"label": self.label(), "version": __version__, "gpus": self.plan["summary"]["gpus"],
                "recipe": self.plan["choice"]["recipe_label"], "state": ss["state"],
                "gen_tps": srv.get("gen_tps"), "prompt_tps": srv.get("prompt_tps"), "running": srv.get("running"),
                "kv_usage": srv.get("kv_usage"), "gpu": lat.get("gpus", []), "host": lat.get("host", {})}

    def fleet(self):
        out = []
        threads = []

        def fetch(entry, slot):
            try:
                req = urllib.request.Request(entry["url"].rstrip("/") + "/api/summary",
                                             headers={"Authorization": "Bearer " + (entry.get("token") or "")})
                with urllib.request.urlopen(req, timeout=3) as r:
                    slot.update(json.loads(r.read().decode()), reachable=True)
            except Exception as e:
                slot.update(reachable=False, error=str(e)[:200])

        for e in self.cfg.get("fleet", []):
            slot = {"name": e["name"], "url": e["url"]}
            out.append(slot)
            t = threading.Thread(target=fetch, args=(e, slot))
            t.start()
            threads.append(t)
        for t in threads:
            t.join(4)
        return [dict(self.summary(), name=self.label(), url=None, reachable=True, me=True)] + out

    # ------------------------------------------------------------------ state
    def state(self):
        return {
            "version": __version__, "demo": self.demo, "label": self.label(), "profile": self.profile,
            "plan": self.plan, "steps": self.step_status(), "autopilot": self.autopilot,
            "server": self.server_state(), "smoke": self.smoke, "attach_url": self.cfg.get("attach_url"),
            "prefs": self.cfg.get("prefs", {}), "registry_dirs": self.registry.get("dirs"),
            "registry_errors": self.registry.get("errors"), "lan": self.bind not in ("127.0.0.1", "::1", "localhost"),
            "job": self.runner.current.info() if self.runner.current else None,
            "suites": {k: v["label"] for k, v in bench.SUITES.items()},
        }


# ---------------------------------------------------------------------------- HTTP


def _loopback(addr):
    try:
        return ipaddress.ip_address(addr).is_loopback
    except ValueError:
        return False


class Handler(BaseHTTPRequestHandler):
    app = None
    server_version = "vllm-rdna-dash"

    def log_message(self, fmt, *args):
        pass

    # ---- helpers
    def _send(self, code, body, ctype="application/json", headers=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, default=str).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 2 * 1024 * 1024:
            raise ValueError("body too large")
        return json.loads(self.rfile.read(n).decode() or "{}") if n else {}

    def _authed(self):
        if not self.app.cfg_requires_token:
            return True
        tok = self.app.cfg["token"]
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer ") and hmac.compare_digest(auth[7:], tok):
            return True
        cookie = self.headers.get("Cookie", "")
        m = re.search(r"(?:^|;\s*)rdna_dash=([^;]+)", cookie)
        return bool(m and hmac.compare_digest(m.group(1), tok))

    # ---- routing
    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(u.query)
        if u.path == "/" and "token" in qs and self.app.cfg_requires_token:
            if hmac.compare_digest(qs["token"][0], self.app.cfg["token"]):
                return self._send(302, "", headers={"Location": "/", "Set-Cookie":
                                  "rdna_dash=%s; HttpOnly; SameSite=Strict; Path=/; Max-Age=31536000" % self.app.cfg["token"]})
        if not self._authed():
            if u.path.startswith("/api/"):
                return self._send(401, {"error": "token required"})
            return self._send(401, LOGIN % "", "text/html; charset=utf-8")
        if not u.path.startswith("/api/"):
            return self._static(u.path)
        try:
            self._api_get(u.path, qs)
        except KeyError as e:
            self._send(404, {"error": "not found: %s" % e})
        except Exception as e:
            self._send(500, {"error": str(e)})

    def _login(self):
        n = min(int(self.headers.get("Content-Length") or 0), 4096)
        form = urllib.parse.parse_qs(self.rfile.read(n).decode("utf-8", "replace"))
        tok = (form.get("token") or [""])[0].strip()
        if tok and hmac.compare_digest(tok, self.app.cfg["token"]):
            return self._send(303, "", headers={"Location": "/", "Set-Cookie":
                              "rdna_dash=%s; HttpOnly; SameSite=Strict; Path=/; Max-Age=31536000" % self.app.cfg["token"]})
        time.sleep(1)  # slow down guessing
        return self._send(401, LOGIN % '<p class="err">That token did not match.</p>', "text/html; charset=utf-8")

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/login":
            return self._login()
        if not self._authed():
            return self._send(401, {"error": "token required"})
        if self.headers.get("X-Rdna-Dash") != "1" and not self.headers.get("Authorization"):
            return self._send(403, {"error": "missing X-Rdna-Dash header"})
        try:
            body = self._body()
            if u.path == "/api/chat":
                return self._chat(body)
            self._api_post(u.path, body)
        except KeyError as e:
            self._send(404, {"error": "not found: %s" % e})
        except Exception as e:
            self._send(400, {"error": str(e)})

    def _static(self, path):
        if path in ("", "/"):
            path = "/index.html"
        full = os.path.normpath(os.path.join(WEB, path.lstrip("/")))
        if not full.startswith(WEB) or not os.path.isfile(full):
            full = os.path.join(WEB, "index.html")
        with open(full, "rb") as f:
            data = f.read()
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith("javascript"):
            ctype += "; charset=utf-8"
        self._send(200, data, ctype)

    def _api_get(self, path, qs):
        a = self.app
        if path == "/api/state":
            return self._send(200, a.state())
        if path == "/api/telemetry":
            return self._send(200, a.sampler.snapshot(history=qs.get("history", ["1"])[0] == "1"))
        if path == "/api/summary":
            return self._send(200, a.summary())
        if path == "/api/profile/share":
            return self._send(200, {"profile": profiler.redact(a.profile), "plan_summary": a.plan["summary"],
                                    "checks": [{"level": c["level"], "title": c["title"]} for c in a.plan["checks"]]})
        if path.startswith("/api/jobs/"):
            job = a.runner.jobs[path.split("/")[3]]
            off = int(qs.get("offset", ["0"])[0])
            text, new = job.read(off)
            return self._send(200, dict(job.info(), text=text, offset=new,
                                        problems=logdoctor.diagnose(text) if job.status == "failed" else []))
        if path == "/api/steps/log":
            sid = qs["step"][0]
            rec = store.load_state()["steps"].get(sid) or {}
            job = a.runner.jobs.get(rec.get("job") or "")
            if not job:
                return self._send(200, {"text": "", "status": rec.get("status")})
            text, _ = job.read(0, 1024 * 1024)
            return self._send(200, dict(job.info(), text=text, problems=logdoctor.diagnose(text)))
        if path == "/api/server/log":
            text = a.serve.log(int(qs.get("lines", ["400"])[0]))
            return self._send(200, {"text": text, "problems": logdoctor.diagnose(text), "phase": logdoctor.phase(text)})
        if path == "/api/bench":
            b = a.bench
            live = None
            if b:
                live = {"status": b.status, "progress": b.progress, "log": b.log[-40:], "cases": b.cases, "suite": b.suite_name}
            return self._send(200, {"live": live, "results": a.bench_results(), "suites": bench.SUITES})
        if path == "/api/topology":
            return self._send(200, a.topology(refresh=qs.get("refresh", ["0"])[0] == "1"))
        if path == "/api/fleet":
            return self._send(200, a.fleet())
        raise KeyError(path)

    def _api_post(self, path, body):
        a = self.app
        if path == "/api/autopilot":
            if body.get("cancel"):
                a.autopilot["cancel"] = True
                return self._send(200, a.autopilot)
            return self._send(200, a.start_autopilot())
        if path == "/api/profile/refresh":
            a.refresh_profile()
            return self._send(200, {"ok": True})
        if path == "/api/prefs":
            allowed = set(tuning.default_prefs())
            prefs = a.cfg.setdefault("prefs", {})
            for k, v in body.items():
                if k in allowed:
                    if v in (None, "", "auto") and k not in ("lan",):
                        prefs.pop(k, None)
                    else:
                        prefs[k] = v
            if "label" in body:
                a.cfg["label"] = (body.get("label") or "").strip()[:60] or None
            store.save_config(a.cfg)
            a.rebuild()
            return self._send(200, {"ok": True})
        m = re.match(r"^/api/steps/([\w-]+)/(run|reset)$", path)
        if m:
            sid, op = m.groups()
            if op == "reset":
                store.update_state(lambda st: st["steps"].pop(sid, None))
                return self._send(200, {"ok": True})
            return self._send(200, a.run_step(sid))
        m = re.match(r"^/api/jobs/(\w+)/cancel$", path)
        if m:
            return self._send(200, {"ok": a.runner.cancel(m.group(1))})
        if path == "/api/server/start":
            return self._send(200, a.serve.start(a.plan))
        if path == "/api/server/stop":
            return self._send(200, a.serve.stop(a.plan))
        if path == "/api/server/restart":
            a.serve.stop(a.plan)
            return self._send(200, a.serve.start(a.plan))
        if path == "/api/attach":
            url = (body.get("url") or "").strip() or None
            if url and not re.match(r"^https?://", url):
                raise ValueError("Use a full URL like http://127.0.0.1:8000/v1")
            a.cfg["attach_url"] = url
            store.save_config(a.cfg)
            return self._send(200, {"ok": True})
        if path == "/api/smoke":
            return self._send(200, a.run_smoke())
        if path == "/api/bench/start":
            return self._send(200, a.start_bench(body.get("suite", "quick"), body.get("thinking_off", True)))
        if path == "/api/bench/cancel":
            if a.bench:
                a.bench.cancelled = True
            return self._send(200, {"ok": True})
        if path == "/api/bench/delete":
            rid = re.sub(r"[^\w-]", "", body.get("id", ""))
            p = os.path.join(store.STATE_DIR, "bench", rid + ".json")
            if rid and os.path.exists(p):
                os.remove(p)
            return self._send(200, {"ok": True})
        if path == "/api/topology/bandwidth":
            return self._send(200, a.run_bandwidth(body.get("tool", "torch")))
        if path == "/api/fleet":
            name = (body.get("name") or "").strip()[:40]
            url = (body.get("url") or "").strip().rstrip("/")
            if not name or not re.match(r"^https?://", url):
                raise ValueError("Give the box a name and its dashboard URL (http://host:7860).")
            a.cfg["fleet"] = [e for e in a.cfg["fleet"] if e["name"] != name] + [{"name": name, "url": url, "token": body.get("token") or ""}]
            store.save_config(a.cfg)
            return self._send(200, {"ok": True})
        if path == "/api/fleet/remove":
            a.cfg["fleet"] = [e for e in a.cfg["fleet"] if e["name"] != body.get("name")]
            store.save_config(a.cfg)
            return self._send(200, {"ok": True})
        raise KeyError(path)

    def _chat(self, body):
        """Stream a chat completion from the model server straight to the browser."""
        a = self.app
        c = bench.Client(a.server_url())
        payload = {"model": body.get("model"), "messages": body.get("messages", []), "stream": True,
                   "max_tokens": min(int(body.get("max_tokens") or 512), 8192),
                   "temperature": float(body.get("temperature", 0.7)),
                   "stream_options": {"include_usage": True}}
        if body.get("thinking") is False:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if not payload["model"]:
            payload["model"] = c.models()[0]["id"]
        conn = c._conn(900)
        conn.request("POST", c.prefix + "/v1/chat/completions", body=json.dumps(payload), headers=c._headers())
        r = conn.getresponse()
        self.send_response(r.status)
        self.send_header("Content-Type", r.getheader("Content-Type") or "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while True:
                line = r.readline()
                if not line:
                    break
                self.wfile.write(line)
                if line == b"\n":
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            conn.close()


def run(host="127.0.0.1", port=7860, demo=False):
    app = App(demo=demo, bind=host)
    app.cfg_requires_token = not _loopback(host) if host not in ("localhost",) else False
    Handler.app = app
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    print("vllm-rdna-dash %s%s" % (__version__, " (demo mode)" if demo else ""))
    if app.cfg_requires_token:
        print("  Open http://<this-box>:%d from any machine on your network and sign in with:" % port)
        print("  token: %s" % app.cfg["token"])
        print("  (or open http://<this-box>:%d/?token=%s once; set RDNA_DASH_TOKEN to pick your own)" % (port, app.cfg["token"]))
    else:
        print("  Open: http://%s:%d/" % (shown, port))
        print("  Local only. To reach it from another machine: ssh -L %d:127.0.0.1:%d <this-box>" % (port, port))
        print("  or restart with --host 0.0.0.0 (token required).")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
