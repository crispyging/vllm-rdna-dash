"""Demo mode: a pretend 4 x V620 box and a fake vLLM server, so the whole UI,
the autopilot and the benchmark can be tried on any machine."""

import json
import math
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer



def free_port():
    """A free local port, so several demo instances never share a fake server."""
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def demo_profile():
    gpus = []
    for i, bus in enumerate(["03", "06", "43", "46"]):
        gpus.append({"pci": "0000:%s:00.0" % bus, "card": "card%d" % (i + 1), "device_id": "73a1", "subsystem_id": "0e34",
                     "name": "Radeon PRO V620", "gfx": "gfx1030", "vram_gb": 29.98, "nominal_vram_gb": 32,
                     "power_cap_w": 250, "power_cap_min_w": 250, "power_cap_max_w": 250,
                     "ecc_errors": {"ue": 0, "ce": 0}, "index": i, "integrated": False})
    return {
        "schema": 1, "hostname": "demo-box",
        "os": {"name": "Ubuntu 24.04.3 LTS", "id": "ubuntu", "family": "debian", "version_id": "24.04",
               "package_manager": "apt", "selinux": "absent", "kernel": "6.8.0-60-generic", "virt": "none",
               "iommu_pt": True, "ras_disabled_param": False, "secure_boot": "disabled"},
        "cpu": {"model": "AMD EPYC 7452 32-Core Processor", "vendor": "amd", "threads": 64},
        "memory": {"total_gb": 251.5, "available_gb": 230.1},
        "disk": {"home_free_gb": 1480.0, "data_free_gb": None},
        "gpus": gpus,
        "rocm": {"installs": {"/opt/rocm": "7.14.0", "/opt/rocm/core-7.14": "7.14.0"}, "primary": "7.14.0",
                 "sdk_root": "/opt/rocm/core-7.14", "home_env": None, "hipcc": "/opt/rocm/core-7.14/bin/hipcc",
                 "rocminfo": True, "agents": ["gfx1030"] * 4},
        "access": {"user": "demo", "groups": ["demo", "video", "render", "docker"], "in_video": True, "in_render": True,
                   "kfd_exists": True, "kfd_rw": True, "render_nodes": 4, "amdgpu_loaded": True},
        "container": {"installed": True, "usable": True, "engine": "docker", "rootless": False,
                      "compose": ["docker", "compose"], "vllm_rdna_images": [], "engines": []},
        "tools": {"python312": "/usr/bin/python3.12", "uv": None, "git": "/usr/bin/git", "curl": "/usr/bin/curl"},
        "env": {"hf_token_set": False},
        "servers": [],
    }


def demo_topology():
    devs = []
    for i, bus in enumerate(["03", "06", "43", "46"]):
        root = "pci0000:00" if i < 2 else "pci0000:40"
        rp = "0000:00:03.1" if i < 2 else "0000:40:03.1"
        sw_up = "0000:%02x:00.0" % (1 if i < 2 else 0x41)
        sw_dn = "0000:%02x:%02x.0" % ((2 if i < 2 else 0x42), (i % 2) * 8)
        devs.append({"index": i, "pci": "0000:%s:00.0" % bus, "name": "Radeon PRO V620", "root": root,
                     "numa": "0" if i < 2 else "1", "socket": 0 if i < 2 else 1,
                     "bridges": [{"addr": rp, "kind": "root-port", "vendor_name": "AMD"},
                                 {"addr": sw_up, "kind": "switch", "vendor_name": "Broadcom/PLX"},
                                 {"addr": sw_dn, "kind": "switch", "vendor_name": "Broadcom/PLX"},
                                 {"addr": "0000:%02x:00.0" % (int(bus, 16) - 2), "kind": "on-card", "vendor_name": "AMD GPU"},
                                 {"addr": "0000:%02x:00.0" % (int(bus, 16) - 1), "kind": "on-card", "vendor_name": "AMD GPU"}],
                     "link": {"gen": 4, "width": 16, "max_gen": 4, "max_width": 16}, "kfd_node": i + 1, "large_bar": 32.0})
    pairs = []
    for i in range(4):
        for j in range(i + 1, 4):
            same = (i < 2) == (j < 2)
            pairs.append({"a": i, "b": j, "via": "switch" if same else "cpu-interconnect",
                          "meet": "0000:%02x:00.0" % (1 if i < 2 else 0x41) if same else None,
                          "hops": 2 if same else 6, "middlemen": [], "kfd_p2p": True, "kfd_weight": 15 if same else 52})
    return {"gpus": devs, "pairs": pairs, "sockets": {"count": 2, "model": "AMD EPYC 7452 32-Core Processor"},
            "switches": [{"addr": "0000:01:00.0", "vendor": "Broadcom/PLX", "gpus": [0, 1]},
                         {"addr": "0000:41:00.0", "vendor": "Broadcom/PLX", "gpus": [2, 3]}],
            "kfd": True, "kfd_p2p_supported": True,
            "tools": {"rocm_bandwidth_test": "/opt/rocm/bin/rocm-bandwidth-test", "amd_smi": "/opt/rocm/bin/amd-smi", "rocm_smi": None}}


def demo_bandwidth():
    gbs = [[None if i == j else (24.6 if (i < 2) == (j < 2) else 13.1) + random.uniform(-0.4, 0.4) for j in range(4)] for i in range(4)]
    return {"devices": 4, "pci": None, "peer": [[True] * 4] * 4, "gbs": [[round(x, 2) if x else None for x in r] for r in gbs],
            "h2d": [round(25.1 + random.uniform(-0.5, 0.5), 2) for _ in range(4)]}


class FakeVllm(object):
    """Enough of the vLLM OpenAI server for the dashboard and the benchmark."""

    def __init__(self, port=None):
        self.port = port or free_port()
        self.ready_at = None
        self.running = 0
        self.counters = {"gen": 0.0, "prompt": 0.0, "ttft_sum": 0.0, "ttft_n": 0, "e2e_sum": 0.0, "e2e_n": 0,
                         "done": 0, "pc_hit": 0.0, "pc_q": 0.0, "sd_acc": 0.0, "sd_draft": 0.0}
        self.lock = threading.Lock()
        self.httpd = None
        self.load = 0.0

    def up(self):
        return self.ready_at is not None and time.time() >= self.ready_at

    def start_server_process(self, boot_seconds=8, background_users=3):
        self.ready_at = time.time() + boot_seconds
        if background_users and not getattr(self, "_traffic", False):
            self._traffic = True
            for k in range(background_users):
                threading.Thread(target=self._user, args=(k,), daemon=True).start()
        if not self.httpd:
            fake = self

            class H(BaseHTTPRequestHandler):
                def log_message(self, *a):
                    pass

                def _json(self, code, obj):
                    b = json.dumps(obj).encode()
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(b)))
                    self.end_headers()
                    self.wfile.write(b)

                def do_GET(self):
                    if self.path == "/health":
                        self.send_response(200 if fake.up() else 503)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                    elif not fake.up():
                        self._json(503, {"error": "starting"})
                    elif self.path == "/v1/models":
                        self._json(200, {"object": "list", "data": [{"id": "flash-next", "object": "model", "max_model_len": 262144}]})
                    elif self.path == "/metrics":
                        c = fake.counters
                        body = "\n".join([
                            "vllm:num_requests_running %d" % fake.running, "vllm:num_requests_waiting 0",
                            "vllm:kv_cache_usage_perc %.3f" % min(0.95, 0.04 + fake.running * 0.06),
                            "vllm:generation_tokens_total %f" % c["gen"], "vllm:prompt_tokens_total %f" % c["prompt"],
                            "vllm:time_to_first_token_seconds_sum %f" % c["ttft_sum"],
                            "vllm:time_to_first_token_seconds_count %d" % c["ttft_n"],
                            "vllm:e2e_request_latency_seconds_sum %f" % c["e2e_sum"],
                            "vllm:e2e_request_latency_seconds_count %d" % c["e2e_n"],
                            "vllm:request_success_total %d" % c["done"],
                            "vllm:prefix_cache_hits_total %f" % c["pc_hit"], "vllm:prefix_cache_queries_total %f" % c["pc_q"],
                            "vllm:external_prefix_cache_hits_total %f" % (c["pc_q"] * 0.04),
                            "vllm:external_prefix_cache_queries_total %f" % (c["pc_q"] * 0.9),
                            "vllm:spec_decode_num_accepted_tokens_total %f" % c["sd_acc"],
                            "vllm:spec_decode_num_draft_tokens_total %f" % c["sd_draft"]]) + "\n"
                        b = body.encode()
                        self.send_response(200)
                        self.send_header("Content-Length", str(len(b)))
                        self.end_headers()
                        self.wfile.write(b)
                    else:
                        self._json(404, {"error": "not found"})

                def do_POST(self):
                    n = int(self.headers.get("Content-Length", "0"))
                    req = json.loads(self.rfile.read(n).decode() or "{}")
                    if not fake.up():
                        return self._json(503, {"error": "starting"})
                    if self.path == "/tokenize":
                        return self._json(200, {"count": max(1, int(len(req.get("prompt", "")) / 4.4))})
                    if self.path != "/v1/chat/completions":
                        return self._json(404, {"error": "not found"})
                    fake.chat(self, req)

            self.httpd = ThreadingHTTPServer(("127.0.0.1", self.port), H)
            self.httpd.daemon_threads = True
            threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def _user(self, k):
        """A pretend client, so the demo dashboard has steady live numbers."""
        import urllib.request
        rnd = random.Random(k)
        time.sleep(k * 1.3)
        while True:
            if not self.up():
                time.sleep(1)
                continue
            body = json.dumps({"model": "flash-next", "max_tokens": rnd.randint(120, 260), "stream": True,
                               "messages": [{"role": "user", "content": "lorem ipsum dolor " * rnd.randint(150, 900)}]}).encode()
            try:
                req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % self.port, data=body,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    r.read()
            except Exception:
                pass
            time.sleep(rnd.uniform(0.2, 1.5))

    def stop(self):
        self.ready_at = None

    def chat(self, h, req):
        prompt_tokens = 0
        has_image = False
        text = ""
        for m in req.get("messages", []):
            c = m.get("content")
            if isinstance(c, list):
                for part in c:
                    if part.get("type") == "text":
                        text += part["text"]
                    else:
                        has_image = True
                        prompt_tokens += 1100
            else:
                text += c or ""
        prompt_tokens += max(1, int(len(text) / 4.4))
        max_tokens = int(req.get("max_tokens") or 256)
        with self.lock:
            self.running += 1
            conc = self.running
        # Prefill slows with concurrency and context; decode per stream slows with concurrency.
        prefill_rate = 1900.0 / (1 + 0.15 * (conc - 1)) / (1 + prompt_tokens / 160000.0)
        decode_rate = 52.0 / (1 + 0.22 * (conc - 1))
        t0 = time.time()
        # Count prompt tokens as they are read so prefill shows up smoothly in /metrics.
        slices = 10
        for _ in range(slices):
            time.sleep(min(60, prompt_tokens / prefill_rate) / slices)
            with self.lock:
                self.counters["prompt"] += prompt_tokens / float(slices)
        ttft = time.time() - t0
        stream = req.get("stream")
        import re as _re
        key = _re.search(r"pass key is (\d+)", text)
        words = ("The pass key is %s. " % key.group(1) if key else "") + (
            "Top-left is red, top-right green, bottom-left blue, bottom-right yellow. " if has_image else "") + \
            "The cache held steady as the lattice of signals moved through the bridge. " * 40
        toks = words.split(" ")
        n = max_tokens if req.get("ignore_eos") else min(max_tokens, 60)
        if stream:
            h.send_response(200)
            h.send_header("Content-Type", "text/event-stream")
            h.end_headers()
        out = []
        for i in range(n):
            piece = toks[i % len(toks)] + " "
            out.append(piece)
            if stream:
                ch = {"choices": [{"index": 0, "delta": {"content": piece}}]}
                try:
                    h.wfile.write(("data: %s\n\n" % json.dumps(ch)).encode())
                    h.wfile.flush()
                except Exception:
                    break
            time.sleep(1.0 / decode_rate)
            with self.lock:
                self.counters["gen"] += 1
        e2e = time.time() - t0
        with self.lock:
            self.running -= 1
            c = self.counters
            c["ttft_sum"] += ttft
            c["ttft_n"] += 1
            c["e2e_sum"] += e2e
            c["e2e_n"] += 1
            c["done"] += 1
            c["pc_q"] += prompt_tokens
            c["pc_hit"] += prompt_tokens * 0.1
            c["sd_draft"] += n * 0.66
            c["sd_acc"] += n * 0.5
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": n, "total_tokens": prompt_tokens + n}
        if stream:
            try:
                h.wfile.write(("data: %s\n\n" % json.dumps({"choices": [], "usage": usage})).encode())
                h.wfile.write(b"data: [DONE]\n\n")
            except Exception:
                pass
        else:
            h._json(200, {"choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(out)}}], "usage": usage})


class DemoGpuReader(object):
    def __init__(self, fake):
        self.fake = fake

    def read(self):
        t = time.time()
        out = []
        busy = self.fake.running > 0
        loaded = self.fake.ready_at is not None
        for i in range(4):
            util = (70 + 25 * math.sin(t / 3 + i)) if busy else (3 + random.random() * 4)
            out.append({"index": i, "util": round(util), "vram_used": (27.4 + 0.3 * math.sin(t / 9 + i)) if loaded else 0.4,
                        "vram_total": 29.98, "temp_edge": (62 if busy else 41) + 3 * math.sin(t / 7 + i),
                        "temp_junction": (78 if busy else 44) + 4 * math.sin(t / 5 + i),
                        "temp_mem": (74 if busy else 50) + 2 * math.sin(t / 6 + i),
                        "power": (205 if busy else 38) + 15 * math.sin(t / 4 + i), "power_cap": 250,
                        "sclk": 2450 if busy else 500, "mclk": 1000 if loaded else 96})
        return out


class DemoHostReader(object):
    def read(self):
        return {"cpu": 6 + random.random() * 4, "ram_used": 62.0 + random.random(), "ram_total": 251.5}


class DemoServe(object):
    def __init__(self, fake):
        self.fake = fake
        self.started = None
        self.lines = []

    def write_compose(self, launch):
        return launch.get("compose_file")

    def start(self, plan):
        self.started = time.time()
        self.fake.start_server_process()
        self.lines = ["$ " + plan["launch"]["command"].replace("\n", " "), "INFO Loading safetensors checkpoint shards",
                      "INFO Compiling a graph for dynamic shape", "INFO Capturing CUDA graph shapes",
                      "INFO Maximum concurrency for 262144 tokens per request: 8.00x",
                      "INFO Application startup complete."]
        return {"ok": True, "message": "Started (demo)."}

    def stop(self, plan=None):
        self.started = None
        self.fake.stop()
        return {"ok": True, "message": "Stopped (demo)."}

    def status(self, plan=None):
        return {"process": "running" if self.started else "stopped", "kind": "demo", "started": self.started}

    def log(self, lines=400):
        if not self.started:
            return ""
        shown = min(len(self.lines), 1 + int((time.time() - self.started) / 1.5))
        return "\n".join(self.lines[:shown])
