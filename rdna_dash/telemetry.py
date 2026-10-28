"""Live samples: per-GPU sysfs readings, host CPU/RAM, and vLLM /metrics.

A background thread samples once a second into fixed-size rings, so the UI can
draw the last few minutes without the browser polling hardware itself.
"""

import collections
import glob
import os
import re
import threading
import time
import urllib.request

RING = 300  # samples kept (5 minutes at 1 Hz)


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return None


def _num(path, scale=1.0):
    v = _read(path)
    try:
        return float(v) / scale
    except Exception:
        return None


def _dpm(path):
    """Current clock from pp_dpm_* ('1: 2200Mhz *')."""
    txt = _read(path) or ""
    for line in txt.splitlines():
        if line.strip().endswith("*"):
            m = re.search(r"(\d+)\s*Mhz", line, re.I)
            if m:
                return int(m.group(1))
    return None


class GpuReader(object):
    def __init__(self, gpus):
        self.gpus = []
        for g in gpus:
            dev = "/sys/bus/pci/devices/" + g["pci"]
            hw = (glob.glob(dev + "/hwmon/hwmon*") or [None])[0]
            temps = {}
            if hw:
                for lab in glob.glob(hw + "/temp*_label"):
                    temps[(_read(lab) or "").lower()] = lab.replace("_label", "_input")
            self.gpus.append({"index": g["index"], "name": g["name"], "dev": dev, "hw": hw, "temps": temps})

    def read(self):
        out = []
        for g in self.gpus:
            d, hw = g["dev"], g["hw"]
            power = None
            if hw:
                power = _num(hw + "/power1_average", 1e6)
                if power is None:
                    power = _num(hw + "/power1_input", 1e6)
            out.append({
                "index": g["index"],
                "util": _num(d + "/gpu_busy_percent"),
                "vram_used": _num(d + "/mem_info_vram_used", 1024 ** 3),
                "vram_total": _num(d + "/mem_info_vram_total", 1024 ** 3),
                "temp_edge": _num(g["temps"].get("edge", ""), 1000) if g["temps"].get("edge") else (_num(hw + "/temp1_input", 1000) if hw else None),
                "temp_junction": _num(g["temps"]["junction"], 1000) if g["temps"].get("junction") else None,
                "temp_mem": _num(g["temps"]["mem"], 1000) if g["temps"].get("mem") else None,
                "power": power,
                "power_cap": _num(hw + "/power1_cap", 1e6) if hw else None,
                "sclk": _dpm(d + "/pp_dpm_sclk"),
                "mclk": _dpm(d + "/pp_dpm_mclk"),
            })
        return out


class HostReader(object):
    def __init__(self):
        self.prev = None

    def read(self):
        cpu = None
        line = (_read("/proc/stat") or "").splitlines()[:1]
        if line:
            vals = [int(x) for x in line[0].split()[1:]]
            idle, total = vals[3] + (vals[4] if len(vals) > 4 else 0), sum(vals)
            if self.prev:
                dt, di = total - self.prev[1], idle - self.prev[0]
                cpu = 100.0 * (1 - di / float(dt)) if dt else None
            self.prev = (idle, total)
        mem = {}
        for l in (_read("/proc/meminfo") or "").splitlines():
            p = l.split()
            if len(p) >= 2:
                mem[p[0].rstrip(":")] = int(p[1])
        total = mem.get("MemTotal", 0) / 1048576.0
        avail = mem.get("MemAvailable", 0) / 1048576.0
        return {"cpu": cpu, "ram_used": total - avail if total else None, "ram_total": total or None}


# ---------------------------------------------------------------------------- vLLM metrics

_METRIC = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)")


def parse_prom(text):
    """Sum Prometheus samples by metric name, dropping labels."""
    out = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = _METRIC.match(line)
        if not m:
            continue
        try:
            v = float(m.group(3))
        except ValueError:
            continue
        out[m.group(1)] = out.get(m.group(1), 0.0) + v
    return out


def pick(m, *names):
    for n in names:
        if n in m:
            return m[n]
    return None


class VllmPoller(object):
    """Turns cumulative vLLM counters into rates and window averages."""

    def __init__(self):
        self.prev = None

    def sample(self, base_url, timeout=1.5):
        base = base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        st = {"reachable": False, "healthy": False}
        try:
            with urllib.request.urlopen(base + "/health", timeout=timeout) as r:
                st["reachable"] = True
                st["healthy"] = r.status == 200
        except urllib.error.HTTPError as e:
            st["reachable"] = True
            st["healthy"] = False
            st["http"] = e.code
        except Exception:
            self.prev = None
            return st
        try:
            with urllib.request.urlopen(base + "/metrics", timeout=timeout) as r:
                m = parse_prom(r.read().decode("utf-8", "replace"))
        except Exception:
            return st
        now = time.time()
        cur = {
            "t": now,
            "gen": pick(m, "vllm:generation_tokens_total", "vllm:generation_tokens"),
            "prompt": pick(m, "vllm:prompt_tokens_total", "vllm:prompt_tokens"),
            "ttft_sum": pick(m, "vllm:time_to_first_token_seconds_sum"),
            "ttft_n": pick(m, "vllm:time_to_first_token_seconds_count"),
            "e2e_sum": pick(m, "vllm:e2e_request_latency_seconds_sum"),
            "e2e_n": pick(m, "vllm:e2e_request_latency_seconds_count"),
            "pc_hit": pick(m, "vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total"),
            "pc_q": pick(m, "vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total"),
            "sd_acc": pick(m, "vllm:spec_decode_num_accepted_tokens_total"),
            "sd_draft": pick(m, "vllm:spec_decode_num_draft_tokens_total"),
            "done": pick(m, "vllm:request_success_total"),
        }
        st.update({
            "running": pick(m, "vllm:num_requests_running"),
            "waiting": pick(m, "vllm:num_requests_waiting"),
            "kv_usage": pick(m, "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
            "requests_done": cur["done"],
        })
        if st["kv_usage"] is not None and st["kv_usage"] <= 1.0:
            st["kv_usage"] *= 100.0
        p = self.prev
        if p and cur["t"] > p["t"]:
            dt = cur["t"] - p["t"]

            def rate(k):
                return (cur[k] - p[k]) / dt if cur[k] is not None and p[k] is not None and cur[k] >= p[k] else None

            def avg(s, n):
                if None in (cur[s], p[s], cur[n], p[n]) or cur[n] <= p[n]:
                    return None
                return (cur[s] - p[s]) / (cur[n] - p[n])

            def ratio(a, b):
                if None in (cur[a], p[a], cur[b], p[b]) or cur[b] <= p[b]:
                    return None
                return 100.0 * (cur[a] - p[a]) / (cur[b] - p[b])

            st.update({"gen_tps": rate("gen"), "prompt_tps": rate("prompt"), "ttft": avg("ttft_sum", "ttft_n"),
                       "e2e": avg("e2e_sum", "e2e_n"), "prefix_hit": ratio("pc_hit", "pc_q"),
                       "spec_accept": ratio("sd_acc", "sd_draft")})
        self.prev = cur
        return st


class Sampler(object):
    def __init__(self, gpu_reader, host_reader, url_fn, period=1.0):
        self.gpu_reader, self.host_reader, self.url_fn = gpu_reader, host_reader, url_fn
        self.period = period
        self.gpu = collections.deque(maxlen=RING)
        self.host = collections.deque(maxlen=RING)
        self.server = collections.deque(maxlen=RING)
        self.poller = VllmPoller()
        self.latest = {}
        self._tick = 0

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            t = time.time()
            try:
                g = self.gpu_reader.read()
                h = self.host_reader.read()
                self.gpu.append({"t": t, "gpus": g})
                self.host.append(dict(h, t=t))
                self.latest["gpus"], self.latest["host"] = g, h
                if self._tick % 2 == 0:
                    url = self.url_fn()
                    s = self.poller.sample(url) if url else {"reachable": False}
                    s["t"] = t
                    self.server.append(s)
                    self.latest["server"] = s
            except Exception:
                pass
            self._tick += 1
            time.sleep(max(0.05, self.period - (time.time() - t)))

    def snapshot(self, history=True):
        out = {"latest": self.latest}
        if history:
            out["gpu"] = list(self.gpu)
            out["server"] = list(self.server)
            out["host"] = list(self.host)
        return out
