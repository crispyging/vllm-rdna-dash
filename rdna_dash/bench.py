"""Consistent benchmark for any OpenAI-compatible server.

Same suite, same prompts, same output length for every model, so runs compare:
- prompts are built from a seeded corpus and sized to exact token counts with the
  server's /tokenize, so "8k" means 8192 prompt tokens for this model's tokenizer;
- each request starts with a unique salt so the prefix cache cannot flatter a run;
- output is pinned (temperature 0, ignore_eos, fixed max_tokens);
- vision cases use generated images (no downloads) and are skipped, not failed,
  when the model is text-only;
- a passkey hidden mid-context checks the model still reads long prompts.
"""

import base64
import http.client
import json
import random
import statistics
import struct
import threading
import time
import urllib.parse
import uuid
import zlib

SUITE_VERSION = 1

SUITES = {
    "quick": {"label": "Quick (about 2 min)", "contexts": [1024, 8192], "concurrency": [1, 4],
              "vision": [512], "output": 128, "repeats": 1},
    "standard": {"label": "Standard (about 10 min)", "contexts": [1024, 8192, 32768], "concurrency": [1, 4, 8],
                 "vision": [512, 1024], "output": 256, "repeats": 2},
    "long": {"label": "Long context (20+ min)", "contexts": [1024, 8192, 32768, 65536, 131072],
             "concurrency": [1, 4], "vision": [1024], "output": 256, "repeats": 1},
}

WORDS = ("signal kernel memory bridge vector packet thread cache lattice river copper stone harbor field "
         "engine market orbit garden ledger window canvas rhythm pillar meadow summit compass beacon "
         "anchor quarry timber falcon glacier prairie canyon lantern harvest").split()


def corpus(seed, n_words):
    rnd = random.Random(seed)
    out, sentence = [], []
    for _ in range(n_words):
        sentence.append(rnd.choice(WORDS))
        if len(sentence) >= rnd.randint(8, 16):
            out.append(" ".join(sentence).capitalize() + ".")
            sentence = []
    return " ".join(out)


# ---------------------------------------------------------------------------- images


def _png(width, height, pixel):
    raw = b"".join(b"\x00" + b"".join(pixel(x, y) for x in range(width)) for y in range(height))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


_IMG_CACHE = {}


def test_image(size):
    """Deterministic image: colored quadrants with a grid, so the answer is checkable."""
    if size not in _IMG_CACHE:
        h = size // 2
        colors = {(0, 0): b"\xd0\x30\x30", (1, 0): b"\x30\x90\x30", (0, 1): b"\x30\x50\xd0", (1, 1): b"\xe0\xc0\x20"}

        def px(x, y):
            if x % 64 == 0 or y % 64 == 0:
                return b"\xff\xff\xff"
            return colors[(1 if x >= h else 0, 1 if y >= h else 0)]

        _IMG_CACHE[size] = "data:image/png;base64," + base64.b64encode(_png(size, size, px)).decode()
    return _IMG_CACHE[size]


# ---------------------------------------------------------------------------- client


class Client(object):
    def __init__(self, base_url, model=None, api_key=None, timeout=900):
        u = urllib.parse.urlparse(base_url.rstrip("/"))
        self.https = u.scheme == "https"
        self.host = u.hostname
        self.port = u.port or (443 if self.https else 80)
        self.prefix = u.path[:-3] if u.path.endswith("/v1") else u.path
        self.timeout = timeout
        self.api_key = api_key
        self.model = model

    def _conn(self, timeout=None):
        cls = http.client.HTTPSConnection if self.https else http.client.HTTPConnection
        return cls(self.host, self.port, timeout=timeout or self.timeout)

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = "Bearer " + self.api_key
        return h

    def get_json(self, path, timeout=10):
        c = self._conn(timeout)
        try:
            c.request("GET", self.prefix + path, headers=self._headers())
            r = c.getresponse()
            body = r.read()
            if r.status != 200:
                raise RuntimeError("%s -> HTTP %d" % (path, r.status))
            return json.loads(body.decode())
        finally:
            c.close()

    def post_json(self, path, payload, timeout=60):
        c = self._conn(timeout)
        try:
            c.request("POST", self.prefix + path, body=json.dumps(payload), headers=self._headers())
            r = c.getresponse()
            body = r.read()
            if r.status != 200:
                raise RuntimeError("%s -> HTTP %d: %s" % (path, r.status, body[:300].decode("utf-8", "replace")))
            return json.loads(body.decode())
        finally:
            c.close()

    def models(self):
        data = self.get_json("/v1/models").get("data", [])
        return data

    def count_tokens(self, text):
        try:
            return self.post_json("/tokenize", {"model": self.model, "prompt": text, "add_special_tokens": False})["count"]
        except Exception:
            return None

    def stream_chat(self, messages, max_tokens, extra=None):
        """One streamed chat completion. Returns timing and usage."""
        payload = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
                   "stream": True, "stream_options": {"include_usage": True}}
        payload.update(extra or {})
        c = self._conn()
        t0 = time.perf_counter()
        first, last, stamps, text, usage = None, None, [], [], None
        try:
            c.request("POST", self.prefix + "/v1/chat/completions", body=json.dumps(payload), headers=self._headers())
            r = c.getresponse()
            if r.status != 200:
                body = r.read().decode("utf-8", "replace")
                return {"ok": False, "status": r.status, "error": body[:500]}
            while True:
                line = r.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                obj = json.loads(data.decode())
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices", []):
                    delta = ch.get("delta", {})
                    piece = (delta.get("content") or "") + (delta.get("reasoning_content") or delta.get("reasoning") or "")
                    if piece:
                        now = time.perf_counter()
                        if first is None:
                            first = now
                        last = now
                        stamps.append(now)
                        text.append(piece)
        except Exception as e:
            return {"ok": False, "error": str(e)}
        finally:
            c.close()
        end = time.perf_counter()
        if first is None:
            return {"ok": False, "error": "no tokens streamed"}
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        return {"ok": True, "t0": t0, "ttft": first - t0, "e2e": end - t0, "decode_time": (last or first) - first,
                "itl": gaps, "text": "".join(text),
                "prompt_tokens": (usage or {}).get("prompt_tokens"),
                "completion_tokens": (usage or {}).get("completion_tokens") or len(stamps)}


# ---------------------------------------------------------------------------- prompts


def build_prompt(client, target_tokens, seed, passkey):
    """Text whose token count is close to target, with a passkey at 50% depth."""
    question = "\n\nWhat is the pass key? Reply with the number first."
    needle = " The pass key is %d. Remember it. " % passkey
    budget = max(64, target_tokens - 64)
    ratio = 1.3  # words per token, refined with /tokenize
    words = int(budget / ratio)
    for _ in range(4):
        body = corpus(seed, words)
        half = len(body) // 2
        text = body[:half] + needle + body[half:] + question
        n = client.count_tokens(text)
        if n is None:
            return text, None
        if abs(n - target_tokens) <= max(16, target_tokens * 0.01):
            return text, n
        words = max(16, int(words * target_tokens / float(n)))
    return text, n


def pct(values, p):
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


def summarize(results, wall):
    ok = [r for r in results if r.get("ok")]
    out = {"requests": len(results), "errors": len(results) - len(ok)}
    if not ok:
        out["error"] = next((r.get("error") for r in results if r.get("error")), "all requests failed")
        return out
    ttft = [r["ttft"] for r in ok]
    decode = [(r["completion_tokens"] - 1) / r["decode_time"] for r in ok if r["decode_time"] > 0 and r["completion_tokens"] > 1]
    prefill = [r["prompt_tokens"] / r["ttft"] for r in ok if r.get("prompt_tokens") and r["ttft"] > 0]
    itl = [g for r in ok for g in r["itl"]]
    out.update({
        "ttft_p50": pct(ttft, 50), "ttft_p95": pct(ttft, 95),
        "prefill_tps": statistics.median(prefill) if prefill else None,
        "decode_tps": statistics.median(decode) if decode else None,
        "itl_p50_ms": pct(itl, 50) * 1000 if itl else None, "itl_p95_ms": pct(itl, 95) * 1000 if itl else None,
        "e2e_p50": pct([r["e2e"] for r in ok], 50),
        "output_tps": sum(r["completion_tokens"] for r in ok) / wall if wall > 0 else None,
        "prompt_tokens": statistics.median([r["prompt_tokens"] for r in ok if r.get("prompt_tokens")] or [0]),
    })
    if any("passkey" in r for r in ok):
        hits = [r["passkey_ok"] for r in ok if "passkey" in r]
        out["recall"] = sum(hits) / float(len(hits))
    return out


# ---------------------------------------------------------------------------- runner


class Bench(object):
    def __init__(self, base_url, suite="standard", model=None, api_key=None, thinking_off=True, fingerprint=None):
        self.client = Client(base_url, model, api_key)
        self.suite_name = suite
        self.suite = SUITES[suite]
        self.thinking_off = thinking_off
        self.fingerprint = fingerprint or {}
        self.cases = []
        self.progress = {"done": 0, "total": 0, "current": None}
        self.status = "idle"
        self.cancelled = False
        self.result = None
        self.log = []

    def _extra(self):
        e = {"ignore_eos": True, "min_tokens": self.suite["output"]}
        if self.thinking_off:
            e["chat_template_kwargs"] = {"enable_thinking": False}
        return e

    def plan(self, max_len):
        cases = []
        out = self.suite["output"]
        for ctx in self.suite["contexts"]:
            if max_len and ctx + out + 128 > max_len:
                cases.append({"kind": "text", "context": ctx, "skip": "exceeds max_model_len %d" % max_len})
                continue
            for c in self.suite["concurrency"]:
                cases.append({"kind": "text", "context": ctx, "concurrency": c})
        for size in self.suite["vision"]:
            for c in (1, max(self.suite["concurrency"])):
                cases.append({"kind": "vision", "image": size, "concurrency": c})
        return cases

    def note(self, msg):
        self.log.append("%s  %s" % (time.strftime("%H:%M:%S"), msg))

    def run(self):
        self.status = "running"
        started = time.time()
        try:
            models = self.client.models()
            if not models:
                raise RuntimeError("server lists no models")
            if not self.client.model:
                self.client.model = models[0]["id"]
            info = next((m for m in models if m["id"] == self.client.model), models[0])
            max_len = info.get("max_model_len")
            self.cases = self.plan(max_len)
            runnable = [c for c in self.cases if not c.get("skip")]
            self.progress["total"] = sum(c["concurrency"] * self.suite["repeats"] for c in runnable) + 1
            self.note("model %s, max_model_len %s, suite %s v%d" % (self.client.model, max_len, self.suite_name, SUITE_VERSION))
            # Warm-up so first-call compile or allocation does not land in a case.
            self.client.stream_chat([{"role": "user", "content": "Say hi."}], 8, self._extra())
            self.progress["done"] += 1
            vision_ok = None
            for case in self.cases:
                if self.cancelled:
                    break
                if case.get("skip"):
                    self.note("skip %s: %s" % (self._label(case), case["skip"]))
                    continue
                if case["kind"] == "vision" and vision_ok is False:
                    case["skip"] = "model is text-only"
                    continue
                self.progress["current"] = self._label(case)
                self.note("run " + self._label(case))
                res = self._run_case(case)
                if case["kind"] == "vision" and res["summary"].get("errors") == res["summary"].get("requests"):
                    err = res["summary"].get("error", "") or ""
                    if any(w in err.lower() for w in ("image", "multimodal", "vision", "not support", "language-model-only")):
                        vision_ok = False
                        case["skip"] = "model is text-only"
                        self.note("vision not supported by this model; skipping vision cases")
                        continue
                if case["kind"] == "vision":
                    vision_ok = True
                case.update(res)
            self.status = "cancelled" if self.cancelled else "done"
        except Exception as e:
            self.status = "failed"
            self.note("failed: %s" % e)
        self.result = {
            "id": time.strftime("%Y%m%d-%H%M%S", time.localtime(started)) + "-" + uuid.uuid4().hex[:4],
            "suite": self.suite_name, "suite_version": SUITE_VERSION, "status": self.status,
            "started": started, "duration": time.time() - started, "model": self.client.model,
            "fingerprint": self.fingerprint, "cases": self.cases, "log": self.log,
        }
        return self.result

    def _label(self, case):
        if case["kind"] == "vision":
            return "vision %dpx x%d" % (case["image"], case.get("concurrency", 1))
        return "%dk ctx x%d" % (case["context"] // 1024, case.get("concurrency", 1))

    def _messages(self, case, i):
        salt = "[run %s]\n" % uuid.uuid4().hex  # defeats prefix caching across requests
        if case["kind"] == "vision":
            content = [{"type": "text", "text": salt + "Describe the colors of the four quadrants of this image, "
                                                       "top-left first, then write a short story about them."},
                       {"type": "image_url", "image_url": {"url": test_image(case["image"])}}]
            return [{"role": "user", "content": content}], None
        passkey = 10000 + (case["context"] * 7 + i * 13) % 89999
        text, n = build_prompt(self.client, case["context"], seed=case["context"] + i, passkey=passkey)
        case.setdefault("measured_prompt", n)
        return [{"role": "user", "content": salt + text}], passkey

    def _run_case(self, case):
        results = []
        wall_total = 0.0
        for rep in range(self.suite["repeats"]):
            batch = [self._messages(case, rep * 100 + k) for k in range(case["concurrency"])]
            out = [None] * len(batch)

            def worker(k):
                msgs, key = batch[k]
                r = self.client.stream_chat(msgs, self.suite["output"], self._extra())
                if key is not None and r.get("ok"):
                    r["passkey"] = key
                    r["passkey_ok"] = str(key) in r["text"][:400]
                r.pop("text", None)
                out[k] = r
                self.progress["done"] += 1

            t = time.perf_counter()
            threads = [threading.Thread(target=worker, args=(k,)) for k in range(len(batch))]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            wall_total += time.perf_counter() - t
            results.extend(out)
            if self.cancelled:
                break
        return {"summary": summarize(results, wall_total)}
