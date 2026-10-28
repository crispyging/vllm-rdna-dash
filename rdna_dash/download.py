"""Download a Hugging Face repo into a plain folder, with a speed limit you can change live.

  python -m rdna_dash.download REPO DEST [--revision REV] [--include GLOB ...] [--limit-file PATH]

Stdlib only. Resumes partial files (HTTP Range), skips files already complete, and
re-reads the limit file every second (MB/s, 0 or empty = unlimited), so the dashboard
can change the speed while a download runs. Prints a PROGRESS {json} line every
couple of seconds for the UI, plus a readable line for people reading the log.
"""

import argparse
import fnmatch
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

CHUNK = 1 << 20


def endpoint():
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def _request(url, token=None, headers=None):
    req = urllib.request.Request(url, headers=dict({"User-Agent": "vllm-rdna-dash"}, **(headers or {})))
    if token:
        # Not forwarded on redirect: the CDN URL is already signed and must not see the token.
        req.add_unredirected_header("Authorization", "Bearer " + token)
    return req


def list_files(repo, revision, token=None):
    url = "%s/api/models/%s/revision/%s?blobs=true" % (endpoint(), repo, urllib.parse.quote(revision, safe=""))
    with urllib.request.urlopen(_request(url, token), timeout=60) as r:
        info = json.loads(r.read().decode())
    out = []
    for s in info.get("siblings", []):
        size = s.get("size")
        if size is None and s.get("lfs"):
            size = s["lfs"].get("size")
        out.append({"path": s["rfilename"], "size": size})
    return out


def select(files, include):
    if not include:
        return files
    return [f for f in files if any(fnmatch.fnmatch(f["path"], g) for g in include)]


class Limiter(object):
    """Token bucket in bytes/s, re-read from a file once a second."""

    def __init__(self, limit_file=None, mbps=0.0):
        self.file = limit_file
        self.mbps = mbps
        self.checked = 0.0
        self.window_start = time.monotonic()
        self.window_bytes = 0

    def current(self):
        now = time.monotonic()
        if self.file and now - self.checked >= 1.0:
            self.checked = now
            try:
                with open(self.file) as f:
                    txt = f.read().strip()
                self.mbps = float(txt) if txt else 0.0
            except (OSError, ValueError):
                pass
        return self.mbps

    def take(self, n):
        mbps = self.current()
        now = time.monotonic()
        if now - self.window_start >= 1.0:
            self.window_start, self.window_bytes = now, 0
        self.window_bytes += n
        if mbps > 0:
            allowed = mbps * 1e6 * max(now - self.window_start, 1e-3)
            if self.window_bytes > allowed:
                time.sleep((self.window_bytes - allowed) / (mbps * 1e6))


class Progress(object):
    def __init__(self, files):
        self.total = sum(f["size"] or 0 for f in files)
        self.done = 0
        self.n = len(files)
        self.last = 0.0
        self.t0 = time.monotonic()
        self.recent = []

    def tick(self, i, name, file_done, file_total, limiter, force=False):
        now = time.monotonic()
        self.recent.append((now, self.done))
        self.recent = [x for x in self.recent if now - x[0] <= 5]
        if not force and now - self.last < 2:
            return
        self.last = now
        rate = 0.0
        if len(self.recent) > 1 and self.recent[-1][0] > self.recent[0][0]:
            rate = (self.recent[-1][1] - self.recent[0][1]) / (self.recent[-1][0] - self.recent[0][0])
        eta = (self.total - self.done) / rate if rate > 0 else None
        p = {"file": i + 1, "files": self.n, "name": name, "file_done": file_done, "file_total": file_total,
             "done": self.done, "total": self.total, "rate": rate, "eta": eta, "limit_mbps": limiter.mbps}
        print("PROGRESS " + json.dumps(p), flush=True)
        print("  [%d/%d] %s  %s / %s  %.1f MB/s%s%s" % (
            i + 1, self.n, name, gb(self.done), gb(self.total), rate / 1e6,
            "  ETA %s" % hms(eta) if eta else "", "  (limit %g MB/s)" % limiter.mbps if limiter.mbps else ""), flush=True)


def gb(n):
    return "%.2f GB" % (n / 1e9) if n >= 1e8 else "%.1f MB" % (n / 1e6)


def hms(s):
    s = int(s)
    return "%dh%02dm" % (s // 3600, s % 3600 // 60) if s >= 3600 else "%dm%02ds" % (s // 60, s % 60)


def fetch(repo, revision, f, dest, token, limiter, progress, i, retries=8):
    path = os.path.join(dest, *f["path"].split("/"))
    size = f["size"]
    if size is not None and os.path.isfile(path) and os.path.getsize(path) == size:
        progress.done += size
        return "skip"
    os.makedirs(os.path.dirname(path) or dest, exist_ok=True)
    part = path + ".part"
    url = "%s/%s/resolve/%s/%s" % (endpoint(), repo, urllib.parse.quote(revision, safe=""),
                                   urllib.parse.quote(f["path"]))
    base_done = progress.done
    for attempt in range(retries):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if size is not None and have > size:
            os.remove(part)
            have = 0
        progress.done = base_done + have
        headers = {"Range": "bytes=%d-" % have} if have else {}
        try:
            with urllib.request.urlopen(_request(url, token, headers), timeout=60) as r:
                if have and r.status != 206:
                    have = 0  # server ignored the range: start over
                    progress.done = base_done
                with open(part, "ab" if have else "wb") as out:
                    while True:
                        buf = r.read(CHUNK)
                        if not buf:
                            break
                        out.write(buf)
                        have += len(buf)
                        progress.done += len(buf)
                        limiter.take(len(buf))
                        progress.tick(i, f["path"], have, size, limiter)
            if size is None or have == size:
                os.replace(part, path)
                return "ok"
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                raise SystemExit("%s: HTTP %d. %s" % (f["path"], e.code,
                                 "Set HF_TOKEN for gated or private repos." if e.code in (401, 403) else "Not found."))
            if e.code == 416 and size is not None and have == size:
                os.replace(part, path)
                return "ok"
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            print("  retrying %s after: %s" % (f["path"], e), flush=True)
        time.sleep(min(60, 2 ** attempt))
    raise SystemExit("gave up on %s after %d tries" % (f["path"], retries))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rdna_dash.download")
    ap.add_argument("repo")
    ap.add_argument("dest")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--include", action="append", default=[])
    ap.add_argument("--limit-file", help="file holding the speed limit in MB/s (0 = unlimited)")
    ap.add_argument("--limit", type=float, default=0.0, help="speed limit in MB/s when no limit file is given")
    a = ap.parse_args(argv)
    token = os.environ.get("HF_TOKEN") or None
    try:
        files = select(list_files(a.repo, a.revision, token), a.include)
    except urllib.error.HTTPError as e:
        raise SystemExit("Could not list %s@%s: HTTP %d%s" % (a.repo, a.revision, e.code,
                         " (set HF_TOKEN for gated or private repos)" if e.code in (401, 403) else ""))
    if not files:
        raise SystemExit("Nothing to download from %s@%s matching %s" % (a.repo, a.revision, a.include or "*"))
    limiter = Limiter(a.limit_file, a.limit)
    limiter.current()
    progress = Progress(files)
    print("Downloading %d files (%s) from %s@%s to %s%s" % (
        len(files), gb(progress.total), a.repo, a.revision, a.dest,
        ", limit %g MB/s" % limiter.mbps if limiter.mbps else ""), flush=True)
    skipped = 0
    for i, f in enumerate(files):
        if fetch(a.repo, a.revision, f, a.dest, token, limiter, progress, i) == "skip":
            skipped += 1
    progress.tick(len(files) - 1, files[-1]["path"], 0, 0, limiter, force=True)
    print("Done: %d files%s in %s." % (len(files), " (%d already there)" % skipped if skipped else "",
                                       hms(time.monotonic() - progress.t0)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
