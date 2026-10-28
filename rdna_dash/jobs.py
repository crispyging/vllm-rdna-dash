"""Run install steps as background shell jobs with a log file each.

Only one job runs at a time. Scripts come from the tuning plan (never from the
browser), run as the current user, and never use sudo.
"""

import os
import signal
import subprocess
import threading
import time
import uuid

from . import store


class Job(object):
    def __init__(self, step, script):
        self.id = uuid.uuid4().hex[:10]
        self.step = step
        self.script = script
        self.status = "running"
        self.exit_code = None
        self.started = time.time()
        self.ended = None
        self.log_path = os.path.join(store.jobs_dir(), "%s-%s.log" % (step, self.id))
        self.proc = None

    def info(self):
        return {"id": self.id, "step": self.step, "status": self.status, "exit_code": self.exit_code,
                "started": self.started, "ended": self.ended}

    def read(self, offset=0, limit=256 * 1024):
        try:
            with open(self.log_path, "rb") as f:
                f.seek(offset)
                data = f.read(limit)
            return data.decode("utf-8", "replace"), offset + len(data)
        except Exception:
            return "", offset


class Runner(object):
    def __init__(self, on_done=None):
        self.jobs = {}
        self.current = None
        self.on_done = on_done
        self.lock = threading.Lock()

    def busy(self):
        return self.current is not None and self.current.status == "running"

    def start(self, step, script, env=None):
        with self.lock:
            if self.busy():
                raise RuntimeError("Another step is still running: %s" % self.current.step)
            job = Job(step, script)
            self.jobs[job.id] = job
            self.current = job
        self._spawn(job, env)
        return job

    def _spawn(self, job, env):
        e = dict(os.environ)
        e["PATH"] = os.path.expanduser("~/.local/bin") + os.pathsep + e.get("PATH", "")
        e.update(env or {})
        log = open(job.log_path, "wb")
        header = "$ %s\n\n" % job.script.replace("\n", "\n$ ")
        log.write(header.encode())
        log.flush()
        job.proc = subprocess.Popen(["bash", "-c", "set -eo pipefail\n" + job.script],
                                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    env=e, start_new_session=True)
        threading.Thread(target=self._wait, args=(job, log), daemon=True).start()

    def _wait(self, job, log):
        code = job.proc.wait()
        log.close()
        self._finish(job, code)

    def _finish(self, job, code):
        job.exit_code = code
        job.ended = time.time()
        if job.status == "running":
            job.status = "ok" if code == 0 else "failed"
        if self.on_done:
            try:
                self.on_done(job)
            except Exception:
                pass

    def cancel(self, job_id):
        job = self.jobs.get(job_id)
        if not job or job.status != "running":
            return False
        job.status = "cancelled"
        try:
            os.killpg(job.proc.pid, signal.SIGTERM)
        except Exception:
            pass
        return True


class DemoRunner(Runner):
    """Pretends to run scripts so the UI can be tried on any machine."""

    def _spawn(self, job, env):
        threading.Thread(target=self._fake, args=(job,), daemon=True).start()

    def _fake(self, job):
        lines = ["$ " + l for l in job.script.splitlines()[:6]] + [""]
        lines += {
            "build": ["Building editable for vllm (pyproject.toml) ..."] + ["[%d/48] hipcc csrc/rocm/rdna/... .hip" % i for i in range(1, 49, 4)]
                     + ["Successfully installed vllm-0.31.0+rdna"],
            "torch": ["Resolved 24 packages", "Installed torch==2.12.0+rocm7.14.0", "2.12.0+rocm7.14.0 True"],
            "verify": ["2.12.0+rocm7.14.0 AMD Radeon PRO V620 vllm 0.31.0"],
            "p2p": ["GPU0 Y Y Y Y", "GPU1 Y Y Y Y", "GPU2 Y Y Y Y", "GPU3 Y Y Y Y",
                    'P2P_RESULT {"devices": 4, "matrix": [], "all": true}'],
        }.get(job.step, ["working ...", "done"])
        with open(job.log_path, "w", encoding="utf-8") as f:
            for l in lines:
                if job.status != "running":
                    break
                f.write(l + "\n")
                f.flush()
                time.sleep(0.35)
        self._finish(job, 0 if job.status == "running" else 143)

    def cancel(self, job_id):
        job = self.jobs.get(job_id)
        if job and job.status == "running":
            job.status = "cancelled"
            return True
        return False
