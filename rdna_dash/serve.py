"""Start, stop and inspect the model server the plan describes.

Container path: write compose.yml and run `<compose> up -d` (or a plain `run` when
no compose front end exists). Host venv path: run serve_rdna.sh in its own
process group with a log file. Attach mode only watches a server you run yourself.
"""

import os
import signal
import subprocess
import time

from . import store


def _alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    stat = ""
    try:
        with open("/proc/%d/stat" % pid) as f:
            stat = f.read()
    except Exception:
        pass
    return " Z " not in stat  # zombie = exited


class ServeManager(object):
    def __init__(self):
        self.log_path = os.path.join(store.STATE_DIR, "serve.log")

    def _cap(self, argv, timeout=600):
        try:
            p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
            return p.returncode, p.stdout.decode("utf-8", "replace")
        except Exception as e:
            return 1, str(e)

    def write_compose(self, launch):
        path = launch["compose_file"]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(launch["compose_yaml"])
        for sub in ("triton", "vllm"):
            os.makedirs(os.path.join(os.path.dirname(path), "cache", sub), exist_ok=True)
        return path

    def start(self, plan):
        launch = plan["launch"]
        st = store.load_state()
        if self.status(plan)["process"] == "running":
            return {"ok": True, "message": "Already running."}
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if launch["kind"] == "container":
            self.write_compose(launch)
            # A stopped container with the same name blocks `run`; compose handles its own.
            self._cap([launch["engine"], "rm", "-f", launch["container"]], 60)
            code, out = self._cap(launch["up"])
            with open(self.log_path, "w") as f:
                f.write("$ %s\n%s\n" % (" ".join(launch["up"]), out))
            st["server"] = {"kind": "container", "engine": launch["engine"], "container": launch["container"],
                            "started": time.time(), "down": launch["down"], "logs": launch["logs"]}
            store.save_state(st)
            return {"ok": code == 0, "message": out[-2000:]}
        log = open(self.log_path, "wb")
        log.write(("$ %s\n\n" % launch["command"]).encode())
        log.flush()
        # The command cds into the checkout itself; /tmp as the starting cwd keeps stray files out of $HOME.
        proc = subprocess.Popen(["bash", "-c", launch["command"]], stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True, cwd="/tmp")
        st["server"] = {"kind": "venv", "pid": proc.pid, "started": time.time()}
        store.save_state(st)
        return {"ok": True, "message": "Started (pid %d)." % proc.pid}

    def stop(self, plan=None):
        st = store.load_state()
        srv = st.get("server") or {}
        if srv.get("kind") == "container":
            code, out = self._cap(srv.get("down") or [srv["engine"], "rm", "-f", srv["container"]], 180)
            msg = out[-1000:]
        elif srv.get("kind") == "venv" and _alive(srv.get("pid")):
            pid = srv["pid"]
            try:
                os.killpg(pid, signal.SIGTERM)
            except Exception:
                pass
            for _ in range(60):
                if not _alive(pid):
                    break
                time.sleep(0.5)
            if _alive(pid):
                try:
                    os.killpg(pid, signal.SIGKILL)
                except Exception:
                    pass
            msg = "Stopped the server process group (workers included)."
        else:
            msg = "Nothing to stop."
        st["server"] = dict(srv, stopped=time.time())
        store.save_state(st)
        return {"ok": True, "message": msg}

    def status(self, plan=None):
        srv = store.load_state().get("server") or {}
        proc = "stopped"
        if srv.get("kind") == "container":
            code, out = self._cap([srv["engine"], "inspect", "-f", "{{.State.Status}}", srv["container"]], 10)
            if code == 0:
                s = out.strip()
                proc = "running" if s == "running" else ("exited" if s in ("exited", "dead") else s or "stopped")
        elif srv.get("kind") == "venv":
            if _alive(srv.get("pid")):
                proc = "running"
            elif srv.get("started") and not srv.get("stopped"):
                proc = "exited"
        if srv.get("stopped") and srv.get("stopped") >= srv.get("started", 0) and proc != "running":
            proc = "stopped"
        return {"process": proc, "kind": srv.get("kind"), "started": srv.get("started")}

    def log(self, lines=400):
        srv = store.load_state().get("server") or {}
        if srv.get("kind") == "container":
            code, out = self._cap(srv.get("logs") or [srv["engine"], "logs", "--tail", str(lines), srv["container"]], 20)
            return out
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 512 * 1024))
                return "\n".join(f.read().decode("utf-8", "replace").splitlines()[-lines:])
        except Exception:
            return ""
