"""Config and state on disk. Config is what the user chose; state is what happened.

Both live under the user's XDG dirs, never in the repo, so a box name, a fleet
URL or a token never ends up in git.
"""

import json
import os
import secrets
import threading

_lock = threading.Lock()


def _xdg(var, default):
    return os.path.join(os.environ.get(var) or os.path.expanduser(default), "vllm-rdna-dash")


CONFIG_DIR = _xdg("XDG_CONFIG_HOME", "~/.config")
STATE_DIR = _xdg("XDG_STATE_HOME", "~/.local/state")


def set_dirs(config_dir, state_dir):
    global CONFIG_DIR, STATE_DIR
    CONFIG_DIR, STATE_DIR = config_dir, state_dir


def _load(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save(path, data, private=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    if private:
        os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def config_path():
    return os.path.join(CONFIG_DIR, "config.json")


def state_path():
    return os.path.join(STATE_DIR, "state.json")


def load_config():
    with _lock:
        cfg = _load(config_path(), {})
        changed = False
        env_tok = os.environ.get("RDNA_DASH_TOKEN", "").strip()
        if env_tok and cfg.get("token") != env_tok:
            cfg["token"] = env_tok
            changed = True
        if not cfg.get("token"):
            cfg["token"] = secrets.token_urlsafe(24)
            changed = True
        cfg.setdefault("prefs", {})
        cfg.setdefault("fleet", [])
        cfg.setdefault("attach_url", None)
        cfg.setdefault("label", None)
        if changed:
            _save(config_path(), cfg, private=True)
        return cfg


def save_config(cfg):
    with _lock:
        _save(config_path(), cfg, private=True)


def load_state():
    with _lock:
        st = _load(state_path(), {})
        st.setdefault("steps", {})
        st.setdefault("server", {})
        return st


def save_state(st):
    with _lock:
        _save(state_path(), st)


def update_state(fn):
    st = load_state()
    fn(st)
    save_state(st)
    return st


def jobs_dir():
    d = os.path.join(STATE_DIR, "jobs")
    os.makedirs(d, exist_ok=True)
    return d
