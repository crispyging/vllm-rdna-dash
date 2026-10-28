"""Read-only system profile: GPUs, ROCm, access rights, tools, host resources.

Nothing here writes to the system or needs root. Every probe is best effort and
degrades to "unknown" instead of raising.
"""

import glob
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import urllib.request

# PCI device id -> (marketing name, LLVM target, nominal VRAM GB).
# Navi 21 boards report gfx1030; the rest are RDNA2 cousins or RDNA3.
AMD_DEVICES = {
    "73a1": ("Radeon PRO V620", "gfx1030", 32),
    "73a3": ("Radeon PRO W6800", "gfx1030", 32),
    "73a5": ("Radeon RX 6950 XT", "gfx1030", 16),
    "73af": ("Radeon RX 6900 XT", "gfx1030", 16),
    "73bf": ("Radeon RX 6800 / 6800 XT / 6900 XT", "gfx1030", 16),
    "73ab": ("Radeon PRO W6800X / W6800X Duo", "gfx1030", 32),
    "73df": ("Radeon RX 6700 XT / 6750 XT / 6700", "gfx1031", 12),
    "73ff": ("Radeon RX 6600 / 6600 XT / 6650 XT", "gfx1032", 8),
    "743f": ("Radeon RX 6500 XT / 6400", "gfx1034", 4),
    "744c": ("Radeon RX 7900 XTX / XT / GRE", "gfx1100", 24),
    "7448": ("Radeon PRO W7900", "gfx1100", 48),
}

COMMON_PORTS = [8000, 8008, 8888, 18094, 18096, 18097, 18105, 18210, 18300]

# Running as the dashboard container: host facts come from mounts and the shared kernel.
IN_CONTAINER = bool(os.environ.get("RDNA_DASH_IN_CONTAINER")) or os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")
HOST_ROOT = os.environ.get("RDNA_DASH_HOST_ROOT", "/host")
FAMILY_PM = {"debian": "apt", "fedora": "dnf", "rhel": "dnf", "suse": "zypper", "arch": "pacman"}


def _run(cmd, timeout=6):
    """Run a command, return stdout text or '' on any failure."""
    try:
        out = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=timeout, check=False,
        )
        return out.stdout.decode("utf-8", "replace")
    except Exception:
        return ""


def _read(path, default=""):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except Exception:
        return default


def _int(path):
    try:
        return int(_read(path))
    except Exception:
        return None


def os_info():
    rel = {}
    host_rel = os.path.join(HOST_ROOT, "etc", "os-release")
    for line in _read(host_rel if IN_CONTAINER and os.path.exists(host_rel) else "/etc/os-release").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            rel[k] = v.strip('"')
    virt = "unknown" if IN_CONTAINER else (_run(["systemd-detect-virt"]).strip() or "unknown")
    if virt in ("unknown", "docker", "podman", "container-other"):
        # The CPU's hypervisor flag is visible from containers too.
        flags = next((l for l in _read("/proc/cpuinfo").splitlines() if l.startswith("flags")), "")
        vendor = _read("/sys/class/dmi/id/sys_vendor").lower()
        virt = ("kvm" if "qemu" in vendor or "kvm" in _read("/sys/class/dmi/id/product_name").lower() else "vm")             if " hypervisor" in flags else "none"
    cmdline = _read("/proc/cmdline")
    sb = _run(["mokutil", "--sb-state"]).strip().lower()
    selinux = _run(["getenforce"]).strip().lower() or "absent"
    pm = next((m for m in ("apt", "dnf", "zypper", "pacman") if shutil.which(m)), "unknown")
    like = (rel.get("ID", "") + " " + rel.get("ID_LIKE", "")).split()
    family = next((f for f in ("debian", "ubuntu", "fedora", "rhel", "suse", "arch") if f in like), rel.get("ID", ""))
    if IN_CONTAINER:
        pm = FAMILY_PM.get("debian" if family == "ubuntu" else family, "unknown")
        selinux = "enforcing" if _read("/sys/fs/selinux/enforce") == "1" else selinux
    return {
        "name": rel.get("PRETTY_NAME") or platform.platform(),
        "id": rel.get("ID", ""),
        "family": "debian" if family == "ubuntu" else family,
        "version_id": rel.get("VERSION_ID", ""),
        "package_manager": pm,
        "selinux": selinux,
        "kernel": platform.release(),
        "virt": virt,
        "iommu_pt": "iommu=pt" in cmdline,
        "ras_disabled_param": bool(re.search(r"\bamdgpu\.ras_enable=0\b", cmdline)),
        "secure_boot": "enabled" if "enabled" in sb else ("disabled" if "disabled" in sb else "unknown"),
    }


def cpu_info():
    model = ""
    vendor = ""
    for line in _read("/proc/cpuinfo").splitlines():
        if not model and line.startswith("model name"):
            model = line.split(":", 1)[1].strip()
        if not vendor and line.startswith("vendor_id"):
            vendor = line.split(":", 1)[1].strip()
        if model and vendor:
            break
    v = "amd" if "AMD" in vendor else ("intel" if "Intel" in vendor else (vendor.lower() or "unknown"))
    return {"model": model or platform.processor(), "vendor": v, "threads": os.cpu_count() or 0}


def mem_info():
    info = {}
    for line in _read("/proc/meminfo").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            info[parts[0].rstrip(":")] = int(parts[1])
    total = info.get("MemTotal", 0) / 1024 / 1024
    avail = info.get("MemAvailable", 0) / 1024 / 1024
    return {"total_gb": round(total, 1), "available_gb": round(avail, 1)}


def disk_free_gb(path):
    try:
        p = path
        while p and not os.path.exists(p):
            p = os.path.dirname(p)
        u = shutil.disk_usage(p or "/")
        return round(u.free / 1024 ** 3, 1)
    except Exception:
        return None


def _hwmon(dev):
    hw = glob.glob(os.path.join(dev, "hwmon", "hwmon*"))
    return hw[0] if hw else None


def gpus():
    """Discrete and integrated AMD GPUs from sysfs, ordered by PCI address."""
    found = {}
    for card in sorted(glob.glob("/sys/class/drm/card[0-9]*")):
        if "-" in os.path.basename(card):
            continue  # connectors like card1-DP-1
        dev = os.path.realpath(os.path.join(card, "device"))
        if _read(os.path.join(dev, "vendor")).lower() != "0x1002":
            continue
        pci = os.path.basename(dev)
        if pci in found:
            continue
        did = _read(os.path.join(dev, "device")).lower().replace("0x", "")
        sub = _read(os.path.join(dev, "subsystem_device")).lower().replace("0x", "")
        name, gfx, nominal = AMD_DEVICES.get(did, ("AMD GPU " + did, "unknown", 0))
        vram = _int(os.path.join(dev, "mem_info_vram_total"))
        hw = _hwmon(dev)
        # ECC error counters ("ue: N" / "ce: N"); present only when RAS/ECC is active.
        ras = _read(os.path.join(dev, "ras", "umc_err_count"))
        ecc = {k: int(v) for k, v in re.findall(r"(ue|ce):\s*(\d+)", ras)} if ras else None
        cap = _int(os.path.join(hw, "power1_cap")) if hw else None
        cap_min = _int(os.path.join(hw, "power1_cap_min")) if hw else None
        cap_max = _int(os.path.join(hw, "power1_cap_max")) if hw else None
        found[pci] = {
            "pci": pci,
            "card": os.path.basename(card),
            "device_id": did,
            "subsystem_id": sub,
            "name": name,
            "gfx": gfx,
            "vram_gb": round(vram / 1024 ** 3, 1) if vram else float(nominal),
            "nominal_vram_gb": nominal,
            "power_cap_w": round(cap / 1e6) if cap else None,
            "power_cap_min_w": round(cap_min / 1e6) if cap_min else None,
            "power_cap_max_w": round(cap_max / 1e6) if cap_max else None,
            "ecc_errors": ecc,
        }
    out = sorted(found.values(), key=lambda g: g["pci"])
    # rocminfo order follows KFD enumeration; use it to fill unknown targets.
    agents = rocminfo_agents()
    for i, g in enumerate(out):
        g["index"] = i
        g["integrated"] = (g["vram_gb"] or 0) < 4
        if g["gfx"] == "unknown" and i < len(agents):
            g["gfx"] = agents[i]
    return out


def rocminfo_agents():
    txt = _run(["rocminfo"], timeout=10)
    return re.findall(r"^\s+Name:\s+(gfx[0-9a-f]+)\s*$", txt, flags=re.M)


def _norm_ver(v):
    m = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", v or "")
    if not m:
        return ""
    return "%s.%s.%s" % (m.group(1), m.group(2), m.group(3) or "0")


def rocm_info():
    versions = {}
    for root in sorted(set(glob.glob("/opt/rocm*") + glob.glob("/opt/rocm/core-*"))):
        v = _read(os.path.join(root, ".info", "version"))
        if not v:
            m = re.search(r"(\d+\.\d+(?:\.\d+)?)", os.path.basename(root))
            v = m.group(1) if m else ""
        if v:
            versions[root] = _norm_ver(v)
    hipcc = shutil.which("hipcc") or next(
        (p for p in (os.path.join(r, "bin", "hipcc") for r in versions) if os.path.exists(p)), None)
    # Distro-packaged ROCm (Fedora, Arch, openSUSE) lives under /usr, not /opt/rocm.
    if hipcc and not any(hipcc.startswith(r + "/") for r in versions):
        root = os.path.dirname(os.path.dirname(os.path.realpath(hipcc)))
        v = _norm_ver(_run([hipcc.replace("hipcc", "hipconfig"), "--version"]) or _run(["hipconfig", "--version"]))
        if v:
            versions[root] = v
    primary = versions.get("/opt/rocm") or (sorted(versions.values())[-1] if versions else "")
    sdk_root = next((r for r in versions if "/core-" in r), None)
    home = os.environ.get("ROCM_PATH") or os.environ.get("ROCM_HOME")
    return {
        "installs": versions,
        "primary": primary,
        "sdk_root": sdk_root,
        "home_env": home,
        "hipcc": hipcc,
        "rocminfo": bool(shutil.which("rocminfo")),
        "agents": rocminfo_agents(),
    }


def device_gids():
    """Numeric groups that own the GPU device nodes. Containers need these numbers:
    a name like "render" is looked up inside the image and may not exist or match."""
    out = {}
    for pattern, role in (("/dev/kfd", "render"), ("/dev/dri/renderD*", "render"), ("/dev/dri/card*", "video")):
        for p in glob.glob(pattern):
            try:
                out.setdefault(role, set()).add(os.stat(p).st_gid)
            except OSError:
                pass
    return {k: sorted(v) for k, v in out.items()}


def access_info():
    groups = _run(["id", "-nG"]).split()
    kfd = os.path.exists("/dev/kfd")
    gids = device_gids()
    return {
        "device_gids": gids,
        "process_gids": sorted(set(os.getgroups() + [os.getgid()])) if hasattr(os, "getgroups") else [],
        "user": os.environ.get("USER", ""),
        "groups": groups,
        "in_video": "video" in groups,
        "in_render": "render" in groups,
        "kfd_exists": kfd,
        "kfd_rw": kfd and os.access("/dev/kfd", os.R_OK | os.W_OK),
        "render_nodes": len(glob.glob("/dev/dri/renderD*")),
        "amdgpu_loaded": os.path.isdir("/sys/module/amdgpu"),
    }


def _compose_cmd(engine):
    """The compose front end that works for this engine, as an argv prefix."""
    candidates = [[engine, "compose"]]
    candidates.append(["docker-compose"] if engine == "docker" else ["podman-compose"])
    for c in candidates:
        if shutil.which(c[0]) and _run(c + ["version"]).strip():
            return c
    return None


def container_info():
    """Docker or Podman, whichever this user can actually reach. Docker wins a tie
    because the wiki's commands are written for it."""
    engines = []
    for engine in ("docker", "podman"):
        exe = shutil.which(engine)
        if not exe:
            continue
        ver = _run([engine, "version", "--format", "{{.Server.Version}}" if engine == "docker" else "{{.Version}}"]).strip()
        rootless = False
        if engine == "podman" and ver:
            rootless = _run(["podman", "info", "--format", "{{.Host.Security.Rootless}}"]).strip() == "true"
        images = []
        if ver:
            images = [l for l in _run([engine, "images", "--format", "{{.Repository}}:{{.Tag}}"]).splitlines()
                      if "vllm-rdna" in l]
        engines.append({"engine": engine, "usable": bool(ver), "version": ver, "rootless": rootless,
                        "compose": _compose_cmd(engine) if ver else None, "vllm_rdna_images": images})
    usable = [e for e in engines if e["usable"]]
    best = (usable or engines or [None])[0]
    # In the dashboard container the docker CLI may be talking to a Podman socket.
    flavor = os.environ.get("RDNA_DASH_ENGINE", "")
    if best and flavor.startswith("podman"):
        best["flavor"] = "podman"
        best["rootless"] = flavor == "podman-rootless"
    return {
        "installed": bool(engines),
        "usable": bool(usable),
        "engine": best["engine"] if best else None,
        "flavor": (best.get("flavor") or best["engine"]) if best else None,
        "rootless": bool(best and best["rootless"]),
        "compose": best["compose"] if best else None,
        "vllm_rdna_images": best["vllm_rdna_images"] if best else [],
        "engines": engines,
    }


def tools_info():
    return {
        "python312": shutil.which("python3.12"),
        "uv": shutil.which("uv") or (os.path.expanduser("~/.local/bin/uv")
                                     if os.path.exists(os.path.expanduser("~/.local/bin/uv")) else None),
        "git": shutil.which("git"),
        "curl": shutil.which("curl"),
    }


def probe_servers(ports=None, timeout=0.6):
    """Find OpenAI-compatible servers already listening on localhost."""
    out = []
    for port in ports or COMMON_PORTS:
        s = socket.socket()
        s.settimeout(0.2)
        try:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                continue
        finally:
            s.close()
        entry = {"port": port, "models": []}
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/v1/models" % port, timeout=timeout) as r:
                data = json.loads(r.read().decode())
                entry["models"] = [m.get("id") for m in data.get("data", [])]
        except Exception:
            pass
        out.append(entry)
    return out


def collect(data_dir=None):
    home = os.path.expanduser("~")
    return {
        "schema": 1,
        "hostname": os.environ.get("RDNA_DASH_LABEL") or socket.gethostname(),
        "runtime": {"container": IN_CONTAINER},
        "os": os_info(),
        "cpu": cpu_info(),
        "memory": mem_info(),
        "disk": {"home_free_gb": disk_free_gb(home),
                 "data_free_gb": disk_free_gb(data_dir) if data_dir else None},
        "gpus": gpus(),
        "rocm": rocm_info(),
        "access": access_info(),
        "container": container_info(),
        "tools": tools_info(),
        "env": dict(
            {k: os.environ[k] for k in ("HSA_OVERRIDE_GFX_VERSION", "HIP_VISIBLE_DEVICES", "VIDEO_GID", "RENDER_GID")
             if os.environ.get(k)},
            hf_token_set=bool(os.environ.get("HF_TOKEN"))),
        "servers": probe_servers(),
    }


def redact(profile):
    """Profile safe to paste in a public issue: no hostname, user, or tokens."""
    p = json.loads(json.dumps(profile))
    p["hostname"] = "<redacted>"
    p.get("access", {})["user"] = "<redacted>"
    return p
