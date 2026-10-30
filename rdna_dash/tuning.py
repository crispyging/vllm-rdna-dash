"""Turn a system profile plus the recipe registry into a tuned plan for this box.

The rules follow the vllm-rdna README and the gfx1030 wiki. Every setting the
plan picks carries a one-line reason so a new user can see why it was chosen.
Recipes (model + layout + requirements) live in recipes/registry.json so the
community can add them without touching this file.
"""

import json
import os
import re
import shlex

WIKI = "https://blivioniag.github.io/gfx1030-wiki/"
RAW = "https://raw.githubusercontent.com/crispyging/vllm-rdna-dash/main/"


def tool_cmd(name, args):
    """Root helper command. Works without a checkout: fetches the script from the repo."""
    return "curl -fsSL %stools/%s | sudo bash -s -- %s" % (RAW, name, args)
AMD_INDEX = "https://repo.amd.com/rocm/whl-multi-arch/"
BUILTIN_RECIPES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "recipes")

COUSINS = {"gfx1031", "gfx1032", "gfx1034", "gfx1035", "gfx1036"}
PRO_CARDS = {"73a1", "73a3", "73ab"}

PKG = {
    "apt": "sudo apt install -y %s",
    "dnf": "sudo dnf install -y %s",
    "zypper": "sudo zypper install -y %s",
    "pacman": "sudo pacman -S --needed %s",
}
CONTAINER_PKGS = {
    "apt": "docker.io docker-compose-v2",
    "dnf": "podman podman-compose",
    "zypper": "docker docker-compose",
    "pacman": "docker docker-compose",
}
ROCM_HINT = {
    "debian": ["# Ubuntu/Debian: add AMD's amdgpu repo, then", "sudo amdgpu-install --usecase=rocm"],
    "fedora": ["# Fedora ships ROCm in its own repos (may be older than 7.14)",
               "sudo dnf install -y rocminfo rocm-smi rocm-hip-devel"],
    "rhel": ["# RHEL/Rocky/Alma: add AMD's amdgpu repo for your release, then", "sudo amdgpu-install --usecase=rocm"],
    "arch": ["sudo pacman -S --needed rocm-hip-sdk rocminfo"],
    "suse": ["# openSUSE/SLES: add AMD's amdgpu repo, then", "sudo amdgpu-install --usecase=rocm"],
}


def recipe_dirs(extra=None):
    """Built-in recipes, then the user's, then RDNA_DASH_RECIPES (os.pathsep-separated),
    then any extra dirs. Later dirs override earlier ones by recipe id."""
    dirs = [BUILTIN_RECIPES,
            os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"), "vllm-rdna-dash", "recipes")]
    dirs += [d for d in os.environ.get("RDNA_DASH_RECIPES", "").split(os.pathsep) if d]
    return dirs + list(extra or [])


def load_registry(dirs=None):
    reg = {"schema": 1, "recipes": [], "errors": [], "dirs": []}
    by_id = {}
    for d in dirs or recipe_dirs():
        d = os.path.expanduser(d)
        if not os.path.isdir(d):
            continue
        reg["dirs"].append(d)
        for name in sorted(os.listdir(d)):
            if not name.endswith(".json"):
                continue
            path = os.path.join(d, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                reg["errors"].append("%s: %s" % (path, e))
                continue
            if name == "_defaults.json":
                reg.update({k: v for k, v in data.items() if k in ("image", "fork")})
                continue
            problem = validate_recipe(data)
            if problem:
                reg["errors"].append("%s: %s" % (path, problem))
                continue
            data["_source"] = path
            by_id[data["id"]] = data
    reg["recipes"] = list(by_id.values())
    return reg


def validate_recipe(r):
    for key in ("id", "label", "model", "requires"):
        if key not in r:
            return "missing '%s'" % key
    if "repo" not in r["model"] or "dir" not in r["model"]:
        return "model needs repo and dir"
    if "container" not in r and "venv" not in r:
        return "needs a container or venv section"
    if "venv" in r and "recipe" not in r["venv"] and "args" not in r["venv"]:
        return "venv needs a launcher recipe name or its own args"
    return None


def q(s):
    return shlex.quote(str(s))


def pkg_install(pm, pkgs):
    return (PKG.get(pm) or "# install with your package manager: %s") % pkgs


def _ver_tuple(v):
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", v or "")
    return tuple(int(x) for x in m.groups()) if m else None


def bad_rccl_rocm(v):
    """ROCm 7.2.1 through 7.13.x: the multi-GPU RCCL bug range from the wiki."""
    t = _ver_tuple(v)
    return bool(t) and (7, 2, 1) <= t < (7, 14, 0)


def check(cid, level, title, detail, fix=None, link=None):
    if link and not link.startswith("http"):
        link = WIKI + link
    return {"id": cid, "level": level, "title": title, "detail": detail, "fix": fix or [], "link": link}


def settle_port_check(plan, attach_url=None, own_server=False):
    """The busy-port warning is only about a server the dashboard neither runs nor watches."""
    for c in list(plan["checks"]):
        if c["id"] != "port":
            continue
        if own_server:
            plan["checks"].remove(c)
        elif attach_url:
            c.pop("attach", None)
            c.update(level="info", title="Watching the server at %s" % attach_url,
                     detail="The dashboard reads this server and never starts or stops it.")
    if plan["verdict"] == "attention" and not any(c["level"] == "warn" for c in plan["checks"]):
        plan["verdict"] = "ready"
    return plan


# Environment that would change what runs, not how vLLM tunes itself.
UNSAFE_ENV = re.compile(r"^(LD_|PATH$|BASH_ENV$|ENV$|PYTHONPATH$|PYTHONSTARTUP$|IFS$)")


def launch_extras(p, checks):
    """The server flags and environment the user added in Change."""
    args = []
    if p.get("kv_offload_gb"):
        args += ["--kv-offloading-size", str(p["kv_offload_gb"]), "--kv-offloading-backend", "native"]
    try:
        args += shlex.split(p.get("extra_args") or "")
    except ValueError as e:
        checks.append(check("extra", "warn", "Extra vLLM flags were ignored", "They do not parse: %s." % e))
    env = {}
    for line in (p.get("extra_env") or "").splitlines():
        line = line.strip()
        if not line:
            continue
        k, eq, v = line.partition("=")
        k = k.strip()
        if not eq or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", k) or UNSAFE_ENV.match(k):
            checks.append(check("extra", "warn", "Ignored the setting %s" % k[:60],
                                "Use KEY=VALUE, one per line. Loader and shell variables such as LD_PRELOAD are not allowed."))
            continue
        env[k] = v.strip()
    return args, env


def gpu_groups(profile):
    """Numeric host GIDs a GPU container needs. Never names: a name like "render" is looked
    up in the image's /etc/group, where it may not exist or may carry another number.

    Order: owners of /dev/kfd and /dev/dri as seen here, then VIDEO_GID/RENDER_GID from the
    environment (the dashboard container gets them from .env), then the host's own group
    database on a host install. Names are the last resort for a box with no GPU devices."""
    gids = profile.get("access", {}).get("device_gids") or {}
    nums = sorted({g for v in gids.values() for g in v if g != 0})
    if nums:
        return [str(g) for g in nums]
    env = [os.environ.get(k, "").strip() for k in ("VIDEO_GID", "RENDER_GID")]
    env = [v for v in env if v.isdigit()]
    if env:
        return sorted(set(env), key=int)
    if not profile.get("runtime", {}).get("container"):
        try:
            import grp
            return sorted({str(grp.getgrnam(n).gr_gid) for n in ("video", "render")
                           if n in {g.gr_name for g in grp.getgrall()}}, key=int) or ["video", "render"]
        except ImportError:
            pass
    return ["video", "render"]


def setting(key, value, why):
    return {"key": key, "value": value, "why": why}


def pick_gpus(profile, wanted=None):
    """Choose the homogeneous, gfx1030-compatible group to serve on."""
    discrete = [g for g in profile.get("gpus", []) if not g.get("integrated")]
    if wanted:
        chosen = [g for g in discrete if g["index"] in wanted]
        if chosen:
            return chosen, discrete
    groups = {}
    for g in discrete:
        groups.setdefault((g["device_id"], g["gfx"]), []).append(g)
    if not groups:
        return [], discrete

    def rank(item):
        (_, gfx), gs = item
        return (gfx == "gfx1030", len(gs), max(x["vram_gb"] or 0 for x in gs))

    return sorted(groups.items(), key=rank, reverse=True)[0][1], discrete


def fit(recipe, gfx, n, vram, ram):
    """Cards this recipe would use on this box, or (None, reason)."""
    req = recipe.get("requires", {})
    if req.get("gfx") and gfx not in req["gfx"]:
        return None, "needs %s" % "/".join(req["gfx"])
    usable = [c for c in req.get("cards", [1]) if c <= n]
    if not usable:
        return None, "needs %s cards" % " or ".join(str(c) for c in req.get("cards", [1]))
    if vram < req.get("min_vram_per_card_gb", 0):
        return None, "needs %d GB per card" % req["min_vram_per_card_gb"]
    for c in sorted(usable, reverse=True):
        if c * vram >= req.get("min_total_vram_gb", 0):
            if ram < req.get("min_ram_gb", 0):
                return None, "needs %d GB host RAM" % req["min_ram_gb"]
            return c, None
    return None, "needs %d GB VRAM in total" % req["min_total_vram_gb"]


def ecc_status(profile, chosen):
    """Where this box is in the two-reboot ECC-off flow, and whether it is safe."""
    pro = [g for g in chosen if g.get("device_id") in PRO_CARDS]
    if not pro:
        return {"state": "n/a"}
    param = profile.get("os", {}).get("ras_disabled_param")
    errs = [(g["index"], g["ecc_errors"]) for g in pro if g.get("ecc_errors") and
            (g["ecc_errors"].get("ce", 0) or g["ecc_errors"].get("ue", 0))]
    hidden = [g for g in pro if 0 < (g["vram_gb"] or 0) < 31]
    cmd = "%s"
    if errs:
        detail = ("Card%s %s reported ECC errors (%s). Keep ECC on: the community advice is to stress-test the "
                  "card (Vulkan memtest, 5-8 minutes per GPU) and replace it if errors continue."
                  % ("s" if len(errs) > 1 else "", ", ".join("#%d" % i for i, _ in errs),
                     "; ".join("#%d ce=%d ue=%d" % (i, e.get("ce", 0), e.get("ue", 0)) for i, e in errs)))
        state, c = "errors", check("ecc", "warn", "ECC errors on a card: leave ECC on", detail,
                                   ["sudo dmesg | grep -iE 'ecc|ras|umc'"], "tuning/ecc.html")
    elif not hidden:
        state, c = "off", check("ecc", "ok", "ECC is off", "Full 32 GB visible on each Pro card.",
                                [tool_cmd("ecc.sh", "on") + "   # to turn it back on (then reboot twice)"], "tuning/ecc.html")
    elif param:
        state, c = "pending", check("ecc", "info", "ECC off is half done: reboot once more",
                                    "amdgpu.ras_enable=0 is set but the cards still show %.1f GB. The firmware "
                                    "applies it on the second boot." % hidden[0]["vram_gb"],
                                    ["sudo reboot"], "tuning/ecc.html")
    else:
        state, c = "on", check("ecc", "rec", "Turn ECC off: ~2 GB more per card",
                               "%.1f GB visible of 32 GB. With ECC off each card gets ~2 GB more KV cache, which the "
                               "community runs by default. No ECC errors are reported right now; stress-test first if the "
                               "cards are second-hand." % hidden[0]["vram_gb"],
                               [tool_cmd("ecc.sh", "off"), "sudo reboot   # first reboot arms it", "sudo reboot   # second reboot applies it"],
                               "tuning/ecc.html")
    c["action"] = "ecc"
    return {"state": state, "check": c}


POWER_TARGET_W = 180


def power_status(chosen):
    """Recommend the community 180 W cap on V620s (stock is 250 W)."""
    v620 = [g for g in chosen if g.get("device_id") == "73a1" and g.get("power_cap_w")]
    if not v620:
        return {"state": "n/a"}
    high = [g for g in v620 if g["power_cap_w"] > POWER_TARGET_W + 5]
    if not high:
        return {"state": "ok", "check": check("power", "ok", "Power capped at %d W" % max(g["power_cap_w"] for g in v620),
                                              "Cooler, quieter cards with little speed loss on memory-bound inference.",
                                              [tool_cmd("power.sh", "status")], "tuning/power.html")}
    floor = max((g.get("power_cap_min_w") or 0) for g in v620)
    cmd = [tool_cmd("power.sh", "%d --persist") % POWER_TARGET_W + "   # applies now and at every boot"]
    detail = ("Cards are at %d W. Inference is memory-bound, so the community caps V620s at %d W: much cooler "
              "and quieter for little speed loss." % (high[0]["power_cap_w"], POWER_TARGET_W))
    if floor > POWER_TARGET_W:
        detail += (" Your driver reports a %d W minimum (the V620 VBIOS floor), so the cap needs the v620_toolbox "
                   "powerfix first. Some older kernels accept the write anyway; the script tries and tells you." % floor)
        cmd = ["git clone https://github.com/blivioniag/v620_toolbox   # then follow powertuning/README"] + cmd
    return {"state": "high", "floor": floor,
            "check": check("power", "rec", "Cap power at %d W" % POWER_TARGET_W, detail, cmd, "tuning/power.html")}


def default_prefs():
    return {
        "recipe": "auto",        # registry id, or auto
        "path": "auto",          # auto | container | venv
        "gpus": None,            # sysfs indices, or None for auto
        "root": os.environ.get("RDNA_DASH_ROOT") or os.path.expanduser("~/vllm-rdna"),
        "port": None,
        "max_len": None,
        "lan": None,             # share the model server on the network (None = yes in a container install)
        "p2p": "auto",           # auto | on | off: whether vLLM uses GPU peer-to-peer
        "p2p_probe": None,       # result of the P2P probe or bandwidth test
        "download_mbps": 0,      # model download speed limit in MB/s, 0 = unlimited
        "ple": "auto",           # auto | bf16 | int4 | int4-disk
        "model_path": None,      # a checkpoint folder already on this box: serve it, download nothing
        "kv_offload_gb": None,   # GB of KV cache vLLM may keep in CPU memory
        "extra_args": None,      # more vllm serve flags, as one line
        "extra_env": None,       # more environment for the server, KEY=VALUE per line
    }


def path_blockers(path, profile):
    """Why a path cannot work on this box yet (empty list = feasible)."""
    rocm, cont, tools = profile.get("rocm", {}), profile.get("container", {}), profile.get("tools", {})
    if path == "container":
        return [] if cont.get("usable") else ["no usable Docker or Podman"]
    if profile.get("runtime", {}).get("container"):
        return ["dashboard runs in a container"]
    out = []
    if not tools.get("git"):
        out.append("git missing")
    if not (rocm.get("hipcc") or rocm.get("sdk_root")):
        out.append("no hipcc")
    return out


def build_plan(profile, prefs=None, registry=None):
    reg = registry or load_registry()
    p = default_prefs()
    p.update({k: v for k, v in (prefs or {}).items() if v is not None})
    checks, settings, alts = [], [], []

    os_i, cpu, mem = profile.get("os", {}), profile.get("cpu", {}), profile.get("memory", {})
    acc, rocm, cont, tools = (profile.get(k, {}) for k in ("access", "rocm", "container", "tools"))
    pm, family = os_i.get("package_manager", "unknown"), os_i.get("family", "")
    chosen, discrete = pick_gpus(profile, p.get("gpus"))
    ram = mem.get("total_gb", 0) or 0

    # ---- hardware -------------------------------------------------------------
    gfx = chosen[0]["gfx"] if chosen else "unknown"
    card = chosen[0] if chosen else {}
    n = len(chosen)
    vram = min((g["vram_gb"] or 0) for g in chosen) if chosen else 0
    if not discrete:
        checks.append(check("gpu", "fail", "No AMD GPU found",
                            "No AMD discrete GPU showed up in /sys/class/drm. Check the card is seated and powered "
                            "and that the amdgpu driver loaded.", ["lspci -nn | grep -i amd", "lsmod | grep amdgpu"],
                            "setup/getting-started.html"))
    else:
        detail = "Serving on GPU index %s." % ",".join(str(g["index"]) for g in chosen)
        if gfx in COUSINS:
            detail += " RDNA2 cousin: the plan sets HSA_OVERRIDE_GFX_VERSION=10.3.0 so gfx1030 kernels load."
        elif gfx.startswith("gfx11"):
            detail += (" RDNA3: the fork has separate gfx1100 W4A16/MoE kernels, but the recipes and TunableOp rows "
                       "are gfx1030 and nothing there is validated on RDNA3.")
        elif gfx != "gfx1030":
            detail += " Unknown target: the fork is built for gfx1030."
        checks.append(check("gpu", "ok" if gfx == "gfx1030" else "warn",
                            "%d x %s (%s, %.0f GB)" % (n, card["name"], gfx, vram), detail, link="setup/hardware.html"))
        others = [g for g in discrete if g not in chosen]
        if others:
            checks.append(check("gpu-mixed", "info", "Other GPUs present",
                                "%s will be hidden with HIP_VISIBLE_DEVICES. Do not tensor-parallel across different "
                                "card models." % ", ".join("%s (#%d)" % (g["name"], g["index"]) for g in others),
                                link="setup/hardware.html"))
        if any(g.get("integrated") for g in profile.get("gpus", [])):
            checks.append(check("igpu", "info", "Integrated GPU detected",
                                "ROCm may enumerate the iGPU. The plan pins the discrete cards with HIP_VISIBLE_DEVICES.",
                                link="setup/installing-rocm.html"))

    # ---- driver, ROCm, access ---------------------------------------------------------
    if not acc.get("amdgpu_loaded"):
        fix = ["sudo modprobe amdgpu", "lsmod | grep amdgpu"]
        if os_i.get("secure_boot") == "enabled":
            fix.insert(0, "# Secure Boot is on: a DKMS amdgpu module must be signed, or disable Secure Boot")
        checks.append(check("amdgpu", "fail", "amdgpu kernel driver not loaded",
                            "ROCm needs the amdgpu module loaded.", fix, "setup/installing-rocm.html"))
    in_ctr = profile.get("runtime", {}).get("container")
    if not acc.get("kfd_exists") and in_ctr:
        checks.append(check("kfd", "fail", "The dashboard container cannot see /dev/kfd",
                            "Either the host has no ROCm driver yet, or the dashboard's compose file does not pass the GPU "
                            "devices through. Check on the host, and keep devices: [/dev/kfd, /dev/dri] in the dashboard compose.yml.",
                            ["ls -l /dev/kfd /dev/dri   # on the host"] +
                            ROCM_HINT.get(family, ["# follow AMD's ROCm install guide for your distro"]),
                            "setup/installing-rocm.html"))
    elif not acc.get("kfd_exists"):
        checks.append(check("kfd", "fail", "/dev/kfd is missing",
                            "The ROCm compute device does not exist, so nothing can use the GPU. Install the ROCm "
                            "driver stack for your distro and reboot.",
                            ROCM_HINT.get(family, ["# follow AMD's ROCm install guide for your distro"]) + ["sudo reboot"],
                            "setup/installing-rocm.html"))
    missing = [g for g in ("video", "render") if not acc.get("in_" + g)]
    if in_ctr:
        if acc.get("kfd_exists") and acc.get("kfd_rw"):
            checks.append(check("access", "ok", "GPU access rights", "The dashboard container can open /dev/kfd and the render nodes."))
        elif acc.get("kfd_exists"):
            g = acc.get("device_gids") or {}
            vid, ren = (g.get("video") or ["?"])[0], (g.get("render") or ["?"])[0]
            checks.append(check("kfd-perm", "fail", "The dashboard container cannot open /dev/kfd",
                                "The GPU devices belong to group %s (render) and %s (video) on this host, but the container "
                                "is not in them. Put those numbers in the .env next to compose.yml, or pass them to "
                                "docker run with --group-add, and recreate the container. Rootless Podman uses "
                                "group_add: [keep-groups] instead; SELinux hosts also need security_opt label=disable." % (ren, vid),
                                ["# in the folder that holds compose.yml",
                                 "printf 'VIDEO_GID=%s\\nRENDER_GID=%s\\n' >> .env" % (vid, ren),
                                 "docker compose up -d --force-recreate"],
                                "troubleshooting/vllm.html#failed-to-infer-device-type--amdsmi_status_not_init"))
    elif missing:
        checks.append(check("groups", "fail", "Not in the %s group%s" % (" and ".join(missing), "s" if len(missing) > 1 else ""),
                            "Without both video and render, vLLM fails with `Failed to infer device type` or "
                            "AMDSMI_STATUS_NOT_INIT. Log out and back in after adding them.",
                            ['sudo usermod -aG video,render "$USER"', "# then log out and back in"],
                            "troubleshooting/vllm.html#failed-to-infer-device-type--amdsmi_status_not_init"))
    elif acc.get("kfd_exists") and not acc.get("kfd_rw"):
        checks.append(check("kfd-perm", "fail", "/dev/kfd is not readable and writable",
                            "Groups are set but this login session predates them. Log out and back in.",
                            ["ls -l /dev/kfd /dev/dri/renderD*"], "troubleshooting/general.html"))
    else:
        checks.append(check("access", "ok", "GPU access rights", "In video and render; /dev/kfd is usable."))

    rv = rocm.get("primary", "")
    if in_ctr:
        pass  # the vLLM image brings ROCm 7.14; the host's userspace is not visible from here
    elif rv:
        lvl, msg = "ok", "ROCm %s userspace at %s." % (rv, ", ".join(sorted(rocm.get("installs", {}))) or "?")
        if n > 1 and bad_rccl_rocm(rv):
            lvl = "warn"
            msg = ("ROCm %s sits in the 7.2.1-7.13 range with the multi-GPU RCCL bug (TP hangs, AllReduce init "
                   "failed). Pin 7.2.0 or 7.14.0. The container image and the 7.14 venv wheels bring their own ROCm, "
                   "so this matters most for code built against the host install." % rv)
        checks.append(check("rocm", lvl, "ROCm userspace", msg, link="setup/installing-rocm.html#multi-gpu-pin-rocm-720-or-7140"))
    else:
        checks.append(check("rocm", "warn", "No ROCm userspace found",
                            "Containers do not need it (the image ships ROCm 7.14). Building the fork on the host "
                            "needs hipcc from a ROCm 7.14 SDK.", ROCM_HINT.get(family), "setup/installing-rocm.html"))
    if rocm.get("home_env") and not os.path.isdir(rocm["home_env"]):
        checks.append(check("rocm-env", "warn", "ROCM_PATH/ROCM_HOME points nowhere",
                            "%s does not exist. Builds will look there for hipcc and headers." % rocm["home_env"]))

    # ---- recipe ---------------------------------------------------------------------------
    recipes = reg["recipes"]
    fits = []
    for r in recipes:
        cards, why = fit(r, gfx, n, vram, ram)
        paths = [x for x in ("container", "venv") if x in r]
        feasible = [x for x in paths if not path_blockers(x, profile)]
        fits.append({"recipe": r, "cards": cards, "why": why, "paths": paths, "feasible": feasible})
    fits_ok = [f for f in fits if f["cards"]]
    pick = None
    if p["recipe"] != "auto":
        pick = next((f for f in fits if f["recipe"]["id"] == p["recipe"]), None)
        if pick and not pick["cards"]:
            checks.append(check("recipe", "warn", "%s does not fit this box" % pick["recipe"]["label"],
                                "It %s. Picking the best recipe that fits instead." % pick["why"], link="choose-a-stack.html"))
            pick = None
    if not pick and fits_ok:
        pick = sorted(fits_ok, key=lambda f: (bool(f["feasible"]), f["recipe"].get("priority", 0)), reverse=True)[0]
    if not pick:
        # Nothing fits: fall back to the smallest recipe so the user still gets a working plan to adjust.
        small = min(fits, key=lambda f: (f["recipe"]["requires"].get("min_total_vram_gb", 0) or
                                         f["recipe"]["requires"].get("min_vram_per_card_gb", 0) * min(f["recipe"]["requires"].get("cards", [1])),
                                         f["recipe"].get("priority", 0)))
        pick = dict(small, cards=max(1, min(n, max(small["recipe"]["requires"].get("cards", [1])))))
        if discrete:
            checks.append(check("fit", "warn", "Nothing in the registry fits this box well",
                                "Closest is %s, which %s. On small cards the wiki points to llama.cpp or a MoE model."
                                % (small["recipe"]["label"], small["why"]), link="choose-a-stack.html"))
    r = pick["recipe"]
    if r.get("verify"):
        checks.append(check("verify", "info", "Recipe marked needs-verify",
                            "%s comes from the fork's recipe files but has not been confirmed by the wiki yet." % r["label"]))

    # ---- path -------------------------------------------------------------------------------
    path = p["path"]
    if path not in pick["paths"]:
        if path != "auto":
            checks.append(check("path", "warn", "%s has no %s path" % (r["label"], path),
                                "Using %s instead." % ("the host venv" if "venv" in pick["paths"] else "a container")))
        pref = r.get("prefer", pick["paths"][0])
        path = pref if pref in pick["feasible"] or not pick["feasible"] else pick["feasible"][0]
    if path == "venv" and profile.get("runtime", {}).get("container") and "container" in pick["feasible"]:
        checks.append(check("path", "info", "Serving in a container",
                            "This dashboard runs in a container, so it cannot build the fork on the host. Using the "
                            "container image instead."))
        path = "container"
    for b in path_blockers(path, profile):
        if b == "no usable Docker or Podman":
            if not cont.get("installed"):
                checks.append(check("container", "fail", "No container engine installed",
                                    "Install Docker or Podman, or switch this plan to the host venv.",
                                    [pkg_install(pm, CONTAINER_PKGS.get(pm, "docker")), 'sudo usermod -aG docker "$USER"  # Docker only']))
            else:
                checks.append(check("container", "fail", "%s is installed but this user cannot reach it" % cont.get("engine"),
                                    "Add yourself to the docker group (Docker) or check the Podman socket, then log back in.",
                                    ['sudo usermod -aG docker "$USER"', "# then log out and back in"]))
        elif b == "git missing":
            checks.append(check("git", "fail", "git is missing", "The fork is cloned with git.", [pkg_install(pm, "git")]))
        elif b == "dashboard runs in a container":
            checks.append(check("venv-ctr", "fail", "Host builds are not available from the dashboard container",
                                "This dashboard runs in a container, so it serves models in containers too. Switch the "
                                "install path to Container, or run rdna-dash directly on the host to build the fork."))
        elif b == "no hipcc":
            checks.append(check("hipcc", "fail", "No hipcc to build the fork",
                                "The host venv compiles the fork's HIP kernels and needs hipcc from a ROCm 7.14 SDK "
                                "(the launcher looks in /opt/rocm/core-7.14). The container path needs none of this.",
                                ROCM_HINT.get(family), "vllm/host-venv.html"))
    if path == "container" and cont.get("usable"):
        eng = cont.get("flavor") or cont.get("engine")
        detail = "%s%s." % (eng.capitalize(), " (rootless)" if cont.get("rootless") else "")
        if not cont.get("compose"):
            checks.append(check("compose", "warn", "No compose front end for %s" % eng,
                                "The plan writes vllm-compose.yml; install %s to run it, or use the plain run command shown." %
                                ("docker compose" if eng == "docker" else "podman-compose"),
                                [pkg_install(pm, "docker-compose-v2" if eng == "docker" and pm == "apt" else
                                             ("docker-compose" if eng == "docker" else "podman-compose"))]))
        else:
            detail += " Compose: %s." % " ".join(cont["compose"])
        checks.append(check("container", "ok", "Container engine ready", detail))
    if path == "venv" and rv and _ver_tuple(rv) and _ver_tuple(rv)[:2] != (7, 14):
        checks.append(check("rocm-build", "warn", "Host ROCm is %s, the fork builds against 7.14" % rv,
                            "The venv wheels are ROCm 7.14. Compiling the fork's kernels with a different hipcc can fail "
                            "or mismatch. Install the 7.14 SDK, or use the container path.", link="vllm/host-venv.html"))

    # ---- layout ---------------------------------------------------------------------------------
    use = pick["cards"] or 1
    used = chosen[:use]
    hip = ",".join(str(g["index"]) for g in used)
    tp, pp = (1, use) if r.get("layout") == "pp" else (use, 1)
    multi = use > 1

    settings.append(setting("Recipe", r["label"], r.get("summary", "")))
    settings.append(setting("Install path", "Container (%s)" % (cont.get("flavor") or cont.get("engine") or "docker") if path == "container"
                            else "Host venv + fork checkout",
                            "Fastest first serve; no build." if path == "container"
                            else "HEAD kernels and the in-tree tuned launcher (serve_rdna.sh %s)." % r["venv"]["recipe"]))
    settings.append(setting("Parallelism", "PP=%d" % pp if pp > 1 else "TP=%d" % tp,
                            "Pipeline parallel over three cards." if pp > 1 else
                            ("Tensor parallel over %d cards." % tp if tp > 1 else "Single card.")))
    settings.append(setting("HIP_VISIBLE_DEVICES", hip or "-", "Pins the chosen cards and hides the rest."))
    settings.append(setting("dtype", "float16", "RDNA2 has no BF16 dot; BF16 is emulated and slow."))
    hsa = "10.3.0" if gfx in COUSINS else None
    if hsa:
        settings.append(setting("HSA_OVERRIDE_GFX_VERSION", hsa, "%s runs gfx1030 kernels with this override." % gfx))

    # ---- P2P ------------------------------------------------------------------------------------
    p2p = None
    if multi:
        virt = os_i.get("virt", "unknown")
        if p.get("p2p") in ("on", "off"):
            p2p, why = p["p2p"] == "on", "Chosen by you on the Topology page."
        elif p.get("p2p_probe") is not None:
            p2p, why = bool(p["p2p_probe"]), "Measured by the P2P probe on this box."
        elif virt not in ("none", "unknown", ""):
            p2p, why = False, "Running in a VM (%s); passthrough usually blocks GPU peer access." % virt
        elif cpu.get("vendor") == "intel":
            p2p, why = False, ("Intel host: P2P can enumerate but often does not speed up inference. "
                               "Run the P2P probe and A/B it.")
        else:
            p2p, why = True, "AMD bare-metal host, the wiki-validated P2P platform."
        settings.append(setting("P2P", "on" if p2p else "off (P2P=0)", why))
        if p2p and cpu.get("vendor") == "amd" and not os_i.get("iommu_pt"):
            checks.append(check("iommu", "info", "iommu=pt is not on the kernel command line",
                                "Validated P2P setups boot with iommu=pt, ACS off, ReBAR / Above 4G on.", link="tuning/p2p.html"))

    # ---- PLE ------------------------------------------------------------------------------------
    ple = None
    if r.get("ple"):
        ple = p.get("ple", "auto")
        if ple == "auto":
            if ram >= 64:
                ple, why = "int4", "%.0f GB RAM: the INT4 table (~35 GB working set) runs at speed parity with BF16." % ram
            else:
                ple, why = "int4-disk", "%.0f GB RAM: mmap the INT4 table from disk instead of pinning it." % ram
        else:
            why = "Chosen by you."
        if ple == "bf16" and ram < 128:
            checks.append(check("ple", "warn", "BF16 PLE needs ~95 GB of RAM",
                                "This box has %.0f GB. Use int4 or int4-disk." % ram, link="vllm/serve-scripts.html"))
        settings.append(setting("PLE", ple, why))

    # ---- power / ECC ----------------------------------------------------------------------------
    for st in (ecc_status(profile, chosen), power_status(chosen)):
        if st["state"] != "n/a":
            checks.append(st["check"])

    # ---- endpoint -------------------------------------------------------------------------------
    venv_cfg = r.get("venv", {})
    port = int(p.get("port") or (8000 if path == "container" else venv_cfg.get("port") or 8000))
    busy = [s for s in profile.get("servers", []) if s["port"] == port]
    if busy:
        c = check("port", "warn", "Port %d is already serving (%s)" % (port, ", ".join(m for m in busy[0]["models"] if m) or "unknown"),
                  "Watch that server from this dashboard, or pick another port to start a new one.")
        c["attach"] = "http://127.0.0.1:%d/v1" % port
        checks.append(c)
    lan = p.get("lan")
    if lan is None:
        lan = bool(profile.get("runtime", {}).get("container"))
    p["lan"] = lan
    host = "0.0.0.0" if lan else "127.0.0.1"
    settings.append(setting("Endpoint", "http://%s:%d/v1" % ("this-box" if lan else "127.0.0.1", port),
                            "Shared on your network: any app on it can use the model." if lan
                            else "Only this machine can reach it. Turn on network sharing in Change to share it."))

    free = profile.get("disk", {}).get("data_free_gb") or profile.get("disk", {}).get("home_free_gb")
    need = 120 if r.get("ple") else 40
    if free is not None and free < need:
        checks.append(check("disk", "warn", "%.0f GB free for models and caches" % free,
                            "Plan for about %d GB (weights, compile caches, build tree)." % need))

    # ---- files and launch --------------------------------------------------------------------
    root = os.path.expanduser(p["root"])
    paths = {"root": root, "venv": os.path.join(root, "venv-7.14.0"), "src": os.path.join(root, "vllm-rdna"),
             "models": os.path.join(root, "models"), "cache": os.path.join(root, "cache"),
             "compose": os.path.join(root, "vllm-compose.yml"),
             "download_limit": os.path.join(root, "dash", "download-limit"),
             # In the dashboard container ~ is the container's; keep everything under the shared data dir.
             "hf_cache": os.path.join(root, "hf-cache") if profile.get("runtime", {}).get("container")
             else os.path.expanduser("~/.cache/huggingface")}
    m = r["model"]
    local_model = (p.get("model_path") or "").strip() or None
    model_dir = local_model or os.path.join(paths["models"], m["dir"])
    if local_model:
        settings.append(setting("Model", local_model, "Already on this box, so nothing is downloaded. Flags and layout come from %s: "
                                                      "pick the recipe that matches this checkpoint." % r["label"]))
        if not os.path.isdir(local_model):
            checks.append(check("model_path", "warn", "Can't see %s" % local_model,
                                "Check the path. A dashboard in a container only sees its data folder and RDNA_DASH_MODEL_DIRS."))
    extra_args, extra_env = launch_extras(p, checks)
    if p.get("kv_offload_gb"):
        settings.append(setting("KV offload", "%s GB" % p["kv_offload_gb"], "Chosen by you. It holds more context in CPU memory, but "
                                                                            "the wiki warns it can slow decoding down."))
    if extra_args or extra_env:
        settings.append(setting("Your additions", " ".join(extra_args + ["%s=%s" % kv for kv in extra_env.items()]), "Added to the launch as you wrote them."))
    ple_dir = os.path.join(paths["models"], r["ple"]["dir"], r["ple"]["subdir"]) if ple else None
    ccfg = r.get("container", {})
    max_len = p.get("max_len") or (ccfg.get("max_len") if path == "container" else None)
    if max_len:
        settings.append(setting("Max context", str(max_len), "Chosen by you." if p.get("max_len")
                                else "Conservative start; raise it once the first request works."))
    rocm_home = rocm.get("sdk_root") or (os.path.dirname(os.path.dirname(rocm["hipcc"])) if rocm.get("hipcc") else "/opt/rocm/core-7.14")
    if path == "container":
        served = ccfg.get("served_name", "vllm-rdna")
        launch = container_launch(reg, profile, paths, r, used, tp, port, host, served, max_len, p2p, hsa, ple, ple_dir, model_dir, extra_args, extra_env)
    else:
        served = venv_cfg.get("served_name")
        launch = venv_launch(paths, venv_cfg, model_dir, port, host, hip, p2p, hsa, ple, ple_dir, max_len, tp, rocm_home, extra_args, extra_env)

    # ---- alternatives -------------------------------------------------------------------------
    other = [x for x in pick["paths"] if x != path]
    if other:
        alts.append({"label": "Use %s instead" % ("a container" if other[0] == "container" else "the host venv"),
                     "prefs": {"path": other[0]},
                     "why": "No compile step; lags the branch by a few weeks." if other[0] == "container"
                     else "Newer kernels than the Hub image and the tuned launcher."})
    for f in sorted(fits_ok, key=lambda f: -f["recipe"].get("priority", 0)):
        if f["recipe"] is not r:
            alts.append({"label": "Serve %s" % f["recipe"]["label"], "prefs": {"recipe": f["recipe"]["id"]},
                         "why": f["recipe"].get("summary", "")})

    levels = [c["level"] for c in checks]
    return {
        "prefs": p,
        "local_models": profile.get("local_models", []),
        "verdict": "blocked" if "fail" in levels else ("attention" if "warn" in levels else "ready"),
        "summary": {"gpus": "%d x %s" % (n, card.get("name", "GPU")) if n else "no GPU", "using": len(used),
                    "gfx": gfx, "vram_gb": vram, "ram_gb": ram, "cpu": cpu.get("model", ""),
                    "os": os_i.get("name", ""), "rocm": rv or "none"},
        "choice": {"recipe_id": r["id"], "recipe_label": r["label"], "path": path,
                   "launcher_recipe": venv_cfg.get("recipe") if path == "venv" else None,
                   "model_repo": m["repo"], "model_dir": model_dir, "served_name": served,
                   "port": port, "host": host, "tp": tp, "pp": pp, "hip_visible_devices": hip,
                   "p2p": p2p, "ple": ple, "ple_dir": ple_dir, "hsa_override": hsa, "max_len": max_len,
                   "rocm_home": rocm_home, "engine": cont.get("engine")},
        "registry": [{"id": f["recipe"]["id"], "label": f["recipe"]["label"], "fits": bool(f["cards"]),
                      "why": f["why"], "paths": f["paths"], "verify": bool(f["recipe"].get("verify"))} for f in fits],
        "paths": paths,
        "checks": checks,
        "settings": settings,
        "alternatives": alts,
        "launch": launch,
        "steps": build_steps(reg, path, r, multi, paths, model_dir, ple, rocm_home, launch, checks, cont, local_model),
    }


# ---------------------------------------------------------------------------------------------
# Launch commands


def container_spec(reg, paths, r, used, tp, port, host, served, max_len, p2p, hsa, ple, ple_dir, model_dir, groups=None, extra_args=(), extra_env=None):
    env = {"HIP_VISIBLE_DEVICES": ",".join(str(g["index"]) for g in used)}
    if hsa:
        env["HSA_OVERRIDE_GFX_VERSION"] = hsa
    if p2p is False:
        env["NCCL_P2P_DISABLE"] = "1"
        env["VLLM_RDNA_AR"] = "0"
    vols = ["%s:/root/.cache/huggingface" % paths["hf_cache"],
            "%s:/root/.triton/cache" % os.path.join(paths["cache"], "triton"),
            "%s:/root/.cache/vllm" % os.path.join(paths["cache"], "vllm")]
    ccfg = r.get("container", {})
    # Always serve a downloaded folder: one download location for every recipe, no HF cache lookups.
    vols.append("%s:/model" % model_dir)
    target = "/model"
    if ple:
        vols.append("%s:/ples_int4" % ple_dir)
        env["VLLM_PLE_QUANT_DIR"] = "/ples_int4"
        if ple == "int4-disk":
            env["VLLM_PLE_CPU_OFFLOAD"] = "1"
    cmd = ["vllm", "serve", target, "--dtype", "float16", "--served-model-name", served,
           "--host", "0.0.0.0", "--port", "8000"]
    if tp > 1:
        cmd += ["--tensor-parallel-size", str(tp)]
    cmd += ccfg.get("args", [])
    if max_len:
        cmd += ["--max-model-len", str(max_len)]
    cmd += list(extra_args)
    env.update(extra_env or {})
    groups = groups or gpu_groups({})
    publish = ("%d:8000" % port) if host == "0.0.0.0" else ("127.0.0.1:%d:8000" % port)
    return {"image": reg["image"], "env": env, "volumes": vols, "command": cmd, "publish": publish, "groups": groups}


def compose_yaml(spec, rootless_podman):
    """compose.yml that runs under docker compose, podman compose and podman-compose."""
    groups = '["keep-groups"]' if rootless_podman else "[%s]" % ", ".join(json.dumps(g) for g in spec["groups"])

    def s(v):
        return json.dumps(str(v))

    lines = [
        "# Generated by vllm-rdna-dash. Works with docker compose and podman compose.",
        "name: vllm-rdna",
        "services:",
        "  vllm:",
        "    image: %s" % s(spec["image"]),
        "    container_name: vllm-rdna",
        "    devices: [/dev/kfd, /dev/dri]",
        "    group_add: %s%s" % (groups, "   # rootless Podman: pass your host video/render groups through" if rootless_podman
                                     else "   # host GIDs of /dev/kfd and /dev/dri"),
        "    ipc: host",
        "    # label=disable lets the container open /dev/kfd under SELinux (Fedora, RHEL).",
        "    security_opt: [seccomp=unconfined, label=disable]",
        "    ports: [%s]" % s(spec["publish"]),
        "    restart: \"no\"",
        "    environment:",
    ] + ["      %s: %s" % (k, s(v)) for k, v in spec["env"].items()] + [
        "    volumes:",
    ] + ["      - %s" % s(v) for v in spec["volumes"]] + [
        "    command: [%s]" % ", ".join(s(c) for c in spec["command"][0:]),
    ]
    return "\n".join(lines) + "\n"


def _run_argv(engine, spec, rootless_podman):
    args = [engine, "run", "-d", "--name", "vllm-rdna", "--device", "/dev/kfd", "--device", "/dev/dri"]
    if rootless_podman:
        args += ["--group-add", "keep-groups"]
    else:
        for g in spec["groups"]:
            args += ["--group-add", g]
    args += ["--security-opt", "seccomp=unconfined", "--security-opt", "label=disable", "--ipc", "host",
             "-p", spec["publish"]]
    for v in spec["volumes"]:
        args += ["-v", v]
    for k, v in spec["env"].items():
        args += ["-e", "%s=%s" % (k, v)]
    return args + [spec["image"]] + spec["command"]


def container_launch(reg, profile, paths, r, used, tp, port, host, served, max_len, p2p, hsa, ple, ple_dir, model_dir, extra_args=(), extra_env=None):
    cont = profile.get("container", {})
    engine = cont.get("engine") or "docker"
    rootless = engine == "podman" and cont.get("rootless")
    spec = container_spec(reg, paths, r, used, tp, port, host, served, max_len, p2p, hsa, ple, ple_dir, model_dir,
                          gpu_groups(profile), extra_args, extra_env)
    compose = cont.get("compose") or None
    run_argv = _run_argv(engine, spec, rootless)
    return {
        "kind": "container", "engine": engine, "container": "vllm-rdna", "spec": spec,
        "compose_file": paths["compose"], "compose_yaml": compose_yaml(spec, rootless),
        "compose_cmd": compose,
        "up": (compose + ["-f", paths["compose"], "up", "-d"]) if compose else run_argv,
        "down": (compose + ["-f", paths["compose"], "down"]) if compose else [engine, "rm", "-f", "vllm-rdna"],
        "logs": [engine, "logs", "--tail", "400", "vllm-rdna"],
        "command": ("%s -f %s up -d" % (" ".join(compose), q(paths["compose"]))) if compose
        else " \\\n  ".join(_pairs(run_argv)),
        "run_command": " \\\n  ".join(_pairs(run_argv)),
    }


def _pairs(args):
    """Group flags with their values so a printed command reads cleanly."""
    i = 0
    while i < len(args) and not (args[i].startswith("-") and i + 1 < len(args) and not args[i + 1].startswith("-")):
        i += 1
    out = [" ".join(q(a) for a in args[:i])] if i else []
    while i < len(args):
        a = q(args[i])
        if args[i].startswith("-") and i + 1 < len(args) and not args[i + 1].startswith("-"):
            out.append(a + " " + q(args[i + 1]))
            i += 2
        else:
            out.append(a)
            i += 1
    return out


def venv_launch(paths, vcfg, model_dir, port, host, hip, p2p, hsa, ple, ple_dir, max_len, tp, rocm_home, extra_args=(), extra_env=None):
    env = {"MODEL": model_dir, "VENV": paths["venv"], "ROCM_SDK_ROOT": rocm_home}
    env.update(extra_env or {})
    if hsa:
        env["HSA_OVERRIDE_GFX_VERSION"] = hsa
    keys = {"PORT": port, "HOST": host, "HIP_VISIBLE_DEVICES": hip}
    if vcfg.get("recipe_tp") and tp != vcfg["recipe_tp"]:
        keys["TP"] = tp
    if p2p is False:
        keys["P2P"] = 0
    if ple:
        keys["PLE"] = ple
        keys["PLE_QUANT_DIR"] = ple_dir
    if max_len:
        keys["MAXLEN"] = max_len
    if extra_args:
        keys["EXTRA_ARGS"] = " ".join(extra_args)  # serve_rdna.sh appends these to vllm serve
    env_s = " ".join("%s=%s" % (k, q(v)) for k, v in env.items())
    keys_s = " ".join("%s=%s" % (k, q(v)) for k, v in keys.items())
    base = "cd %s && %s %%sbash tools/rdna/serve_rdna.sh RECIPE=%s %s" % (q(paths["src"]), env_s, vcfg["recipe"], keys_s)
    return {"kind": "venv", "recipe": vcfg["recipe"], "command": base % "", "dry_run": base % "PRINT=1 ",
            "env": env, "keys": keys}


# ---------------------------------------------------------------------------------------------
# Install steps


def build_steps(reg, path, r, multi, paths, model_dir, ple, rocm_home, launch, checks, cont, local_model=None):
    """Ordered install steps. kind: check (from the profile), run (the dashboard runs it),
    serve (start the server), smoke (first request). Steps that need sudo are never run by
    the dashboard: they show up as fixes on the host check."""
    fails = sorted(c["id"] for c in checks if c["level"] == "fail")
    steps = [{"id": "host", "kind": "check", "title": "Check the host",
              "summary": "Driver, GPU access, groups, and the tools this path needs. Anything that needs root shows the "
                         "exact commands; run them in a terminal, then press Re-check.", "blocked_by": fails}]
    tune = [c for c in checks if c["id"] in ("ecc", "power")]
    if tune:
        steps.append({"id": "tune", "kind": "tune", "title": "Recommended V620 tuning",
                      "summary": "ECC off for ~2 GB more VRAM per card and a 180 W power cap. Both need root, so run the "
                                 "commands below in a terminal. Setup carries on without them; you can do them any time.",
                      "pending": [c["id"] for c in tune if c["level"] == "rec"], "checks": tune})
    m = r["model"]
    venv, src = paths["venv"], paths["src"]
    if path == "container":
        eng = cont.get("engine") or "docker"
        steps.append({"id": "pull", "kind": "run", "title": "Pull the vllm-rdna image",
                      "summary": "Hub bake of rdna_extras: ROCm 7.14, torch 2.12, Triton 3.7.1.",
                      "script": "%s pull %s" % (eng, reg["image"])})
        steps += download_steps(m, r, ple, paths, model_dir, local_model)
        steps.append({"id": "compose", "kind": "write", "title": "Write the vLLM compose file",
                      "summary": "Saves the tuned service to %s so you can also run it by hand with compose." % paths["compose"]})
    else:
        uv = '"$(command -v uv || echo "$HOME/.local/bin/uv")"'
        steps.append({"id": "uv", "kind": "run", "title": "Install uv",
                      "summary": "Astral's Python package manager. It also fetches Python 3.12 if the distro lacks it.",
                      "script": 'command -v uv >/dev/null || [ -x "$HOME/.local/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh\n'
                                '%s --version' % uv})
        steps.append({"id": "torch", "kind": "run", "title": "Create the venv with ROCm 7.14 PyTorch",
                      "summary": "torch 2.12.0+rocm7.14.0 and the gfx1030 device wheels, pinned to the Docker bake. "
                                 "Never upgrade Triton: 3.9+ drops RDNA2.",
                      "script": "\n".join([
                          "UV=%s" % uv, "mkdir -p %s" % q(paths["root"]),
                          '[ -x %s/bin/python ] || "$UV" venv --python 3.12 %s' % (q(venv), q(venv)),
                          "export VIRTUAL_ENV=%s" % q(venv),
                          '"$UV" pip install --pre --force-reinstall --index-url %s "torch==2.12.0+rocm7.14.0" '
                          '"torchvision==0.27.0+rocm7.14.0" "torchaudio==2.11.0+rocm7.14.0" "triton==3.7.1"' % AMD_INDEX,
                          '"$UV" pip install --index-url %s "rocm-sdk-device-gfx1030==7.14.0" '
                          '"amd-torch-device-gfx1030==2.12.0+rocm7.14.0"' % AMD_INDEX,
                          "%s/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'" % q(venv)])})
        fk = reg["fork"]
        steps.append({"id": "clone", "kind": "run", "title": "Clone the fork",
                      "summary": "opengfx1030/vllm-rdna, branch %s (upstream vLLM 0.31.0 + the RDNA port)." % fk["branch"],
                      "script": "if [ -d %s/.git ]; then git -C %s fetch origin && git -C %s checkout %s && git -C %s pull --ff-only; "
                                "else git clone -b %s %s %s; fi\ngit -C %s log -1 --oneline"
                                % (q(src), q(src), q(src), fk["branch"], q(src), fk["branch"], fk["url"], q(src), q(src))})
        steps.append({"id": "build", "kind": "run", "title": "Build the fork into the venv",
                      "summary": "Compiles the HIP kernels for gfx1030 only. The long step: often 20-60 minutes.",
                      "script": "\n".join([
                          "UV=%s" % uv, "export VIRTUAL_ENV=%s PATH=%s/bin:$PATH" % (q(venv), q(venv)), "cd %s" % q(src),
                          '"$UV" pip install "cmake>=3.26.1" ninja packaging "setuptools>=77.0.3,<81.0.0" '
                          '"setuptools-scm>=8" "setuptools-rust>=1.9.0" wheel jinja2',
                          'VLLM_USE_PRECOMPILED=0 ENABLE_CK=0 ROCM_HOME=%s PYTORCH_ROCM_ARCH=gfx1030 MAX_JOBS="${MAX_JOBS:-$(nproc)}" '
                          '"$UV" pip install -e . --no-build-isolation' % q(rocm_home)])})
        steps.append({"id": "verify", "kind": "run", "title": "Verify the install",
                      "summary": "Should print a +rocm7.14 torch and your gfx1030 card.",
                      "script": "%s/bin/python -c 'import torch, vllm, vllm._rocm_C; "
                                "print(torch.__version__, torch.cuda.get_device_name(0), \"vllm\", vllm.__version__)'" % q(venv)})
        if multi:
            steps.append({"id": "p2p", "kind": "run", "optional": True, "title": "Probe PCIe P2P",
                          "summary": "Asks PyTorch which card pairs reach each other directly. The plan turns P2P on only "
                                     "if every pair can.",
                          "script": "%s/bin/python - <<'PY'\n%sPY" % (q(venv), P2P_PROBE)})
        steps += download_steps(m, r, ple, paths, model_dir, local_model)
        steps.append({"id": "dryrun", "kind": "run", "title": "Dry-run the tuned launch",
                      "summary": "serve_rdna.sh PRINT=1 prints the exact env and argv without starting anything.",
                      "script": launch["dry_run"]})
    steps.append({"id": "serve", "kind": "serve", "title": "Start the server",
                  "summary": "First boot compiles graphs and can take a while; later boots reuse the cache. The dashboard "
                             "watches the log and flags known errors."})
    steps.append({"id": "smoke", "kind": "smoke", "title": "First request",
                  "summary": "Health check, model list, and one chat completion with timing."})
    return steps


def download_steps(m, r, ple, paths, model_dir, local_model):
    """A model already on the box downloads nothing, except a PLE table the recipe still needs."""
    if not local_model:
        return [download_step(m, r, ple, paths, model_dir)]
    if not ple:
        return []
    dl = '"${RDNA_PY:-python3}" -m rdna_dash.download'
    return [{"id": "model", "kind": "run", "title": "Download the PLE table", "download": True,
             "summary": "Your model stays where it is. This recipe still needs its PLE table from %s." % r["ple"]["repo"],
             "script": "%s %s %s --include %s --limit-file %s" % (dl, q(r["ple"]["repo"]), q(os.path.join(paths["models"], r["ple"]["dir"])),
                                                                  q(r["ple"]["subdir"] + "/*"), q(paths["download_limit"]))}]


def download_step(m, r, ple, paths, model_dir):
    """Model (and PLE table) into plain folders, with a live speed limit and progress."""
    dl = '"${RDNA_PY:-python3}" -m rdna_dash.download'
    lim = "--limit-file %s" % q(paths["download_limit"])
    lines = ["%s %s %s%s %s" % (dl, q(m["repo"]), q(model_dir),
                                " --revision %s" % q(m["revision"]) if m.get("revision") else "", lim)]
    if ple:
        lines.append("%s %s %s --include %s %s" % (dl, q(r["ple"]["repo"]), q(os.path.join(paths["models"], r["ple"]["dir"])),
                                                   q(r["ple"]["subdir"] + "/*"), lim))
    if m.get("must_contain"):
        lines.append('test -f %s/%s || { echo "missing %s in the checkpoint"; exit 1; }'
                     % (q(model_dir), m["must_contain"], m["must_contain"]))
    return {"id": "model", "kind": "run", "title": "Download the model", "download": True,
            "summary": "%s into %s. Resumes if interrupted; set a speed limit below, even while it runs. "
                       "Gated models need HF_TOKEN." % (m["repo"], model_dir),
            "script": "\n".join(lines)}


P2P_PROBE = """import json, torch
n = torch.cuda.device_count()
m = [[i == j or torch.cuda.can_device_access_peer(i, j) for j in range(n)] for i in range(n)]
for i, row in enumerate(m):
    print("GPU%d" % i, " ".join("Y" if x else "." for x in row))
print("P2P_RESULT " + json.dumps({"devices": n, "matrix": m, "all": all(all(r) for r in m)}))
"""
