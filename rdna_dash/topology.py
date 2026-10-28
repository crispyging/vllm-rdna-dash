"""PCIe and KFD topology: who talks to whom, and through how many bridges.

Read-only sysfs walk. Each GPU's sysfs path is its chain of PCIe bridges up to a
root complex; two GPUs meet at their deepest common bridge. The KFD topology
adds the kernel's own view: io_links (how the driver routes traffic) and, on
newer kernels, p2p_links (direct peer access the driver actually enabled).
"""

import glob
import os
import re
import shutil
import subprocess

# Bridge vendors that are PCIe switches rather than root ports.
SWITCH_VENDORS = {"0x10b5": "Broadcom/PLX", "0x11f8": "Microchip/PMC", "0x1000": "Broadcom", "0x12d8": "Pericom",
                  "0x1b21": "ASMedia"}
CPU_VENDORS = {"0x1022": "AMD", "0x8086": "Intel"}


def _read(path, default=""):
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except Exception:
        return default


def _dev(addr):
    return "/sys/bus/pci/devices/" + addr


def _link(addr):
    d = _dev(addr)
    speed = _read(d + "/current_link_speed")
    width = _read(d + "/current_link_width")
    mspeed = _read(d + "/max_link_speed")
    mwidth = _read(d + "/max_link_width")
    gen = {"2.5": 1, "5.0": 2, "8.0": 3, "16.0": 4, "32.0": 5, "64.0": 6}

    def g(s):
        m = re.match(r"([\d.]+) GT/s", s or "")
        return gen.get(m.group(1)) if m else None

    return {"gen": g(speed), "width": int(width) if width.isdigit() else None,
            "max_gen": g(mspeed), "max_width": int(mwidth) if mwidth.isdigit() else None}


def _classify(addr, is_first):
    vendor = _read(_dev(addr) + "/vendor").lower()
    device = _read(_dev(addr) + "/device").lower()
    if vendor == "0x1002":
        kind = "on-card"   # Navi 21 boards carry their own small switch in front of the GPU
    elif is_first:
        kind = "root-port"
    else:
        kind = "switch"
    return {"addr": addr, "vendor": vendor, "device": device, "kind": kind,
            "vendor_name": SWITCH_VENDORS.get(vendor) or CPU_VENDORS.get(vendor) or ("AMD GPU" if vendor == "0x1002" else vendor)}


def gpu_path(addr):
    """[root bus, bridge..., gpu] from the sysfs device path."""
    real = os.path.realpath(_dev(addr))
    parts = real.split("/")
    try:
        i = next(k for k, p in enumerate(parts) if p.startswith("pci"))
    except StopIteration:
        return None, []
    root = parts[i]
    chain = [p for p in parts[i + 1:] if re.match(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$", p)]
    bridges = [_classify(a, k == 0) for k, a in enumerate(chain[:-1])]
    return root, bridges


def kfd_nodes():
    """KFD node id -> {pci, gpu_id, io_links, p2p_links}."""
    out = {}
    for nd in glob.glob("/sys/class/kfd/kfd/topology/nodes/*"):
        props = {}
        for line in _read(nd + "/properties").splitlines():
            k, _, v = line.partition(" ")
            props[k] = v.strip()
        if props.get("simd_count", "0") == "0":
            continue  # CPU node
        loc = int(props.get("location_id", "0"))
        dom = int(props.get("domain", "0"))
        pci = "%04x:%02x:%02x.%x" % (dom, loc >> 8, (loc >> 3) & 0x1F, loc & 0x7)

        def links(sub):
            res = []
            for ld in glob.glob(os.path.join(nd, sub, "*")):
                lp = {}
                for line in _read(ld + "/properties").splitlines():
                    k, _, v = line.partition(" ")
                    lp[k] = v.strip()
                res.append({"to": int(lp.get("node_to", -1)), "type": int(lp.get("type", 0)),
                            "weight": int(lp.get("weight", 0))})
            return res

        out[int(os.path.basename(nd))] = {"pci": pci, "gpu_id": props.get("gpu_id"),
                                          "io_links": links("io_links"), "p2p_links": links("p2p_links"),
                                          "has_p2p_dir": os.path.isdir(os.path.join(nd, "p2p_links"))}
    return out


def discover(gpus):
    """Topology for the given profile GPUs (discrete only)."""
    cards = [g for g in gpus if not g.get("integrated")]
    nodes = kfd_nodes()
    by_pci = {v["pci"]: k for k, v in nodes.items()}
    devs = []
    for g in cards:
        root, bridges = gpu_path(g["pci"])
        oncard = [b for b in bridges if b["kind"] == "on-card"]
        slot = oncard[0]["addr"] if oncard else g["pci"]
        devs.append({
            "index": g["index"], "pci": g["pci"], "name": g["name"], "root": root,
            "numa": _read(_dev(g["pci"]) + "/numa_node", "-1"),
            "bridges": bridges, "link": _link(slot), "kfd_node": by_pci.get(g["pci"]),
            "large_bar": _large_bar(g["pci"]),
        })
    pairs = []
    for i, a in enumerate(devs):
        for b in devs[i + 1:]:
            pairs.append(_pair(a, b, nodes))
    switches = {}
    for d in devs:
        for br in d["bridges"]:
            if br["kind"] == "switch":
                switches.setdefault(br["addr"], {"addr": br["addr"], "vendor": br["vendor_name"], "gpus": []})
                switches[br["addr"]]["gpus"].append(d["index"])
    return {"gpus": devs, "pairs": pairs, "switches": list(switches.values()),
            "kfd": bool(nodes), "kfd_p2p_supported": any(n["has_p2p_dir"] for n in nodes.values()),
            "tools": {"rocm_bandwidth_test": bandwidth_tool(), "amd_smi": shutil.which("amd-smi"),
                      "rocm_smi": shutil.which("rocm-smi")}}


def _large_bar(addr):
    """Resizable/large BAR: the VRAM BAR covers (nearly) all of VRAM."""
    best = 0
    for line in _read(_dev(addr) + "/resource").splitlines():
        try:
            s, e, _ = (int(x, 16) for x in line.split())
            best = max(best, e - s + 1 if e else 0)
        except Exception:
            pass
    return round(best / 1024 ** 3, 1)


def _pair(a, b, nodes):
    pa = [x["addr"] for x in a["bridges"]]
    pb = [x["addr"] for x in b["bridges"]]
    common = 0
    if a["root"] == b["root"]:
        while common < min(len(pa), len(pb)) and pa[common] == pb[common]:
            common += 1
    # Middle men: bridges each side must climb through to meet, not counting the on-card switch.
    up_a = [x for x in a["bridges"][common:] if x["kind"] != "on-card"]
    up_b = [x for x in b["bridges"][common:] if x["kind"] != "on-card"]
    if a["root"] != b["root"]:
        via = "cpu-interconnect" if a["numa"] != b["numa"] and a["numa"] != "-1" else "cpu-root-complex"
        meet = None
    elif common == 0:
        via, meet = "cpu-root-complex", None
    else:
        meet = a["bridges"][common - 1]
        via = "switch" if meet["kind"] == "switch" else "cpu-root-complex"
    kfd_p2p = None
    na, nb = a["kfd_node"], b["kfd_node"]
    if na is not None and nb is not None and nodes.get(na, {}).get("has_p2p_dir"):
        kfd_p2p = any(l["to"] == nb for l in nodes[na]["p2p_links"])
    weight = None
    if na is not None and nb is not None:
        w = [l["weight"] for l in nodes.get(na, {}).get("io_links", []) if l["to"] == nb]
        weight = w[0] if w else None
    return {"a": a["index"], "b": b["index"], "via": via, "meet": meet["addr"] if meet else None,
            "hops": len(up_a) + len(up_b), "middlemen": [x["addr"] + " " + x["kind"] for x in up_a + up_b],
            "kfd_p2p": kfd_p2p, "kfd_weight": weight}


def bandwidth_tool():
    for name in ("rocm-bandwidth-test", "rocm_bandwidth_test"):
        p = shutil.which(name)
        if p:
            return p
    for p in glob.glob("/opt/rocm*/bin/rocm*bandwidth*test"):
        return p
    return None


def amd_smi_topology():
    """Raw `amd-smi topology` text when the tool exists; shown as-is in the UI."""
    exe = shutil.which("amd-smi")
    if not exe:
        exe = shutil.which("rocm-smi")
        args = [exe, "--showtopo"] if exe else None
    else:
        args = [exe, "topology"]
    if not args:
        return None
    try:
        return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20).stdout.decode("utf-8", "replace")
    except Exception:
        return None


# Peer-copy benchmark run inside the venv or the container image. Prints one
# JSON line the dashboard parses; sizes are small enough to finish in seconds.
BANDWIDTH_BENCH = r"""import json, time, torch
n = torch.cuda.device_count()
size = 256 * 1024 * 1024
def pci(i):
    p = torch.cuda.get_device_properties(i)
    return getattr(p, "pci_bus_id", None)
def bench(src, dst, iters=8):
    a = torch.empty(size, dtype=torch.uint8, device=src)
    b = torch.empty(size, dtype=torch.uint8, device=dst)
    b.copy_(a); torch.cuda.synchronize(src); torch.cuda.synchronize(dst)
    t = time.perf_counter()
    for _ in range(iters):
        b.copy_(a, non_blocking=True)
    torch.cuda.synchronize(src); torch.cuda.synchronize(dst)
    return size * iters / (time.perf_counter() - t) / 1e9
names = ["cuda:%d" % i for i in range(n)]
peer = [[torch.cuda.can_device_access_peer(i, j) if i != j else True for j in range(n)] for i in range(n)]
gbs = [[None] * n for _ in range(n)]
for i in range(n):
    for j in range(n):
        if i != j:
            gbs[i][j] = round(bench(names[i], names[j]), 2)
            print("GPU%d -> GPU%d  %6.2f GB/s  %s" % (i, j, gbs[i][j], "peer" if peer[i][j] else "via host"), flush=True)
h2d = []
for i in range(n):
    h = torch.empty(size, dtype=torch.uint8).pin_memory()
    d = torch.empty(size, dtype=torch.uint8, device=names[i])
    d.copy_(h); torch.cuda.synchronize(i)
    t = time.perf_counter()
    for _ in range(8):
        d.copy_(h, non_blocking=True)
    torch.cuda.synchronize(i)
    h2d.append(round(size * 8 / (time.perf_counter() - t) / 1e9, 2))
    print("host -> GPU%d  %6.2f GB/s" % (i, h2d[i]), flush=True)
print("BW_RESULT " + json.dumps({"devices": n, "pci": [pci(i) for i in range(n)], "peer": peer, "gbs": gbs, "h2d": h2d}))
"""
