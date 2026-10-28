"""Spot known vLLM-on-RDNA failures in a server or job log and say what to do.

Each rule is a regex, a plain-language explanation, and a link to the wiki page
that covers it. Add rules here when the community finds a new failure mode.
"""

import re

WIKI = "https://blivioniag.github.io/gfx1030-wiki/"

RULES = [
    (r"AMDSMI_STATUS_NOT_INIT|Failed to infer device type|ROCm platform is not available",
     "vLLM cannot see the GPU",
     "The process has no access to /dev/kfd or the render nodes. Containers need both video and render "
     "(or keep-groups on rootless Podman); on the host, log out and back in after joining those groups.",
     "troubleshooting/vllm.html#failed-to-infer-device-type--amdsmi_status_not_init"),
    (r"hipErrorInvalidImage|no kernel image is available",
     "Kernels were not built for this GPU",
     "The torch install is missing the gfx1030 device wheels, or a cousin card needs HSA_OVERRIDE_GFX_VERSION=10.3.0.",
     "vllm/host-venv.html#what-device-gfx1030-is"),
    (r"AllReduce init failed|NCCL error|RCCL.*(error|failed)|ncclInternalError",
     "Multi-GPU communication failed",
     "Check the ROCm version (7.2.1-7.13 has an RCCL bug; use 7.2.0 or 7.14.0). If P2P is broken on this host, "
     "restart with P2P off.",
     "troubleshooting/vllm.html#multi-gpu-rccl-hangs-or-cards-drop-offline"),
    (r"RPC call .* timed out|rdna_ar wedged",
     "Engine stopped responding during decode",
     "Known one-shot all-reduce hang on older bakes. Use the latest fork HEAD, or restart with P2P off "
     "(VLLM_RDNA_AR=0).",
     "troubleshooting/vllm.html#rdna-ar-silent-hang"),
    (r"(OutOfMemoryError|out of memory|hipErrorOutOfMemory)",
     "Ran out of GPU memory",
     "Lower the max context, the KV budget, or gpu-memory-utilization, or spread over more cards. "
     "Graph capture also needs headroom.",
     "troubleshooting/vllm.html#cuda-graph-capture-crashes"),
    (r"LocalEntryNotFoundError|HF_HUB_OFFLINE|We couldn't connect to 'https://huggingface.co'|Repository Not Found|401 Client Error",
     "Model files not found",
     "The launcher runs offline, so the model must already be downloaded to the folder in MODEL. Gated repos "
     "need HF_TOKEN.",
     "vllm/serve-scripts.html"),
    (r"Address already in use|address already in use",
     "Port already in use",
     "Another server holds this port. Stop it, or change the port in Settings.",
     None),
    (r"assert_size_stride",
     "Stale compiled graph",
     "Inductor reused an old graph after a kernel change. Clear TORCHINDUCTOR_CACHE_DIR and the vLLM compile "
     "cache, then start again.",
     "troubleshooting/vllm.html#torchinductor-cache-survives-vllm-disable-compile-cache"),
    (r"TunableOp.*(mismatch|abort)|rocBLAS version",
     "TunableOp table does not match this rocBLAS",
     "The tuned GEMM rows are tied to a ROCm/rocBLAS build. Use the matching tunableop folder or disable TunableOp.",
     "troubleshooting/vllm.html#tunableop-rocblas-mismatch"),
    (r"Using (Triton|Exllama)\w*LinearKernel",
     "AWQ is not using the RDNA2 kernels",
     "Throughput will be low. Use a current -extras image or fork build; as a stopgap set "
     "VLLM_DISABLED_KERNELS=ExllamaLinearKernel,TritonW4A16LinearKernel.",
     "troubleshooting/vllm.html#gptqawq-not-using-rdna2-kernels"),
    (r"PLE_QUANT_DIR=.* has no META.json|PLE=.* needs PLE_QUANT_DIR",
     "PLE table folder is wrong",
     "Point PLE_QUANT_DIR at the ples_int4 folder from primitive-ai/Qwen3.8-Flash-Next-PLE-quant (it has META.json).",
     "vllm/serve-scripts.html"),
    (r"permission denied.*(/dev/kfd|/dev/dri)|Permission denied: '/dev/kfd'",
     "No permission on the GPU device",
     "On SELinux hosts (Fedora, RHEL) containers need security_opt label=disable; on the host, join video and render.",
     "troubleshooting/vllm.html#failed-to-infer-device-type--amdsmi_status_not_init"),
    (r"bfloat16.*not supported|BF16",
     "BF16 path selected",
     "RDNA2 emulates BF16. Keep --dtype float16.",
     "setup/hardware.html"),
]

_COMPILED = [(re.compile(p, re.I), t, d, l) for p, t, d, l in RULES]

PROGRESS = [
    (re.compile(r"Loading (safetensors|weights)|Loading model", re.I), "Loading model weights"),
    (re.compile(r"Compiling a graph|torch.compile|Inductor|dynamo", re.I), "Compiling (first boot is slow, later boots reuse the cache)"),
    (re.compile(r"Capturing (CUDA|HIP) graph|cudagraph", re.I), "Capturing graphs"),
    (re.compile(r"KV cache|num_gpu_blocks|Maximum concurrency", re.I), "Sizing the KV cache"),
    (re.compile(r"Application startup complete|Uvicorn running|Starting vLLM API server", re.I), "Server is up"),
]


def diagnose(text, limit=6):
    """Return findings for the last occurrence of each known failure."""
    found = []
    seen = set()
    lines = text.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        for rx, title, detail, link in _COMPILED:
            if title in seen:
                continue
            if rx.search(line):
                seen.add(title)
                found.append({"title": title, "detail": detail, "line": line.strip()[:300],
                              "link": WIKI + link if link else None})
        if len(found) >= limit:
            break
    return found


def phase(text):
    """Best guess at what a starting server is doing right now."""
    current = None
    for line in text.splitlines()[-400:]:
        for rx, label in PROGRESS:
            if rx.search(line):
                current = label
    return current
