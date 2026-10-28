# vllm-rdna-dash

**Install, tune, benchmark and watch [vllm-rdna](https://github.com/opengfx1030/vllm-rdna) on AMD RDNA2 (gfx1030) GPUs, from one page.**

Install the tool, open it, press **Set everything up**, and chat with a model. It looks at your machine, picks the recipe that fits, runs the install steps it can, shows you the commands for the ones that need root, starts the server, and turns into a live dashboard.

Built for the [gfx1030 community](https://blivioniag.github.io/gfx1030-wiki/): Radeon PRO V620 and W6800 first, RX 6800/6900 and the RDNA2 cousins (via `HSA_OVERRIDE_GFX_VERSION`) too.

## Quick start

On the GPU box (Linux, Python 3.8+, git):

```bash
curl -fsSL https://raw.githubusercontent.com/crispyging/vllm-rdna-dash/main/install.sh | sh -s -- --start
```

Then open <http://127.0.0.1:7860>. From another computer, either tunnel it:

```bash
ssh -L 7860:127.0.0.1:7860 your-gpu-box
```

or start it with `rdna-dash --host 0.0.0.0` and open the token link it prints.

Just looking? `python -m rdna_dash --demo` runs the whole UI against a pretend 4 × V620 box and a fake model server. It installs nothing.

No browser handy? `rdna-dash doctor` prints the same checks in the terminal.

## What it does

**Setup.** A read-only profile of the box covers:
- GPUs (PCI id, gfx target, VRAM, ECC counters, power cap)
- ROCm installs (`/opt/rocm`, TheRock `core-*`, or distro ROCm under `/usr`)
- `video`/`render` groups and `/dev/kfd`
- Docker or Podman (rootless too) and compose
- package manager, SELinux, virtualization, CPU vendor, RAM and disk

Every finding comes as a plain sentence, the fix for your package manager, and a wiki link. Steps that need root are never run for you; they show up as commands to paste. Everything else (uv, the ROCm 7.14 PyTorch venv, cloning and building the fork, the model download, the P2P probe, the dry run, compose.yml, the start-up and a first request) runs from one button or one step at a time, with live logs.

**Tuning per box.** The plan picks:
- the recipe
- container or host venv
- TP or PP and which GPUs (`HIP_VISIBLE_DEVICES`)
- P2P on or off: measured when possible; off in VMs; A/B on Intel
- the PLE mode for Flash-Next, based on RAM
- `HSA_OVERRIDE_GFX_VERSION` for cousin cards
- the port, and whether to bind to localhost or the LAN

It warns about the ROCm 7.2.1-7.13 multi-GPU RCCL range. Every setting shows why it was chosen. "Change" lets you override any of it.

**Dashboard.** Server state in words (Serving / Starting: compiling / Stopped with an error), plus:
- output and prompt tok/s
- running and waiting requests
- KV cache use, TTFT, prefix-cache hits, MTP draft acceptance
- per-GPU busy %, VRAM, edge/junction/memory temperatures, power against the cap, and clocks
- host CPU and RAM

A log doctor matches known vLLM-on-RDNA failures (no GPU access, missing gfx1030 wheels, RCCL, `rdna_ar` hangs, OOM, offline model, stale Inductor cache, TunableOp mismatch and more) and links each one to its fix.

**Chat.** Streams from the local server, shows first-token time and tok/s per reply, and supports image input for vision models.

**Benchmark.** The same test for every model, so runs compare:
- Prompts come from a seeded corpus, sized to exact token counts with the server's own `/tokenize`.
- A unique salt on every request keeps the prefix cache from flattering a run.
- Output is fixed with `ignore_eos` at temperature 0.
- Contexts run from 1k to 128k, at 1/4/8 concurrent users.
- Vision cases use generated images and are skipped on text-only models.
- A hidden passkey at 50% depth checks that long context actually works.

It reports TTFT p50/p95, prefill tok/s, per-user decode tok/s, inter-token latency, aggregate output and recall. Each run is saved with a fingerprint (box, recipe, TP/PP, P2P, PLE, ROCm) and can be compared with others. The CLI version: `rdna-dash bench http://host:port/v1 --suite standard`.

**Topology.** Draws each GPU's PCIe path (root port, PLX/Broadcom switches, the on-card bridge), its link gen and width, NUMA node and BAR size. For every pair it shows the path type, hop count, and whether the driver created a KFD P2P link. "Measure" runs a PyTorch peer-copy test (in the venv or the image), drawn as a heatmap; the result feeds the P2P setting. If `rocm-bandwidth-test` is installed it runs that too, and raw `amd-smi topology` output is one click away.

**Recommended V620 tuning: ECC off and a 180 W cap.** Both are recommended by default and shown as a setup step right after the host check. They need root and reboots, so setup carries on without them.

The 180 W cap keeps the cards much cooler and quieter, with little speed loss on memory-bound inference:

```bash
sudo ./tools/power.sh status
sudo ./tools/power.sh 180 --persist   # apply now and at every boot
sudo ./tools/power.sh --remove        # back to stock on the next reboot
```

When the driver enforces the 250 W VBIOS floor, the dashboard and the script point you to the [v620_toolbox](https://github.com/blivioniag/v620_toolbox) powerfix first.

For ECC, the dashboard shows whether it's on, half-way through the two-reboot switch, or off. It never recommends turning it off when a card already reports ECC errors. The script does the bootloader work:

```bash
sudo ./tools/ecc.sh status
sudo ./tools/ecc.sh off    # then reboot twice
sudo ./tools/ecc.sh on     # undo, then reboot twice
```

It handles grubby (Fedora/RHEL), update-grub / grub-mkconfig (Debian/Ubuntu/Arch/openSUSE), kernelstub (Pop!_OS) and systemd-boot `/etc/kernel/cmdline`, and backs up what it edits.

**Fleet.** Add your other boxes (dashboard URL and token) and see them side by side. Box names and addresses stay in your local config, never in this repo.

## Recipes

A recipe is one JSON file: model, requirements, and how to launch it in a container and/or with the fork's launcher. Built-ins live in [`recipes/`](recipes/). The tool also loads:

1. `~/.config/vllm-rdna-dash/recipes/*.json` (your own)
2. every folder in `RDNA_DASH_RECIPES` (separated by `:`), for example a team repo checkout

A later file with the same `id` overrides an earlier one. Broken files are listed in the UI, not silently dropped.

```json
{
  "id": "my-model-tp2",
  "label": "My model on two cards",
  "summary": "One line shown in the plan.",
  "priority": 15,
  "layout": "tp",
  "model": { "repo": "org/model-AWQ", "dir": "model-AWQ", "revision": "main" },
  "requires": { "gfx": ["gfx1030"], "cards": [2], "min_vram_per_card_gb": 28, "min_ram_gb": 32 },
  "prefer": "container",
  "container": { "served_name": "my-model", "max_len": 16384, "args": ["--trust-remote-code"] },
  "venv": { "recipe": "27b-awq", "served_name": "q27d", "port": 18210 }
}
```

`venv.recipe` names a launcher recipe in the fork's `tools/rdna/recipes/`. The tuner keeps the recipes that fit the box, prefers ones whose install path works here, and then takes the highest `priority`.

## Running the container by hand

[`compose.yml`](compose.yml) plus [`.env.example`](.env.example) work with `docker compose`, `podman compose` and `podman-compose`. It uses `label=disable` for SELinux hosts and documents the `keep-groups` change for rootless Podman. The dashboard writes a copy tuned for your box to `~/vllm-rdna/compose.yml`.

## Security

- The dashboard binds to `127.0.0.1` by default.
- On any other address, every request needs the token (cookie from the one-time link, or `Authorization: Bearer` for fleet peers). POSTs also need a custom header, which a cross-site page cannot send.
- The model server binds to localhost unless you turn on LAN sharing.
- Install scripts come from the plan, never from the browser, and never use sudo.
- `rdna-dash share` and "Copy redacted report" strip the hostname, username and tokens.

## Layout

```
rdna_dash/
  profile.py    read-only system profile (sysfs, /proc, rocminfo, containers)
  tuning.py     recipes + profile -> checks, settings, steps, launch, compose
  topology.py   PCIe / KFD topology and the peer-copy benchmark
  bench.py      the consistent benchmark (also usable as a library)
  telemetry.py  1 Hz GPU / host / vLLM metrics sampler
  logdoctor.py  known-failure rules
  jobs.py serve.py server.py   step runner, server lifecycle, HTTP API
  demo.py       pretend box + fake vLLM server
  web/          the UI (no build step, no CDN scripts)
recipes/        built-in recipes
tools/ecc.sh    ECC on/off/status
tools/power.sh  V620 power cap (180 W default), optional boot unit
```

Stdlib only. Tests: `python -m unittest discover -s tests`.

## Status

The tuner, benchmark, log doctor and UI are tested in demo mode, and the profiler and `doctor` run on Linux. Not yet verified end to end on real gfx1030 hardware: the first real runs on V620 boxes should check the install steps, the PCIe/KFD parsing and the bandwidth test. Please open an issue with `rdna-dash share` output when something is off.

Model choices and tuning rules follow the [gfx1030 wiki](https://blivioniag.github.io/gfx1030-wiki/) and the fork's README and recipes.
