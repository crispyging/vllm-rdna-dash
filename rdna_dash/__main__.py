"""rdna-dash command line.

  rdna-dash                 start the dashboard (http://127.0.0.1:7860)
  rdna-dash --host 0.0.0.0  reachable from the LAN (token required)
  rdna-dash --demo          try the UI with a pretend 4 x V620 box
  rdna-dash doctor          print what this box has and what is missing
  rdna-dash plan            print the tuned plan and launch command
  rdna-dash share           print a redacted profile to paste in an issue or Discord
  rdna-dash bench URL       run the benchmark against any OpenAI-compatible server
"""

import argparse
import json
import sys

from . import __version__, bench, profile, store, tuning

MARK = {"ok": "  ok  ", "info": " info ", "rec": " TUNE ", "warn": " WARN ", "fail": " FAIL "}


def doctor(args):
    prof = profile.collect()
    cfg = store.load_config()
    plan = tuning.build_plan(prof, cfg.get("prefs"))
    s = plan["summary"]
    print("vllm-rdna-dash %s doctor\n" % __version__)
    print("  GPUs   %s (%s, %.0f GB each)" % (s["gpus"], s["gfx"], s["vram_gb"] or 0))
    print("  Host   %s | %s | %.0f GB RAM | ROCm %s\n" % (s["os"], s["cpu"], s["ram_gb"] or 0, s["rocm"]))
    for c in plan["checks"]:
        print("[%s] %s" % (MARK[c["level"]], c["title"]))
        if c["level"] in ("warn", "fail", "rec") or args.verbose:
            print("         %s" % c["detail"])
            for f in c["fix"]:
                print("           $ %s" % f)
            if c["link"]:
                print("           see %s" % c["link"])
    ch = plan["choice"]
    print("\nPlan: %s via %s, %s, port %d" % (ch["recipe_label"], ch["path"],
                                              "PP=%d" % ch["pp"] if ch["pp"] > 1 else "TP=%d" % ch["tp"], ch["port"]))
    print("Verdict: %s" % {"ready": "ready to install", "attention": "ready, with warnings",
                           "blocked": "fix the FAIL items first"}[plan["verdict"]])
    return 1 if plan["verdict"] == "blocked" else 0


def show_plan(args):
    plan = tuning.build_plan(profile.collect(), store.load_config().get("prefs"))
    if args.json:
        print(json.dumps(plan, indent=2, default=str))
        return 0
    for st in plan["settings"]:
        print("%-22s %s\n%-22s   %s" % (st["key"], st["value"], "", st["why"]))
    print("\nLaunch:\n" + plan["launch"]["command"])
    if plan["launch"]["kind"] == "container":
        print("\ncompose.yml:\n" + plan["launch"]["compose_yaml"])
    return 0


def share(args):
    print(json.dumps(profile.redact(profile.collect()), indent=2))
    return 0


def run_bench(args):
    b = bench.Bench(args.url, args.suite, model=args.model, api_key=args.api_key, thinking_off=not args.thinking)
    import threading
    import time
    t = threading.Thread(target=b.run)
    t.start()
    shown = 0
    while t.is_alive():
        for line in b.log[shown:]:
            print(line)
        shown = len(b.log)
        time.sleep(0.5)
    for line in b.log[shown:]:
        print(line)
    res = b.result
    print("\n%-18s %8s %8s %9s %9s %8s %7s" % ("case", "TTFT p50", "p95", "prefill", "decode", "agg", "recall"))
    for c in res["cases"]:
        name = ("vision %dpx" % c["image"]) if c["kind"] == "vision" else "%dk" % (c["context"] // 1024)
        name += " x%d" % c.get("concurrency", 1) if "concurrency" in c else ""
        if c.get("skip"):
            print("%-18s skipped: %s" % (name, c["skip"]))
            continue
        s = c.get("summary", {})
        f = lambda v, fmt: (fmt % v) if v is not None else "-"
        print("%-18s %8s %8s %9s %9s %8s %7s" % (name, f(s.get("ttft_p50"), "%.2fs"), f(s.get("ttft_p95"), "%.2fs"),
                                                 f(s.get("prefill_tps"), "%.0f"), f(s.get("decode_tps"), "%.1f"),
                                                 f(s.get("output_tps"), "%.1f"),
                                                 f(s.get("recall") * 100 if s.get("recall") is not None else None, "%.0f%%")))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)
    return 0 if res["status"] == "done" else 1


def main(argv=None):
    p = argparse.ArgumentParser(prog="rdna-dash", description="Install, tune and watch vllm-rdna.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--demo", action="store_true", help="pretend 4 x V620 box and fake model server")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd")
    d = sub.add_parser("doctor", help="check this box and say what is missing")
    d.add_argument("-v", "--verbose", action="store_true")
    pl = sub.add_parser("plan", help="print the tuned plan")
    pl.add_argument("--json", action="store_true")
    sub.add_parser("share", help="redacted profile for bug reports")
    b = sub.add_parser("bench", help="benchmark an OpenAI-compatible server")
    b.add_argument("url", help="e.g. http://127.0.0.1:8000/v1")
    b.add_argument("--suite", choices=sorted(bench.SUITES), default="standard")
    b.add_argument("--model")
    b.add_argument("--api-key")
    b.add_argument("--thinking", action="store_true", help="leave the model's thinking mode on")
    b.add_argument("--out", help="write the full JSON result here")
    args = p.parse_args(argv)
    if args.cmd == "doctor":
        return doctor(args)
    if args.cmd == "plan":
        return show_plan(args)
    if args.cmd == "share":
        return share(args)
    if args.cmd == "bench":
        return run_bench(args)
    if args.demo:
        import os
        import tempfile
        base = os.path.join(tempfile.gettempdir(), "vllm-rdna-dash-demo")
        store.set_dirs(os.path.join(base, "config"), os.path.join(base, "state"))
    from .server import run
    run(args.host, args.port, demo=args.demo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
