"""Stdlib tests: python -m unittest discover -s tests"""

import copy
import json
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdna_dash import bench, demo, logdoctor, server, telemetry, topology, tuning  # noqa: E402

REG = tuning.load_registry([tuning.BUILTIN_RECIPES])


def plan(profile, prefs=None):
    return tuning.build_plan(profile, prefs, REG)


def levels(p):
    return {c["id"]: c["level"] for c in p["checks"]}


class TuningTest(unittest.TestCase):
    def setUp(self):
        self.base = demo.demo_profile()

    def test_four_v620_gets_flashnext_mtp2_on_venv(self):
        p = plan(self.base)
        self.assertEqual(p["choice"]["recipe_id"], "flashnext-awq-mtp2")
        self.assertEqual(p["choice"]["path"], "venv")
        self.assertEqual((p["choice"]["tp"], p["choice"]["pp"]), (4, 1))
        self.assertEqual(p["choice"]["ple"], "int4")
        self.assertIn("RECIPE=flashnext-mtp2", p["launch"]["command"])
        self.assertIn("PLE=int4", p["launch"]["command"])

    def test_three_cards_get_pp3_and_low_ram_gets_disk_ple(self):
        prof = copy.deepcopy(self.base)
        prof["gpus"] = prof["gpus"][:3]
        prof["memory"]["total_gb"] = 48
        p = plan(prof)
        self.assertEqual(p["choice"]["recipe_id"], "flashnext-awq-pp3")
        self.assertEqual(p["choice"]["pp"], 3)
        self.assertEqual(p["choice"]["ple"], "int4-disk")

    def test_vm_and_intel_turn_p2p_off(self):
        for key, val in (("virt", "kvm"),):
            prof = copy.deepcopy(self.base)
            prof["os"][key] = val
            self.assertIs(plan(prof)["choice"]["p2p"], False)
        prof = copy.deepcopy(self.base)
        prof["cpu"]["vendor"] = "intel"
        p = plan(prof, {"recipe": "qwen38-27b-awq"})
        self.assertIs(p["choice"]["p2p"], False)
        self.assertEqual(p["launch"]["spec"]["env"].get("NCCL_P2P_DISABLE"), "1")

    def test_probe_result_overrides_guess(self):
        prof = copy.deepcopy(self.base)
        prof["os"]["virt"] = "kvm"
        self.assertIs(plan(prof, {"p2p_probe": True})["choice"]["p2p"], True)

    def test_rccl_bug_range_warns_on_multi_gpu(self):
        prof = copy.deepcopy(self.base)
        prof["rocm"]["primary"] = "7.2.4"
        self.assertEqual(levels(plan(prof))["rocm"], "warn")
        self.assertFalse(tuning.bad_rccl_rocm("7.2.0"))
        self.assertTrue(tuning.bad_rccl_rocm("7.13.1"))
        self.assertFalse(tuning.bad_rccl_rocm("7.14.0"))

    def test_missing_groups_and_kfd_block(self):
        prof = copy.deepcopy(self.base)
        prof["access"].update(in_render=False, kfd_exists=False)
        p = plan(prof)
        self.assertEqual(p["verdict"], "blocked")
        self.assertEqual(levels(p)["groups"], "fail")
        self.assertIn("render", " ".join(p["checks"][[c["id"] for c in p["checks"]].index("groups")]["fix"]))

    def test_cousin_card_gets_hsa_override(self):
        prof = copy.deepcopy(self.base)
        prof["gpus"] = [dict(g, gfx="gfx1031", device_id="73df", name="RX 6700 XT", vram_gb=12, nominal_vram_gb=12) for g in prof["gpus"][:2]]
        p = plan(prof)
        self.assertEqual(p["choice"]["hsa_override"], "10.3.0")

    def test_small_card_falls_back_to_smallest_recipe(self):
        prof = copy.deepcopy(self.base)
        prof["gpus"] = [dict(prof["gpus"][0], device_id="73bf", name="RX 6800", vram_gb=16, nominal_vram_gb=16)]
        p = plan(prof)
        self.assertEqual(p["choice"]["recipe_id"], "qwen38-27b-awq")
        self.assertIn("fit", levels(p))

    def test_rootless_podman_compose_uses_keep_groups_and_selinux_label(self):
        prof = copy.deepcopy(self.base)
        prof["container"].update(engine="podman", rootless=True, compose=["podman", "compose"])
        p = plan(prof, {"recipe": "qwen38-27b-awq"})
        y = p["launch"]["compose_yaml"]
        self.assertIn('group_add: ["keep-groups"]', y)
        self.assertIn("label=disable", y)
        self.assertEqual(p["launch"]["up"][:2], ["podman", "compose"])

    def test_lan_off_binds_loopback(self):
        p = plan(self.base, {"recipe": "qwen38-27b-awq"})
        self.assertTrue(p["launch"]["spec"]["publish"].startswith("127.0.0.1:"))
        p = plan(self.base, {"recipe": "qwen38-27b-awq", "lan": True})
        self.assertFalse(p["launch"]["spec"]["publish"].startswith("127.0.0.1:"))

    def test_ecc_states(self):
        self.assertEqual(levels(plan(self.base))["ecc"], "rec")
        prof = copy.deepcopy(self.base)
        prof["gpus"][1]["ecc_errors"] = {"ce": 2, "ue": 0}
        self.assertEqual(levels(plan(prof))["ecc"], "warn")
        prof = copy.deepcopy(self.base)
        prof["os"]["ras_disabled_param"] = True
        self.assertIn("reboot once more", [c for c in plan(prof)["checks"] if c["id"] == "ecc"][0]["title"])
        prof = copy.deepcopy(self.base)
        for g in prof["gpus"]:
            g["vram_gb"] = 31.98
        self.assertEqual(levels(plan(prof))["ecc"], "ok")

    def test_power_cap_recommended_at_180(self):
        p = plan(self.base)
        self.assertEqual(levels(p)["power"], "rec")
        tune = [s for s in p["steps"] if s["id"] == "tune"][0]
        self.assertEqual(sorted(tune["pending"]), ["ecc", "power"])
        self.assertTrue(any("v620_toolbox" in f for f in [c for c in p["checks"] if c["id"] == "power"][0]["fix"]))
        prof = copy.deepcopy(self.base)
        for g in prof["gpus"]:
            g.update(power_cap_w=180, power_cap_min_w=120, vram_gb=31.98)
        p = plan(prof)
        self.assertEqual(levels(p)["power"], "ok")
        self.assertEqual([s for s in p["steps"] if s["id"] == "tune"][0]["pending"], [])

    def test_container_mode_serves_in_containers(self):
        prof = copy.deepcopy(self.base)
        prof["runtime"] = {"container": True}
        prof["access"].update(in_video=False, in_render=False)  # container groups say nothing about the host
        p = plan(prof, {"root": "/srv/vllm-rdna"})
        self.assertEqual(p["choice"]["path"], "container")
        self.assertEqual(p["choice"]["recipe_id"], "flashnext-awq-mtp2")
        self.assertNotIn("groups", levels(p))
        self.assertEqual(p["verdict"], "ready")
        self.assertEqual(p["paths"]["hf_cache"], os.path.join("/srv/vllm-rdna", "hf-cache"))
        self.assertTrue(p["paths"]["compose"].endswith("vllm-compose.yml"))
        self.assertTrue(all(v.split(":")[0].startswith("/srv/vllm-rdna") for v in p["launch"]["spec"]["volumes"]))
        ecc = [c for c in p["checks"] if c["id"] == "ecc"][0]
        self.assertIn("raw.githubusercontent.com", ecc["fix"][0])
        p = plan(prof, {"path": "venv", "recipe": "qwen38-27b-awq"})
        self.assertEqual(p["choice"]["path"], "container")
        three = copy.deepcopy(prof)
        three["gpus"] = three["gpus"][:3]
        self.assertEqual(plan(three)["choice"]["recipe_id"], "qwen38-27b-awq")  # PP3 needs a host build

    def test_container_mode_model_download_lands_where_server_reads(self):
        prof = copy.deepcopy(self.base)
        prof["runtime"] = {"container": True}
        p = plan(prof, {"root": "/srv/vllm-rdna", "recipe": "qwen38-27b-awq"})
        step = [s for s in p["steps"] if s["id"] == "model"][0]
        model_dir = p["choice"]["model_dir"]
        self.assertIn("rdna_dash.download", step["script"])
        self.assertIn(model_dir, step["script"].replace("'", ""))
        self.assertIn("--limit-file", step["script"])
        self.assertIn(model_dir + ":/model", p["launch"]["spec"]["volumes"])
        self.assertEqual(p["launch"]["spec"]["command"][2], "/model")

    def test_network_sharing_defaults(self):
        prof = copy.deepcopy(self.base)
        prof["runtime"] = {"container": True}
        self.assertEqual(plan(prof, {"recipe": "qwen38-27b-awq"})["choice"]["host"], "0.0.0.0")
        self.assertEqual(plan(self.base, {"recipe": "qwen38-27b-awq"})["choice"]["host"], "127.0.0.1")
        self.assertEqual(plan(prof, {"recipe": "qwen38-27b-awq", "lan": False})["choice"]["host"], "127.0.0.1")

    def test_p2p_user_choice_wins(self):
        self.assertIs(plan(self.base, {"p2p": "off"})["choice"]["p2p"], False)
        prof = copy.deepcopy(self.base)
        prof["os"]["virt"] = "kvm"
        self.assertIs(plan(prof, {"p2p": "on", "p2p_probe": False})["choice"]["p2p"], True)

    def test_containers_use_host_gpu_gids(self):
        prof = copy.deepcopy(self.base)
        prof["access"]["device_gids"] = {"render": [105], "video": [39]}
        p = plan(prof, {"recipe": "qwen38-27b-awq"})
        self.assertIn('group_add: ["39", "105"]', p["launch"]["compose_yaml"])
        self.assertIn("--group-add 105", " ".join(p["launch"]["run_command"].split()))
        # No readable devices: use VIDEO_GID/RENDER_GID from the environment, never names.
        old = {k: os.environ.get(k) for k in ("VIDEO_GID", "RENDER_GID")}
        os.environ.update(VIDEO_GID="44", RENDER_GID="992")
        try:
            p = plan(self.base, {"recipe": "qwen38-27b-awq"})
        finally:
            for k, v in old.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        self.assertIn('group_add: ["44", "992"]', p["launch"]["compose_yaml"])
        self.assertNotIn('"render"', p["launch"]["compose_yaml"])
        # A placeholder pasted into .env is ignored rather than passed to Docker.
        os.environ["VIDEO_GID"] = "<44>"
        try:
            self.assertNotIn("<44>", tuning.gpu_groups({"runtime": {"container": True}}))
        finally:
            os.environ.pop("VIDEO_GID")

    def test_step_scripts_never_use_sudo(self):
        for prefs in ({}, {"recipe": "qwen38-27b-awq"}, {"path": "venv", "recipe": "qwen38-27b-awq"}):
            for s in plan(self.base, prefs)["steps"]:
                self.assertNotIn("sudo", s.get("script", ""), s["id"])


class DownloadTest(unittest.TestCase):
    """The downloader against a fake Hugging Face endpoint."""

    @classmethod
    def setUpClass(cls):
        import http.server
        import threading
        cls.files = {"config.json": b"{}" * 10, "model-1.safetensors": os.urandom(300000), "ples_int4/META.json": b"m" * 50}

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path.startswith("/api/models/"):
                    body = json.dumps({"siblings": [{"rfilename": k, "size": len(v)} for k, v in cls.files.items()]}).encode()
                    self.send_response(200)
                    self.end_headers()
                    return self.wfile.write(body)
                name = self.path.split("/resolve/main/", 1)[1]
                data = cls.files[name.replace("%2F", "/")]
                start = 0
                rng = self.headers.get("Range")
                if rng:
                    start = int(rng.split("=")[1].split("-")[0])
                    self.send_response(206)
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(len(data) - start))
                self.end_headers()
                self.wfile.write(data[start:])

        cls.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        os.environ["HF_ENDPOINT"] = "http://127.0.0.1:%d" % cls.httpd.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        os.environ.pop("HF_ENDPOINT", None)

    def test_downloads_resumes_and_filters(self):
        from rdna_dash import download
        with tempfile.TemporaryDirectory() as d:
            dest = os.path.join(d, "m")
            os.makedirs(dest)
            with open(os.path.join(dest, "model-1.safetensors.part"), "wb") as f:
                f.write(self.files["model-1.safetensors"][:1000])  # a half-finished earlier run
            self.assertEqual(download.main(["org/repo", dest]), 0)
            for k, v in self.files.items():
                with open(os.path.join(dest, *k.split("/")), "rb") as f:
                    self.assertEqual(f.read(), v, k)
            dest2 = os.path.join(d, "ple")
            download.main(["org/repo", dest2, "--include", "ples_int4/*"])
            self.assertEqual(os.listdir(dest2), ["ples_int4"])

    def test_limit_slows_and_is_reread(self):
        from rdna_dash import download
        with tempfile.TemporaryDirectory() as d:
            lf = os.path.join(d, "limit")
            with open(lf, "w") as f:
                f.write("0.5")
            lim = download.Limiter(lf)
            t = time.monotonic()
            for _ in range(5):
                lim.take(200000)  # 1 MB at 0.5 MB/s: about 2 s
            self.assertGreater(time.monotonic() - t, 1.2)
            with open(lf, "w") as f:
                f.write("0")
            lim.checked = 0
            self.assertEqual(lim.current(), 0.0)


class TopologyTest(unittest.TestCase):
    def test_bar_ignores_sriov_vf_ranges(self):
        # BAR0 32 GiB VRAM, BAR2 256 MiB, BAR5 regs, ROM, then VF BARs (12 VFs x 32 GiB = 384 GiB).
        lines = ["0x0000038000000000 0x00000387ffffffff 0x14220c", "0x0 0x0 0x0",
                 "0x0000038800000000 0x000003880fffffff 0x14220c", "0x0 0x0 0x0", "0x0 0x0 0x0",
                 "0x00000000f0000000 0x00000000f00fffff 0x40200", "0x00000000f0100000 0x00000000f011ffff 0x46200",
                 "0x0000039000000000 0x000003efffffffff 0x14220c"] + ["0x0 0x0 0x0"] * 5
        old = topology._read
        topology._read = lambda path, default="": chr(10).join(lines) if path.endswith("/resource") else old(path, default)
        try:
            self.assertEqual(topology._large_bar("0000:03:00.0"), 32.0)
        finally:
            topology._read = old

    def _dev(self, i, root, socket):
        return {"index": i, "root": root, "socket": socket, "numa": str(i), "bridges": [
            {"addr": root + "-rp%d" % i, "kind": "root-port"}], "kfd_node": None}

    def test_one_cpu_many_root_complexes_is_not_cross_socket(self):
        sk = {"count": 1, "node_package": {}, "model": "AMD Ryzen Threadripper PRO 3945WX 12-Cores"}
        self.assertEqual(topology.socket_of("3", sk), 0)
        a, b = self._dev(0, "pci0000:00", 0), self._dev(1, "pci0000:20", 0)
        self.assertEqual(topology._pair(a, b, {})["via"], "cpu-fabric")

    def test_two_sockets_is_cross_socket(self):
        sk = {"count": 2, "node_package": {"0": 0, "1": 0, "2": 1, "3": 1}, "model": "x"}
        self.assertEqual(topology.socket_of("1", sk), 0)
        self.assertEqual(topology.socket_of("2", sk), 1)
        a, b = self._dev(0, "pci0000:00", 0), self._dev(1, "pci0000:80", 1)
        self.assertEqual(topology._pair(a, b, {})["via"], "cpu-interconnect")


class HostGuardTest(unittest.TestCase):
    def test_allows_ips_and_local_names_only(self):
        ok = ["192.168.1.20:7860", "10.0.0.5", "[::1]:7860", "localhost:7860", "gpu-box", "gpu-box:7860",
              "gpu-box.local", "box.lan:7860", "box.home.arpa"]
        for h in ok:
            self.assertTrue(server.host_allowed(h), h)
        for h in ["evil.example.com", "attacker.net:7860", "", None]:
            self.assertFalse(server.host_allowed(h), h)
        self.assertTrue(server.host_allowed("dash.example.com", ["dash.example.com"]))


class RegistryTest(unittest.TestCase):
    def test_user_dir_overrides_and_bad_files_are_reported(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(tuning.BUILTIN_RECIPES, "qwen38-27b-awq.json"), encoding="utf-8") as f:
                override = dict(json.load(f), label="Mine")
            for name, data in (("a.json", override), ("b.json", {"id": "broken"})):
                with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                    json.dump(data, f)
            reg = tuning.load_registry([tuning.BUILTIN_RECIPES, d])
            labels = {r["id"]: r["label"] for r in reg["recipes"]}
            self.assertEqual(labels["qwen38-27b-awq"], "Mine")
            self.assertTrue(any("b.json" in e for e in reg["errors"]))

    def test_builtin_recipes_validate(self):
        self.assertEqual(REG["errors"], [])
        self.assertGreaterEqual(len(REG["recipes"]), 4)
        self.assertTrue(REG["image"].startswith("docker.io/"))


class DoctorTest(unittest.TestCase):
    def test_unknown_group_points_at_gid_settings(self):
        f = logdoctor.diagnose("Error response from daemon: unable to find group render: no matching entries in group file")
        self.assertEqual(f[0]["title"], "Container group not found")
        self.assertIn("VIDEO_GID", f[0]["detail"])

    def test_finds_known_failures(self):
        log = "x\nERROR AMDSMI_STATUS_NOT_INIT - Device not initialized\ny\nRuntimeError: hipErrorInvalidImage\n"
        titles = [f["title"] for f in logdoctor.diagnose(log)]
        self.assertIn("vLLM cannot see the GPU", titles)
        self.assertIn("Kernels were not built for this GPU", titles)
        self.assertEqual(logdoctor.phase("INFO Loading safetensors\nINFO Capturing CUDA graph shapes"), "Capturing graphs")


class PromTest(unittest.TestCase):
    def test_offload_and_external_hits(self):
        m = telemetry.parse_prom("vllm:kv_cache_usage_perc 0.2\nvllm:cpu_kv_offload_usage_perc 0.35\n")
        self.assertAlmostEqual(telemetry.offload_usage(m), 35.0)
        self.assertIsNone(telemetry.offload_usage(telemetry.parse_prom("vllm:kv_cache_usage_perc 0.2\n")))

    def test_parse_sums_labels(self):
        m = telemetry.parse_prom('# HELP x\nvllm:num_requests_running{a="1"} 2\nvllm:num_requests_running{a="2"} 3\nvllm:x_total 1e3\n')
        self.assertEqual(m["vllm:num_requests_running"], 5)
        self.assertEqual(m["vllm:x_total"], 1000)


class BenchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake = demo.FakeVllm(port=18998)
        cls.fake.start_server_process(boot_seconds=0, background_users=0)

    def test_prompt_sizing_hits_target(self):
        c = bench.Client("http://127.0.0.1:18998/v1", "flash-next")
        text, n = bench.build_prompt(c, 4096, seed=1, passkey=12345)
        self.assertIn("12345", text)
        self.assertLess(abs(n - 4096), 4096 * 0.02)

    def test_png_is_valid(self):
        url = bench.test_image(64)
        import base64
        raw = base64.b64decode(url.split(",", 1)[1])
        self.assertTrue(raw.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_quick_suite_runs_and_recalls(self):
        old = bench.SUITES["quick"]
        bench.SUITES["t"] = dict(old, contexts=[1024], concurrency=[1, 2], vision=[64], output=16)
        try:
            b = bench.Bench("http://127.0.0.1:18998/v1", "t")
            res = b.run()
        finally:
            bench.SUITES.pop("t")
        self.assertEqual(res["status"], "done", res["log"])
        text = [c for c in res["cases"] if c["kind"] == "text"]
        self.assertTrue(all(c["summary"]["errors"] == 0 for c in text))
        self.assertTrue(all(c["summary"]["recall"] == 1.0 for c in text))
        self.assertTrue(all(c["summary"]["decode_tps"] > 0 for c in text))


if __name__ == "__main__":
    unittest.main()
