"""Stdlib tests: python -m unittest discover -s tests"""

import copy
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdna_dash import bench, demo, logdoctor, telemetry, tuning  # noqa: E402

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
        self.assertIn(os.path.join("/srv/vllm-rdna", "hf-cache") + "':/root/.cache/huggingface", step["script"])
        self.assertNotIn("--local-dir", step["script"])

    def test_step_scripts_never_use_sudo(self):
        for prefs in ({}, {"recipe": "qwen38-27b-awq"}, {"path": "venv", "recipe": "qwen38-27b-awq"}):
            for s in plan(self.base, prefs)["steps"]:
                self.assertNotIn("sudo", s.get("script", ""), s["id"])


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
    def test_finds_known_failures(self):
        log = "x\nERROR AMDSMI_STATUS_NOT_INIT - Device not initialized\ny\nRuntimeError: hipErrorInvalidImage\n"
        titles = [f["title"] for f in logdoctor.diagnose(log)]
        self.assertIn("vLLM cannot see the GPU", titles)
        self.assertIn("Kernels were not built for this GPU", titles)
        self.assertEqual(logdoctor.phase("INFO Loading safetensors\nINFO Capturing CUDA graph shapes"), "Capturing graphs")


class PromTest(unittest.TestCase):
    def test_parse_sums_labels(self):
        m = telemetry.parse_prom('# HELP x\nvllm:num_requests_running{a="1"} 2\nvllm:num_requests_running{a="2"} 3\nvllm:x_total 1e3\n')
        self.assertEqual(m["vllm:num_requests_running"], 5)
        self.assertEqual(m["vllm:x_total"], 1000)


class BenchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake = demo.FakeVllm(port=18998)
        cls.fake.start_server_process(boot_seconds=0)

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
