"""Streaming GEMV mapping, export and live-activation RTL contracts."""
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import npu_mapper as mapper
from test_npu_mapper import tiny_model


class GemvTests(unittest.TestCase):
    def helpers(self):
        self.assertTrue((ROOT / "tools/npu_gemv.py").is_file(), "GEMV helpers are missing")
        return importlib.import_module("npu_gemv")

    def test_gemv_memory_is_only_one_partial_sum_per_output_lane(self):
        gemv = self.helpers()
        for lanes in (1, 2, 4, 8, 16):
            memory = gemv.estimate_gemv_memory(lanes)
            self.assertEqual([memory[key] for key in ("A_bytes", "B_bytes", "C_bytes",
                             "logical_working_set_bytes")], [0, 0, 4*lanes, 4*lanes])
            self.assertIn("partial", memory["note"])
            self.assertIn("no complete input", memory["note"])
            self.assertEqual(memory["parameter_stream_bits_per_result"], 70)
        for lanes in (0, 17, True):
            with self.assertRaises(ValueError):
                gemv.estimate_gemv_memory(lanes)

    def test_schedule_counts_replayed_scalar_and_tail_weight_lanes(self):
        gemv = self.helpers()
        schedule = gemv.estimate_gemv_schedule(5, 3, 4, output_bytes=1)
        for key in ("operand_vector_count", "accepted_beats", "ideal_compute_cycles", "phase_compute_cycles"):
            self.assertEqual(schedule[key], 6)
        self.assertEqual(schedule["MAC_count"], 15)
        self.assertEqual(schedule["operand_load_bytes"], 30)
        self.assertEqual(schedule["A_stream_bytes"], 6)
        self.assertEqual(schedule["B_stream_bytes"], 24)
        self.assertEqual(schedule["parameter_stream_bytes"], 44)
        self.assertEqual(schedule["output_store_bytes"], 5)
        for n, k in ((0, 1), (1, 0), (65536, 1), (1, 65536)):
            with self.assertRaises(ValueError):
                gemv.estimate_gemv_schedule(n, k)

    def test_operand_order_replays_scalar_and_zero_pads_weight_tail(self):
        gemv = self.helpers()
        self.assertEqual(list(gemv.operand_beats([[-1, 2]], [[1, 2, 3], [-4, 5, 6]], 2)),
                         [(-1, [1, 2]), (2, [-4, 5]), (-1, [3, 0]), (2, [6, 0])])
        for a, b in (([[1], [2]], [[3]]), ([[1, 2]], [[3]])):
            with self.assertRaises(ValueError):
                list(gemv.operand_beats(a, b))

    def test_backend_defaults_preserve_gemm_positional_api(self):
        default = mapper.map_model(tiny_model(), [[1, -1]], 2)
        explicit = mapper.map_model(tiny_model(), [[1, -1]], 2, 16, backend="gemm")
        self.assertEqual(default, explicit)
        self.assertEqual(default["report"]["backend"], "gemm")
        self.assertEqual(default["report"]["physical_mac_count"], 4)
        self.assertEqual(default["report"]["k_tile"], 16)

    def test_gemv_keeps_compilation_and_numeric_results_identical(self):
        gemm = mapper.map_model(tiny_model(), [[1, -1]], 2, 1)
        gemv = mapper.map_model(tiny_model(), [[1, -1]], 2, backend="gemv")
        self.assertEqual(gemv["compiled"], gemm["compiled"])
        self.assertEqual(gemv["layers"], gemm["layers"])
        report = gemv["report"]
        self.assertEqual(report["backend"], "gemv")
        self.assertEqual(report["physical_mac_count"], 2)
        self.assertIsNone(report["k_tile"])
        self.assertEqual(report["operand_stream_bytes"], 21)
        self.assertEqual(report["MAC_count"], 9)
        self.assertEqual(report["host_activation_peak_bytes"], 5)
        self.assertEqual(report["layers"][0]["A_stream_bytes"], 4)
        self.assertEqual(report["layers"][0]["B_stream_bytes"], 8)

    def test_gemv_rejects_batch_and_explicit_k_tile(self):
        for kwargs, samples in (({"backend": "gemv"}, [[1, -1], [2, 3]]),
                                ({"backend": "gemv", "k_tile": 16}, [[1, -1]]),
                                ({"backend": "other"}, [[1, -1]])):
            with self.subTest(kwargs=kwargs, samples=samples), self.assertRaises(ValueError):
                mapper.map_model(tiny_model(), samples, **kwargs)

    def test_export_version_packing_parameters_and_determinism(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = mapper.export_bundle(tiny_model(), [[1, -1]], root, 2, backend="gemv")
            self.assertEqual((root / "jobs.txt").read_text().splitlines(),
                             ["gemv-v1 2 2", "1 3 2 1", "1 1 3 0"])
            self.assertEqual((root / "operands.txt").read_text().splitlines()[:4],
                             ["02 fe02", "fe 0100", "02 0000", "fe 0002"])
            params = [line.split()[:2] for line in (root / "parameters.txt").read_text().splitlines()]
            self.assertEqual(params, [["0", "0"], ["0", "1"], ["0", "2"], ["0", "0"]])
            self.assertEqual(report, json.loads((root / "report.json").read_text()))
            mapper.export_bundle(tiny_model(), [[1, -1]], root / "again", 2, backend="gemv")
            for name in ("mapped-model.json", "operands.txt", "parameters.txt", "jobs.txt", "expected.txt", "inputs.txt", "report.json"):
                self.assertEqual((root / name).read_bytes(), (root / "again" / name).read_bytes())

    def test_cli_backend_and_k_tile_validation_clear_prior_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "model.json").write_text(json.dumps(tiny_model()), encoding="utf-8")
            (root / "input.json").write_text("[[1, -1]]", encoding="utf-8")
            command = [sys.executable, str(ROOT / "tools/npu_mapper.py"), "--model", str(root / "model.json"),
                       "--inputs", str(root / "input.json"), "--output-dir", str(root / "out"), "--backend", "gemv"]
            good = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(good.returncode, 0, good.stderr)
            self.assertEqual(json.loads((root / "out/report.json").read_text())["backend"], "gemv")
            bad = subprocess.run(command + ["--k-tile", "16"], capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("k_tile", bad.stderr)
            self.assertFalse((root / "out/report.json").exists())

    def test_result_verification_requires_order_and_gemv_mac_capacity(self):
        from npu_mapper_sim import verify_results
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = mapper.export_bundle(tiny_model(), [[1, -1]], root, 2, backend="gemv")
            (root / "rtl-results.txt").write_text("0 0 0 4\n0 0 1 0\n0 0 2 0\n1 0 0 5\n")
            (root / "rtl-cycles.txt").write_text("0 1 3 2 20 4 4 0 0 0 4 6\n1 1 1 3 10 3 3 0 0 0 3 3\n")
            verified = verify_results(report, root)
            self.assertEqual(verified["layers"][0]["rtl"]["useful_array_utilization"], 6/40)
            # Optional capacity metadata must use the same backend-aware default
            # during validation and measurement, including standalone verification.
            report.pop("physical_mac_count")
            verified = verify_results(report, root)
            self.assertEqual(verified["layers"][0]["rtl"]["useful_array_utilization"], 6/40)
            (root / "rtl-results.txt").write_text("0 0 1 0\n0 0 0 4\n0 0 2 0\n1 0 0 5\n")
            with self.assertRaisesRegex(RuntimeError, "order"):
                verify_results(report, root)

    def test_actual_rtl_supports_both_backends_and_stall_modes(self):
        from hdl_tools import find_tool
        from npu_mapper_sim import simulate_bundle
        try:
            find_tool("iverilog");find_tool("vvp")
        except RuntimeError as error:
            self.skipTest(str(error))
        if not any((ROOT / ("rtl/npu/npu_gemv_linear_top" + suffix)).exists() for suffix in (".v", ".sv")):
            self.skipTest("GEMV RTL integration is not available yet")
        with tempfile.TemporaryDirectory() as folder:
            for backend in ("gemm", "gemv"):
                for stalls in (False, True):
                    with self.subTest(backend=backend, stalls=stalls):
                        root = Path(folder) / (backend + str(stalls))
                        mapper.export_bundle(tiny_model(), [[1, -1]], root, 2, backend=backend)
                        report = simulate_bundle(root, stalls=stalls)
                        self.assertEqual(report["rtl_validation"]["status"], "passed")
                        self.assertEqual(report["rtl_validation"]["stalls"], stalls)
                        if backend == "gemv":
                            for layer in report["layers"]:
                                rtl = layer["rtl"]
                                self.assertEqual(rtl["overlap_cycles"], layer["schedule"]["accepted_beats"])
                                self.assertEqual(rtl["buffer_stall_cycles"], 0)
                                if not stalls:
                                    self.assertEqual(rtl["input_wait_cycles"], 0)
                        text = (root / "operands.txt").read_text().splitlines()
                        text[0] = "00 0000" if backend == "gemv" else "0000 0000"
                        (root / "operands.txt").write_text("\n".join(text) + "\n")
                        with self.assertRaises(RuntimeError):
                            simulate_bundle(root, stalls=stalls)
                        failed = json.loads((root / "report.json").read_text())
                        self.assertEqual(failed["rtl_validation"]["status"], "failed")
                        self.assertTrue(all("rtl" not in layer for layer in failed["layers"]))

    def test_no_stalls_harness_drives_consecutive_gemm_operands(self):
        from hdl_tools import find_tool
        from npu_mapper_sim import simulate_bundle
        try:
            find_tool("iverilog");find_tool("vvp")
        except RuntimeError as error:
            self.skipTest(str(error))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            mapper.export_bundle(tiny_model(), [[1, -1]], root, 2)
            uninterrupted = simulate_bundle(root, stalls=False)
            self.assertEqual(sum(layer["rtl"]["input_wait_cycles"] for layer in uninterrupted["layers"]), 0)
            delayed = simulate_bundle(root)
            self.assertGreater(sum(layer["rtl"]["input_wait_cycles"] for layer in delayed["layers"]), 0)

    def test_overwide_scalar_is_rejected_instead_of_silently_truncated(self):
        from hdl_tools import find_tool
        from npu_mapper_sim import simulate_bundle
        try:
            find_tool("iverilog");find_tool("vvp")
        except RuntimeError as error:
            self.skipTest(str(error))
        if not any((ROOT / ("rtl/npu/npu_gemv_linear_top" + suffix)).exists() for suffix in (".v", ".sv")):
            self.skipTest("GEMV RTL integration is not available yet")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            mapper.export_bundle(tiny_model(), [[1, -1]], root, 2, backend="gemv")
            operands = (root / "operands.txt").read_text()
            (root / "operands.txt").write_text("1" + operands)
            with self.assertRaisesRegex(RuntimeError, "operand.*encoding"):
                simulate_bundle(root)

    def test_gemv_tail_lanes_are_independently_masked_by_hardware(self):
        from hdl_tools import find_tool
        from npu_mapper_sim import simulate_bundle
        try:
            find_tool("iverilog");find_tool("vvp")
        except RuntimeError as error:
            self.skipTest(str(error))
        if not any((ROOT / ("rtl/npu/npu_gemv_linear_top" + suffix)).exists() for suffix in (".v", ".sv")):
            self.skipTest("GEMV RTL integration is not available yet")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            mapper.export_bundle(tiny_model(), [[1, -1]], root, 2, backend="gemv")
            rows = (root / "operands.txt").read_text().splitlines()
            # First layer's last output group and the whole second layer have one real lane.
            for index in range(2, len(rows)):
                scalar, weights = rows[index].split()
                self.assertEqual(weights[:2], "00")
                rows[index] = scalar + " 7f" + weights[2:]
            (root / "operands.txt").write_text("\n".join(rows) + "\n")
            self.assertEqual(simulate_bundle(root, stalls=False)["rtl_validation"]["status"], "passed")

    def test_gemv_cli_simulation_and_extra_stream_tokens(self):
        from hdl_tools import find_tool
        from npu_mapper_sim import simulate_bundle
        try:
            find_tool("iverilog");find_tool("vvp")
        except RuntimeError as error:
            self.skipTest(str(error))
        if not any((ROOT / ("rtl/npu/npu_gemv_linear_top" + suffix)).exists() for suffix in (".v", ".sv")):
            self.skipTest("GEMV RTL integration is not available yet")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "model.json").write_text(json.dumps(tiny_model()), encoding="utf-8")
            (root / "input.json").write_text("[[1, -1]]", encoding="utf-8")
            result = subprocess.run([sys.executable, str(ROOT / "tools/npu_mapper.py"), "--model",
                str(root / "model.json"), "--inputs", str(root / "input.json"), "--output-dir", str(root / "out"),
                "--backend", "gemv", "--simulate"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((root / "out/report.json").read_text())["rtl_validation"]["status"], "passed")
            for name in ("jobs.txt", "inputs.txt", "operands.txt", "parameters.txt", "expected.txt"):
                with self.subTest(stream=name):
                    mapper.export_bundle(tiny_model(), [[1, -1]], root / "out", 2, backend="gemv")
                    with (root / "out" / name).open("a", encoding="ascii") as stream:
                        stream.write("123\n")
                    with self.assertRaisesRegex(RuntimeError, "extra tokens"):
                        simulate_bundle(root / "out")

    def test_bundle_decimal_fields_reject_overflow_aliases_before_narrowing(self):
        from hdl_tools import find_tool
        from npu_mapper_sim import simulate_bundle
        try:
            find_tool("iverilog");find_tool("vvp")
        except RuntimeError as error:
            self.skipTest(str(error))
        if not any((ROOT / ("rtl/npu/npu_gemv_linear_top" + suffix)).exists() for suffix in (".v", ".sv")):
            self.skipTest("GEMV RTL integration is not available yet")
        cases = [("input-positive-byte-alias", "inputs.txt", 0, 256, [[1, -1]]),
                 ("input-negative-byte-alias", "inputs.txt", 0, -256, [[-64, -1]]),
                 ("bias-int32-alias", "parameters.txt", 2, 1 << 32, [[1, -1]]),
                 ("multiplier-int32-alias", "parameters.txt", 3, 1 << 32, [[1, -1]]),
                 ("expected-int32-alias", "expected.txt", 0, 1 << 32, [[1, -1]]),
                 ("parameter-row-alias", "parameters.txt", 0, 1 << 32, [[1, -1]]),
                 ("parameter-column-alias", "parameters.txt", 1, 1 << 32, [[1, -1]]),
                 ("shift-alias", "parameters.txt", 4, 1 << 32, [[1, -1]]),
                 ("shape-m-alias", "jobs.txt", 3, 1 << 32, [[1, -1]]),
                 ("shape-n-alias", "jobs.txt", 4, 1 << 32, [[1, -1]]),
                 ("shape-k-alias", "jobs.txt", 5, 1 << 32, [[1, -1]]),
                 ("relu-alias", "jobs.txt", 6, 1 << 32, [[1, -1]]),
                 ("input-arbitrary-length", "inputs.txt", 0, 1 << 128, [[1, -1]])]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for backend in ("gemm", "gemv"):
                header_cases = [("header-layer-count", "jobs.txt", 0 if backend == "gemm" else 1,
                                 1 << 32, [[1, -1]]),
                                ("header-lanes", "jobs.txt", 1 if backend == "gemm" else 2,
                                 1 << 32, [[1, -1]])]
                if backend == "gemm":
                    header_cases.append(("header-k-tile", "jobs.txt", 2, 1 << 32, [[1, -1]]))
                for label, name, index, delta, samples in cases + header_cases:
                    with self.subTest(backend=backend, mutation=label):
                        mapper.export_bundle(tiny_model(), samples, root, 2, backend=backend)
                        tokens = (root / name).read_text().split()
                        tokens[index] = str(int(tokens[index]) + delta)
                        (root / name).write_text(" ".join(tokens) + "\n")
                        with self.assertRaisesRegex(RuntimeError, "decimal"):
                            simulate_bundle(root, stalls=False)
                        self.assertEqual(json.loads((root / "report.json").read_text())["rtl_validation"]["status"], "failed")

    def test_invalid_report_metadata_cannot_leave_stale_pass_or_running_status(self):
        from npu_mapper_sim import simulate_bundle
        mutations = [("batch-type", lambda report: report.update(batch="bad")),
                     ("batch-bool", lambda report: report.update(batch=True)),
                     ("layers-missing", lambda report: report.pop("layers")),
                     ("layers-none", lambda report: report.update(layers=None)),
                     ("layers-dict", lambda report: report.update(layers={})),
                     ("layers-empty", lambda report: report.update(layers=[])),
                     ("layer-not-dict", lambda report: report["layers"].__setitem__(0, "bad")),
                     ("shape-missing", lambda report: report["layers"][0].pop("shape")),
                     ("shape-dimension-type", lambda report: report["layers"][0]["shape"].update(n="bad")),
                     ("shape-dimension-zero", lambda report: report["layers"][0]["shape"].update(n=0)),
                     ("backend", lambda report: report.update(backend="bad")),
                     ("lanes-type", lambda report: report.update(lanes="bad")),
                     ("lanes-bool", lambda report: report.update(lanes=True)),
                     ("lanes-zero", lambda report: report.update(lanes=0)),
                     ("capacity-type", lambda report: report.update(physical_mac_count="bad")),
                     ("capacity-zero", lambda report: report.update(physical_mac_count=0)),
                     ("capacity-mismatch", lambda report: report.update(physical_mac_count=99)),
                     ("k-tile-type", lambda report: report.update(k_tile="bad")),
                     ("output-shape", lambda report: report["layers"][0].update(int8_output=[])),
                     ("output-value", lambda report: report["layers"][0]["int8_output"][0].__setitem__(0, 256)),
                     ("schedule-missing", lambda report: report["layers"][0].pop("schedule")),
                     ("schedule-type", lambda report: report["layers"][0]["schedule"].update(MAC_count="bad"))]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for backend in ("gemm", "gemv"):
                baseline = mapper.export_bundle(tiny_model(), [[1, -1]], root, 2, backend=backend)
                baseline["rtl_validation"] = {"status": "passed"}
                for layer in baseline["layers"]:
                    layer["rtl"] = {"integer_mismatches": 0}
                for label, mutation in mutations:
                    with self.subTest(backend=backend, mutation=label):
                        report = copy.deepcopy(baseline)
                        mutation(report)
                        (root / "report.json").write_text(json.dumps(report), encoding="utf-8")
                        with self.assertRaisesRegex(RuntimeError, "report"):
                            simulate_bundle(root, stalls=False)
                        if (root / "report.json").exists():
                            failed = json.loads((root / "report.json").read_text())
                            self.assertEqual(failed["rtl_validation"]["status"], "failed")
                            if isinstance(failed.get("layers"), list):
                                self.assertTrue(all("rtl" not in layer for layer in failed["layers"] if isinstance(layer, dict)))
            for encoded in ("{broken JSON", "[]", "null"):
                with self.subTest(report=encoded):
                    (root / "report.json").write_text(encoded, encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, "report"):
                        simulate_bundle(root)
                    self.assertFalse((root / "report.json").exists())

    def test_decimal_parser_preserves_valid_signed_boundaries_and_legacy_reports(self):
        from hdl_tools import find_tool
        from npu_mapper_sim import simulate_bundle
        try:
            find_tool("iverilog");find_tool("vvp")
        except RuntimeError as error:
            self.skipTest(str(error))
        if not any((ROOT / ("rtl/npu/npu_gemv_linear_top" + suffix)).exists() for suffix in (".v", ".sv")):
            self.skipTest("GEMV RTL integration is not available yet")
        source = {"format": "xilinx-gpu.linear.v1", "input_scale": 1, "layers": [
            {"name": "boundaries", "weights": [[1, 1]], "weight_scales": [1, 1],
             "bias": [-(1 << 31), (1 << 31)-1], "output_scale": 1/((1 << 31)-1)}]}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for backend in ("gemm", "gemv"):
                with self.subTest(backend=backend):
                    report = mapper.export_bundle(source, [[-128]], root, 2, backend=backend)
                    self.assertEqual(report["layers"][0]["int8_output"], [[-128, 127]])
                    if backend == "gemm":
                        report.pop("backend");report.pop("physical_mac_count")
                        (root / "report.json").write_text(json.dumps(report), encoding="utf-8")
                    self.assertEqual(simulate_bundle(root, stalls=False)["rtl_validation"]["status"], "passed")


if __name__ == "__main__":
    unittest.main()
