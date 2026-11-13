"""Numerical, layout and failure contracts for the model-to-stream mapper."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import npu_mapper as mapper


def tiny_model():
    return {"format": "xilinx-gpu.linear.v1", "input_scale": 0.5, "layers": [
        {"name": "hidden", "weights": [[1.0, -1.0, 0.0], [0.0, 0.5, 1.0]],
         "bias": [0.0, 0.25, -0.5], "weight_scales": [0.5, 0.5, 0.5],
         "output_scale": 0.25, "relu": True},
        {"name": "output", "weights": [[1.0], [-0.5], [0.5]], "bias": [0.25],
         "weight_scales": [0.5], "output_scale": 0.25, "relu": False}]}


class MapperTests(unittest.TestCase):
    def test_scale_encoding_and_unsupported_ranges(self):
        self.assertEqual(mapper.requantization_params(0.5), (1 << 30, 31))
        self.assertEqual(mapper.requantization_params(1.0), (1 << 30, 30))
        self.assertEqual(mapper.requantization_params((1 << 31)-1), ((1 << 31)-1, 0))
        for ratio in [0, -1, 1e-30, 1e30, math.nan, math.inf, True]:
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                mapper.requantization_params(ratio)

    def test_rounding_and_input_clipping_are_explicit(self):
        values, clipped = mapper.quantize_values([-1.5, -0.5, 0.5, 1.5, -200, 200], 1.0)
        self.assertEqual(values, [-2, -1, 1, 2, -128, 127])
        self.assertEqual(clipped, 2)

    def test_float_values_immediately_below_half_are_not_rounded_up(self):
        below = math.nextafter(0.5, 0.0)
        above = math.nextafter(0.5, 1.0)
        values, count = mapper.quantize_values([below, -below, 0.5, -0.5, above, -above], 1)
        self.assertEqual(values, [0, 0, 1, -1, 1, -1])
        self.assertEqual(count, 0)
        source = {"format": "xilinx-gpu.linear.v1", "input_scale": 1, "layers": [
            {"name": "bias", "weights": [[0]], "weight_scales": [1],
             "bias": [below], "output_scale": 1}]}
        self.assertEqual(mapper.compile_model(source)["layers"][0]["bias"], [0])

    def test_finite_inputs_clip_even_when_the_division_overflows(self):
        values, clipped = mapper.quantize_values([1e308, -1e308], 1e-308)
        self.assertEqual(values, [127, -128])
        self.assertEqual(clipped, 2)
        source = {"format": "xilinx-gpu.linear.v1", "input_scale": 1e-308, "layers": [
            {"name": "clip", "weights": [[1]], "weight_scales": [1],
             "bias": [0], "output_scale": 1e-308}]}
        plan = mapper.map_model(source, [[1e308]])
        self.assertEqual(plan["report"]["input_clipped_elements"], 1)
        self.assertEqual(plan["layers"][0]["outputs"], [[127]])

    def test_compile_bias_scales_and_zero_weight_channel(self):
        source = tiny_model()
        before = copy.deepcopy(source)
        compiled = mapper.compile_model(source)
        self.assertEqual(source, before)
        first = compiled["layers"][0]
        self.assertEqual(first["weights"], [[2, -2, 0], [0, 1, 2]])
        self.assertEqual(first["bias"], [0, 1, -2])
        self.assertEqual(first["multipliers"], [1 << 30]*3)
        self.assertEqual(first["shifts"], [30]*3)
        source["layers"][0].pop("weight_scales")
        source["layers"][0]["weights"] = [[1, 0, 0], [-1, 0, 0]]
        auto = mapper.compile_model(source)["layers"][0]
        self.assertEqual(auto["weight_scales"], [1/127, 1.0, 1.0])
        self.assertEqual([row[0] for row in auto["weights"]], [127, -127])

    def test_float_integer_paths_and_bounded_memory(self):
        plan = mapper.map_model(tiny_model(), [[1, -1], [-1, 2], [0.5, 0.5]], lanes=2, k_tile=1)
        self.assertEqual(plan["layers"][0]["outputs"], [[4, 0, 0], [0, 9, 6], [2, 0, 0]])
        self.assertEqual(plan["layers"][1]["inputs"], plan["layers"][0]["outputs"])
        self.assertEqual(plan["layers"][1]["outputs"], [[5], [-1], [3]])
        report = plan["report"]
        self.assertEqual(report["host_activation_peak_bytes"], 15)
        self.assertEqual(report["MAC_count"], 27)
        self.assertEqual(report["tile_memory"]["logical_working_set_bytes"], 24)
        self.assertEqual(report["rtl_validation"]["status"], "not_run")
        self.assertEqual(report["layers"][0]["error"]["max_abs"], 0.0)

    def test_saturation_is_distinct_from_merely_hitting_int8_bounds(self):
        model = {"format": "xilinx-gpu.linear.v1", "input_scale": 1, "layers": [
            {"name": "clip", "weights": [[2, -2]], "weight_scales": [1, 1],
             "bias": [0, 0], "output_scale": 1}]}
        report = mapper.map_model(model, [[200]])["report"]
        self.assertEqual(report["input_clipped_elements"], 1)
        self.assertEqual(report["layers"][0]["saturated_elements"], 2)
        self.assertEqual(report["layers"][0]["error"]["max_abs"], 273)
        model["layers"][0]["weights"] = [[200, -200]]
        report = mapper.map_model(model, [[1]])["report"]
        self.assertEqual(report["layers"][0]["weight_clipped_elements"], 2)
        self.assertEqual(report["layers"][0]["saturated_elements"], 0)
        self.assertEqual(report["layers"][0]["at_int8_bounds"], 2)

    def test_export_uses_actual_tile_order_and_lane_packing(self):
        source = tiny_model()
        samples = [[1, -1], [-1, 2], [0.5, 0.5]]
        with tempfile.TemporaryDirectory() as folder:
            report = mapper.export_bundle(source, samples, folder, lanes=2, k_tile=1)
            root = Path(folder)
            self.assertEqual((root/"jobs.txt").read_text().splitlines(), ["2 2 1", "3 3 2 1", "3 1 3 0"])
            operands = (root/"operands.txt").read_text().splitlines()
            self.assertEqual(operands[0], "fe02 fe02")
            self.assertEqual(operands[1], "04fe 0100")
            parameters = [list(map(int, line.split())) for line in (root/"parameters.txt").read_text().splitlines()]
            self.assertEqual([row[:2] for row in parameters[:9]],
                             [[0,0],[0,1],[1,0],[1,1],[0,2],[1,2],[2,0],[2,1],[2,2]])
            self.assertEqual(report, json.loads((root/"report.json").read_text()))
            again = root/"again"
            mapper.export_bundle(source, samples, again, lanes=2, k_tile=1)
            for name in ["mapped-model.json", "operands.txt", "parameters.txt", "jobs.txt", "expected.txt"]:
                self.assertEqual((root/name).read_bytes(), (again/name).read_bytes())

    def test_model_rejects_invalid_values_and_unsupported_features(self):
        invalid = []
        for key, value in [("input_scale", 0), ("input_scale", math.nan), ("zero_point", 1)]:
            source = tiny_model();source[key] = value;invalid.append(source)
        for key, value in [("weights", [[1], [0,1]]), ("weights", [[math.inf]*3]*2),
                           ("bias", [0]), ("bias", [1e50]*3), ("relu", "false"),
                           ("weight_scales", [0]*3), ("activation", "gelu"),
                           ("output_scale", 1e300), ("name", "")]:
            source = tiny_model();source["layers"][0][key] = value;invalid.append(source)
        source = tiny_model();source["layers"][1]["weights"] = [[1.0]];invalid.append(source)
        source = tiny_model();source["layers"][1]["name"] = "hidden";invalid.append(source)
        for source in invalid:
            with self.subTest(source=source), self.assertRaises(ValueError):
                mapper.compile_model(source)
        for samples in [[], [[1]], [[True, 1]], [[math.nan, 0]]]:
            with self.assertRaises(ValueError):
                mapper.map_model(tiny_model(), samples)

    def test_cli_fails_without_leaving_a_success_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"model.json").write_text(json.dumps(tiny_model()), encoding="utf-8")
            (root/"input.json").write_text("[[1, -1]]", encoding="utf-8")
            command = [sys.executable, str(ROOT/"tools/npu_mapper.py"), "--model", str(root/"model.json"),
                       "--inputs", str(root/"input.json"), "--output-dir", str(root/"out")]
            good = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(good.returncode, 0, good.stderr)
            (root/"input.json").write_text("[[1]]", encoding="utf-8")
            bad = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertFalse((root/"out/report.json").exists())

    def test_exported_streams_run_on_rtl_and_validation_is_not_stale(self):
        from npu_mapper_sim import simulate_bundle
        from hdl_tools import find_tool
        try:
            find_tool("iverilog");find_tool("vvp")
        except RuntimeError as error:
            self.skipTest(str(error))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/"bundle with spaces"
            mapper.export_bundle(tiny_model(), [[1, -1], [-1, 2], [0.5, 0.5]], root, 2, 1)
            report = simulate_bundle(root)
            self.assertEqual(report["rtl_validation"]["status"], "passed")
            self.assertTrue(all(layer["rtl"]["integer_mismatches"] == 0 for layer in report["layers"]))
            self.assertTrue(all(layer["rtl"]["cycles"] > 0 for layer in report["layers"]))
            # Breaking the exported order must fail, and clear prior passing state.
            operands = (root/"operands.txt").read_text().splitlines()
            operands[0] = "0000 0000"
            (root/"operands.txt").write_text("\n".join(operands)+"\n", encoding="ascii")
            with self.assertRaises(RuntimeError):
                simulate_bundle(root)
            self.assertEqual(json.loads((root/"report.json").read_text())["rtl_validation"]["status"], "failed")
            mapper.export_bundle(tiny_model(), [[1, -1]], root, 2, 1)
            with (root/"parameters.txt").open("a", encoding="ascii") as stream:
                stream.write("123\n")
            with self.assertRaisesRegex(RuntimeError, "extra tokens"):
                simulate_bundle(root)


if __name__ == "__main__":
    unittest.main()
