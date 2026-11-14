"""Acceptance gates must reject incorrect or regressed GEMV evidence."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import check


def evidence():
    resources = {f"gemv_{kind}_{lanes}": {
        "dsp48e1": lanes + (4 if kind == "linear" else 0),
        "ramb18e1": 0, "ramb36e1": 0, "distributed_ram_primitives": {}}
        for lanes in (1, 2, 4) for kind in ("raw", "linear")}
    resources["stream_linear_2"] = {"dsp48e1": 8}
    measurements = {}
    for case in ("mlp_batch1", "tail_batch1"):
        for mode in ("continuous", "stalled"):
            for backend, lanes in (("gemm", 2), ("gemm", 4), ("gemv", 1), ("gemv", 2), ("gemv", 4)):
                measurements[f"{case}_{mode}_{backend}_l{lanes}"] = {
                    "backend": backend, "lanes": lanes, "batch": 1,
                    "MAC_count": 8, "operand_stream_bytes": 16,
                    "rtl_validation": {"status": "passed"},
                    "layers": [{"name": "layer", "shape": {"m": 1, "n": 2, "k": 4},
                                "int8_output": [[1, -2]], "rtl": {"cycles": 20 if backend == "gemm" else 15}}]}
    return resources, measurements


class GemvAcceptanceTests(unittest.TestCase):
    def test_complete_matching_evidence_passes(self):
        self.assertTrue(hasattr(check, "gemv_acceptance"), "GEMV acceptance gate is missing")
        resources, measured = evidence()
        result = check.gemv_acceptance(resources, measured)
        self.assertEqual(len(result["configurations"]), 20)
        self.assertEqual(result["integer_mismatches"], 0)

    def test_resource_cycle_and_numerical_regressions_fail(self):
        self.assertTrue(hasattr(check, "gemv_acceptance"), "GEMV acceptance gate is missing")
        resources, measured = evidence()
        corruptions = [
            ("resource", lambda r, m: r["gemv_raw_4"].update(dsp48e1=16)),
            ("memory", lambda r, m: r["gemv_raw_2"].update(distributed_ram_primitives={"RAM32M": 1})),
            ("cycle", lambda r, m: m["mlp_batch1_continuous_gemv_l4"]["layers"][0]["rtl"].update(cycles=21)),
            ("numerical", lambda r, m: m["tail_batch1_stalled_gemv_l2"]["layers"][0].update(int8_output=[[1, 2]])),
            ("stale", lambda r, m: m["mlp_batch1_continuous_gemm_l4"]["rtl_validation"].update(status="failed")),
            ("missing", lambda r, m: m.pop("tail_batch1_stalled_gemv_l1")),
        ]
        for label, corrupt in corruptions:
            with self.subTest(label=label):
                r, m = copy.deepcopy(resources), copy.deepcopy(measured)
                corrupt(r, m)
                with self.assertRaises(RuntimeError):
                    check.gemv_acceptance(r, m)


if __name__ == "__main__":
    unittest.main()
