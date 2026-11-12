"""Contracts for the bounded-memory streaming scheduler."""
import sys
from pathlib import Path
import unittest
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import npu_stream as stream
from npu_model import matmul_int8


class StreamTests(unittest.TestCase):
    def test_memory_is_independent_of_matrix_size(self):
        memory = stream.estimate_tile_memory()
        self.assertEqual(memory["A_bytes"], 128)
        self.assertEqual(memory["B_bytes"], 128)
        self.assertEqual(memory["C_bytes"], 64)
        self.assertEqual(memory["logical_working_set_bytes"], 320)

    def test_tail_schedule_and_transfer_accounting(self):
        schedule = stream.estimate_gemm_schedule(3, 5, 17)
        self.assertEqual(schedule["tile_count"], 2)
        self.assertEqual(schedule["k_chunk_count"], 4)
        self.assertEqual(schedule["MAC_count"], 255)
        self.assertEqual(schedule["operand_vector_count"], 34)
        self.assertEqual(schedule["operand_load_bytes"], 272)
        self.assertEqual(schedule["useful_operand_bytes"], 187)
        self.assertEqual(schedule["output_store_bytes"], 60)

    def test_chunked_reference_and_zero_padding(self):
        a = [[r + k - 5 for k in range(17)] for r in range(3)]
        b = [[k - c - 4 for c in range(5)] for k in range(17)]
        self.assertEqual(stream.stream_gemm_reference(a, b), matmul_int8(a, b))
        vectors = list(stream.operand_vectors(a, b))
        self.assertEqual(len(vectors), 34)
        self.assertEqual(vectors[0], ([a[r][0] for r in range(3)] + [0], b[0][:4]))
        self.assertEqual(vectors[17][1], [b[0][4], 0, 0, 0])

    def test_mlp_and_sweep(self):
        layers = stream.mlp_case()
        self.assertEqual(len(layers[0]["a"][0]), 16)
        self.assertEqual(len(layers[0]["expected"][0]), 32)
        self.assertEqual(layers[1]["a"], layers[0]["expected"])
        self.assertEqual(len(layers[1]["expected"][0]), 16)
        self.assertEqual(stream.estimate_mlp_schedule()["MAC_count"], 1024)
        self.assertTrue(stream.parameter_sweep())

    def test_invalid_configuration(self):
        for values in [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 65536)]:
            with self.assertRaises(ValueError):
                stream.estimate_gemm_schedule(*values)
        with self.assertRaises(ValueError):
            stream.estimate_tile_memory(lanes=0)

    def test_measurement_checks_reject_inconsistent_counters(self):
        from check import stream_measurements
        # 1x1 K1 on a 4-lane array: one load, one setup plus seven array steps.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cycles.txt"
            path.write_text("1 1 1 20 1 8 3 0 2 0 1 0\n", encoding="ascii")
            rows = stream_measurements(path, 4, 16, False)
            self.assertEqual(rows[0]["predicted"]["MAC_count"], 1)
            self.assertEqual(rows[0]["useful_MACs_per_cycle"], 0.05)
            for text in ["1 1 1 20 1 7 3 0 2 0 1 0\n",
                         "1 1 1 20 1 8 3 0 2 2 1 0\n", "", "1 1\n"]:
                path.write_text(text, encoding="ascii")
                with self.assertRaises(RuntimeError):
                    stream_measurements(path, 4, 16, False)


if __name__ == "__main__":
    unittest.main()
