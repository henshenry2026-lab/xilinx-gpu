"""Bit-exact integer reference checks for the NPU, independent of RTL scheduling."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("npu_model", ROOT / "tools/npu_model.py")
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


class NpuModelTests(unittest.TestCase):
    def test_signed_and_extreme_products(self):
        self.assertEqual(model.matmul_int8([[-128, 127]], [[-128], [127]]), [[32513]])
        self.assertEqual(model.matmul_int8([[-128]], [[127]]), [[-16256]])
        self.assertEqual(model.matmul_int8([[-128] * 64], [[-128] for _ in range(64)]), [[1048576]])

    def test_nonsquare_matrix_and_zero(self):
        self.assertEqual(model.matmul_int8([[1, -2, 3], [4, 5, -6]], [[7, 8], [9, -10], [11, 12]]),
                         [[22, 64], [7, -90]])
        self.assertEqual(model.matmul_int8([[0, 0]], [[127], [-128]]), [[0]])

    def test_rejects_invalid_shapes_and_values(self):
        for a, b in [([], [[1]]), ([[]], [[1]]), ([[1], [1, 2]], [[1]]),
                     ([[1, 2]], [[1]]), ([[128]], [[1]]), ([[-129]], [[1]]),
                     ([[1.5]], [[1]]), ([[True]], [[1]]), ([[1]], [[256]])]:
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                model.matmul_int8(a, b)

    def test_fixed_stride_write_addresses(self):
        writes = model.pack_writes([[1, -2, 3]], [[4, 5], [6, 7], [8, -9]])
        self.assertEqual(writes, [(0, 0, 1), (0, 1, 254), (0, 2, 3),
                                 (1, 0, 4), (1, 1, 5), (1, 16, 6), (1, 17, 7),
                                 (1, 32, 8), (1, 33, 247)])
        with self.assertRaises(ValueError):
            model.pack_writes([[0] * 65], [[0] for _ in range(65)])

    def test_reproducible_vectors_include_tail_tiles(self):
        cases = model.validation_cases()
        self.assertEqual(cases, model.validation_cases())
        self.assertGreaterEqual(len(cases), 24)
        self.assertTrue(any(len(a) % 4 and len(b[0]) % 4 for a, b in cases))
        for a, b in cases:
            self.assertEqual(len(model.matmul_int8(a, b)), len(a))

    def test_linear_rounds_ties_away_from_zero(self):
        self.assertEqual(model.linear_int8([[ -5, -3, -1, 1, 3, 5]], [0] * 6,
                                           [1] * 6, [1] * 6),
                         [[-3, -2, -1, 1, 2, 3]])

    def test_linear_bias_saturation_and_relu(self):
        accumulators = [[-100, -1, 10, 127, 0]]
        bias = [0, -200, 200, 0, -(1 << 31)]
        multipliers = [1] * 5
        shifts = [0] * 5
        self.assertEqual(model.linear_int8(accumulators, bias, multipliers, shifts),
                         [[-100, -128, 127, 127, -128]])
        self.assertEqual(model.linear_int8(accumulators, bias, multipliers, shifts, relu=True),
                         [[0, 0, 127, 127, 0]])

    def test_linear_large_multiplier_and_maximum_shift(self):
        self.assertEqual(model.linear_int8([[-1, 1]], [0, 0],
                                           [(1 << 31) - 1] * 2, [31, 31]), [[-1, 1]])

    def test_linear_validation(self):
        invalid = [
            ([[1]], [], [1], [0]),
            ([[1]], [0], [0], [0]),
            ([[1]], [0], [1 << 31], [0]),
            ([[1]], [0], [1], [32]),
            ([[1 << 31]], [0], [1], [0]),
            ([[1.5]], [0], [1], [0]),
        ]
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(ValueError):
                model.linear_int8(*args)


if __name__ == "__main__":
    unittest.main()
