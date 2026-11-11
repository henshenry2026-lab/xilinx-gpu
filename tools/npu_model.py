"""Integer GEMM reference and deterministic RTL vectors; no ML framework required."""
import argparse
import json
from pathlib import Path
import random


def _shape(matrix):
    if not matrix or not matrix[0]:
        raise ValueError("matrix must be nonempty")
    cols = len(matrix[0])
    for row in matrix:
        if len(row) != cols or any(type(value) is not int or not -128 <= value <= 127 for value in row):
            raise ValueError("matrix must be rectangular and contain signed INT8 integers")
    return len(matrix), cols


def matmul_int8(a, b):
    """Exact integer C=A*B, without scaling, rounding, bias or activation."""
    m, inner = _shape(a)
    b_rows, n = _shape(b)
    if inner != b_rows:
        raise ValueError("A columns must equal B rows")
    return [[sum(a[r][k] * b[k][c] for k in range(inner)) for c in range(n)] for r in range(m)]


def linear_int8(accumulators, bias, multipliers, shifts, relu=False):
    """Apply per-channel bias, ties-away fixed-point scaling, ReLU and INT8 saturation."""
    if not accumulators or not accumulators[0]:
        raise ValueError("accumulator matrix must be nonempty")
    cols = len(accumulators[0])
    for row in accumulators:
        if len(row) != cols or any(type(value) is not int or not -(1 << 31) <= value < (1 << 31)
                                   for value in row):
            raise ValueError("accumulators must be rectangular signed INT32 integers")
    if not (len(bias) == len(multipliers) == len(shifts) == cols):
        raise ValueError("each output channel requires bias, multiplier and shift")
    if any(type(value) is not int or not -(1 << 31) <= value < (1 << 31) for value in bias):
        raise ValueError("bias values must be signed INT32 integers")
    if any(type(value) is not int or not 1 <= value <= (1 << 31) - 1 for value in multipliers):
        raise ValueError("multipliers must be positive signed INT32 integers")
    if any(type(value) is not int or not 0 <= value <= 31 for value in shifts):
        raise ValueError("shifts must be between 0 and 31")
    if type(relu) is not bool:
        raise ValueError("relu must be a boolean")

    output = []
    for row in accumulators:
        output_row = []
        for column, accumulator in enumerate(row):
            product = (accumulator + bias[column]) * multipliers[column]
            shift = shifts[column]
            if shift:
                magnitude = (abs(product) + (1 << (shift - 1))) >> shift
                scaled = -magnitude if product < 0 else magnitude
            else:
                scaled = product
            if relu and scaled < 0:
                scaled = 0
            output_row.append(max(-128, min(127, scaled)))
        output.append(output_row)
    return output


def pack_writes(a, b, max_m=16, max_n=16, max_k=64):
    """(buffer, byte-address, two's-complement byte), using the hardware's fixed strides."""
    m, inner = _shape(a)
    b_rows, n = _shape(b)
    if inner != b_rows or m > max_m or n > max_n or inner > max_k:
        raise ValueError("matrix shape exceeds the configured hardware buffers")
    return ([(0, r * max_k + k, a[r][k] & 255) for r in range(m) for k in range(inner)]
            + [(1, k * max_n + c, b[k][c] & 255) for k in range(inner) for c in range(n)])


def validation_cases():
    rng = random.Random(2395)
    cases = [([[-128]], [[-128]]), ([[-128]], [[127]]),
             ([[0, 0, 0]], [[127], [-128], [1]]),
             ([[-128] * 64 for _ in range(16)], [[-128] * 16 for _ in range(64)])]
    shapes = [(4, 4, 1), (7, 5, 9), (1, 16, 1), (16, 1, 64), (16, 16, 64)]
    shapes += [(rng.randint(1, 16), rng.randint(1, 16), rng.randint(1, 64)) for _ in range(20)]
    for m, n, inner in shapes:
        a = [[rng.randint(-128, 127) for _ in range(inner)] for _ in range(m)]
        b = [[rng.randint(-128, 127) for _ in range(n)] for _ in range(inner)]
        cases.append((a, b))
    return cases


def write_vectors(path):
    cases = validation_cases()
    lines = [str(len(cases))]
    for a, b in cases:
        lines.append(f"{len(a)} {len(b[0])} {len(b)}")
        for matrix in [a, b, matmul_int8(a, b)]:
            lines.extend(" ".join(map(str, row)) for row in matrix)
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")
    return len(cases)


def write_linear_vectors(path):
    """Generate complete GEMM + per-channel postprocessing jobs for RTL checking."""
    rng = random.Random(7419)
    cases = validation_cases()
    lines = [str(len(cases))]
    for case_index, (a, b) in enumerate(cases):
        columns = len(b[0])
        relu = bool(case_index % 2)
        bias = [rng.choice([rng.randint(-1500, 1500), 0, -(1 << 31), (1 << 31)-1])
                for _ in range(columns)]
        multipliers = [rng.choice([1, 3, 7, 127, (1 << 31)-1]) for _ in range(columns)]
        shifts = [(case_index + column) % 32 for column in range(columns)]
        result = linear_int8(matmul_int8(a, b), bias, multipliers, shifts, relu)
        lines.append(f"{len(a)} {columns} {len(b)} {int(relu)}")
        for matrix in [a, b]:
            lines.extend(" ".join(map(str, row)) for row in matrix)
        lines.extend(f"{bias[c]} {multipliers[c]} {shifts[c]}" for c in range(columns))
        lines.extend(" ".join(map(str, row)) for row in result)
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")
    return len(cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=Path, help="write deterministic simulation vectors")
    args = parser.parse_args()
    if args.vectors:
        args.vectors.parent.mkdir(parents=True, exist_ok=True)
        print(f"Wrote {write_vectors(args.vectors)} matrix cases to {args.vectors}")
    else:
        a, b = [[1, -2, 3], [4, 5, -6]], [[7, 8], [9, -10], [11, 12]]
        accumulated = matmul_int8(a, b)
        bias, multipliers, shifts = [2, -4], [3, 1], [2, 1]
        print(json.dumps({"A": a, "B": b, "C_int32": accumulated,
                          "linear_parameters": {"bias": bias, "multipliers": multipliers,
                                                "shifts": shifts, "relu": False},
                          "Y_int8": linear_int8(accumulated, bias, multipliers, shifts),
                          "writes": pack_writes(a, b)}, indent=2))


if __name__ == "__main__":
    main()
