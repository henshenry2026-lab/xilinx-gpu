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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=Path, help="write deterministic simulation vectors")
    args = parser.parse_args()
    if args.vectors:
        args.vectors.parent.mkdir(parents=True, exist_ok=True)
        print(f"Wrote {write_vectors(args.vectors)} matrix cases to {args.vectors}")
    else:
        a, b = [[1, -2, 3], [4, 5, -6]], [[7, 8], [9, -10], [11, 12]]
        print(json.dumps({"A": a, "B": b, "C_int32": matmul_int8(a, b),
                          "writes": pack_writes(a, b)}, indent=2))


if __name__ == "__main__":
    main()
