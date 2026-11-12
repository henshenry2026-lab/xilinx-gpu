"""Bounded-memory streaming schedules, integer references and measurement budgets."""
import argparse
import json
from pathlib import Path
import random

from npu_model import _shape, matmul_int8, linear_int8


def _positive(value, maximum=65535):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"expected integer in 1..{maximum}")


def estimate_tile_memory(lanes=4, k_tile=16):
    _positive(lanes, 16)
    _positive(k_tile)
    a = b = 2 * lanes * k_tile
    c = lanes * lanes * 4
    return {"lanes": lanes, "k_tile": k_tile, "A_bytes": a, "B_bytes": b, "C_bytes": c,
            "logical_working_set_bytes": a + b + c,
            "bias_scale_table_bytes": 0, "parameter_stream_bits_per_result": 70,
            "linear_output_register_bits": 41,
            "array_forward_and_skew_bits": 18 * lanes * lanes + 9 * lanes * (lanes-1),
            "note": "A/B/C payload only; pipeline, control, counters and physical allocation reported separately"}


def estimate_gemm_schedule(m, n, k, lanes=4, k_tile=16, tiles=1, output_bytes=4):
    for value in [m, n, k, tiles]:
        _positive(value)
    estimate_tile_memory(lanes, k_tile)
    if output_bytes not in [1, 4]:
        raise ValueError("output_bytes must be 1 or 4")
    mr, nr, kr = ((m+lanes-1)//lanes, (n+lanes-1)//lanes, (k+k_tile-1)//k_tile)
    count = mr * nr
    vectors = count * k
    chunks = count * kr
    return {"tile_count": count, "k_chunk_count": chunks, "MAC_count": m*n*k,
            "operand_vector_count": vectors, "operand_load_bytes": vectors*2*lanes,
            "useful_operand_bytes": k*(m*nr+n*mr),
            "output_store_bytes": m*n*output_bytes,
            "parameter_stream_bytes": (m*n*70+7)//8 if output_bytes == 1 else 0,
            "ideal_compute_cycles": (vectors+tiles-1)//tiles,
            "serialized_output_cycles": m*n,
            "array_step_cycles": vectors+chunks*2*(lanes-1),
            "phase_compute_cycles": vectors+chunks*(2*(lanes-1)+1),
            "description": "Ideal lower bound; each chunk includes a setup clock and array drain; no assumed clock rate"}


def operand_vectors(a, b, lanes=4):
    m, k = _shape(a)
    kb, n = _shape(b)
    if k != kb:
        raise ValueError("incompatible matrix dimensions")
    _positive(lanes, 16)
    for row in range(0, m, lanes):
        for col in range(0, n, lanes):
            for inner in range(k):
                yield ([a[row+r][inner] if row+r < m else 0 for r in range(lanes)],
                       [b[inner][col+c] if col+c < n else 0 for c in range(lanes)])


def stream_gemm_reference(a, b, lanes=4, k_tile=16):
    m, k = _shape(a)
    kb, n = _shape(b)
    if kb != k:
        raise ValueError("incompatible matrix dimensions")
    estimate_gemm_schedule(m, n, k, lanes, k_tile)
    c = [[0]*n for _ in range(m)]
    for row in range(0, m, lanes):
        for col in range(0, n, lanes):
            for start in range(0, k, k_tile):
                for r in range(row, min(row+lanes, m)):
                    for s in range(col, min(col+lanes, n)):
                        c[r][s] += sum(a[r][t]*b[t][s] for t in range(start, min(start+k_tile, k)))
    return c


linear_reference = linear_int8


def mlp_case():
    rng = random.Random(31916)
    a = [[rng.randint(-16, 16) for _ in range(16)]]
    layers = []
    for k, n, relu in [(16, 32, True), (32, 16, False)]:
        b = [[rng.randint(-8, 8) for _ in range(n)] for _ in range(k)]
        bias = [rng.randint(-80, 80) for _ in range(n)]
        multipliers, shifts = [3]*n, [6]*n
        raw = matmul_int8(a, b)
        expected = linear_int8(raw, bias, multipliers, shifts, relu)
        layers.append(dict(a=a, b=b, bias=bias, multipliers=multipliers, shifts=shifts,
                           relu=relu, expected=expected, raw=raw))
        a = expected
    return layers


def estimate_mlp_schedule(lanes=4, k_tile=16):
    layers = [estimate_gemm_schedule(1, n, k, lanes, k_tile, output_bytes=1)
              for k, n in [(16, 32), (32, 16)]]
    return {"layers": layers, "MAC_count": sum(x["MAC_count"] for x in layers),
            "operand_load_bytes": sum(x["operand_load_bytes"] for x in layers),
            "host_activation_peak_bytes": 48, "hardware_layer_scheduler": False}


def parameter_sweep():
    return [{"memory": estimate_tile_memory(lanes, chunk), "tiles": tiles,
             "mlp_layer1": estimate_gemm_schedule(1, 32, 16, lanes, chunk, tiles)}
            for lanes in [2, 4, 8] for chunk in [8, 16, 32] for tiles in [1, 2, 4]]


def write_cases(path):
    rng = random.Random(161664)
    shapes = [(1, 1, k) for k in [1, 15, 16, 17, 32, 64, 65, 257]]
    shapes += [(3, 5, 17), (4, 4, 16), (5, 7, 32), (1, 32, 16)]
    shapes += [(rng.randint(1, 8), rng.randint(1, 12), rng.randint(1, 90)) for _ in range(12)]
    cases = []
    for i, (m, n, k) in enumerate(shapes):
        a = [[rng.randint(-128, 127) for _ in range(k)] for _ in range(m)]
        b = [[rng.randint(-128, 127) for _ in range(n)] for _ in range(k)]
        if i == 7:
            a, b = [[-128]*k], [[-128] for _ in range(k)]
        raw = matmul_int8(a, b)
        bias = [rng.choice([0, rng.randint(-1500, 1500), -(1 << 31), (1 << 31)-1]) for _ in range(n)]
        multipliers = [rng.choice([1, 3, 127, (1 << 31)-1]) for _ in range(n)]
        shifts = [(i+c) % 32 for c in range(n)]
        relu = bool(i % 2)
        cases.append(dict(a=a, b=b, raw=raw, bias=bias, multipliers=multipliers, shifts=shifts,
                          relu=relu, expected=linear_int8(raw, bias, multipliers, shifts, relu), mlp=0))
    # Unsaturated half-way results and signed extremes exercise the new stream
    # wrapper directly; random large products would otherwise mostly saturate.
    directed = [([-3, -1, 1, 3], [0]*4, [1]*4, [1]*4, False),
                ([-100, -1, 10, 127], [0, -200, 200, 0], [1]*4, [0]*4, False),
                ([-5, -1, 1, 5], [0]*4, [1]*4, [0]*4, True),
                ([-128, 127], [-(1 << 31), (1 << 31)-1], [1]*2, [0]*2, False),
                ([-1, 1], [0]*2, [(1 << 31)-1]*2, [31]*2, False)]
    for values, bias, multipliers, shifts, relu in directed:
        raw = [values]
        cases.append(dict(a=[[1]], b=[values], raw=raw, bias=bias, multipliers=multipliers,
                          shifts=shifts, relu=relu,
                          expected=linear_int8(raw, bias, multipliers, shifts, relu), mlp=0))
    for i, layer in enumerate(mlp_case()):
        cases.append(dict(layer, mlp=i+1))
    lines = [str(len(cases))]
    for case in cases:
        a, b = case["a"], case["b"]
        lines.append(f"{len(a)} {len(b[0])} {len(b)} {int(case['relu'])} {case['mlp']}")
        for matrix in [a, b]:
            lines.extend(" ".join(map(str, row)) for row in matrix)
        lines.extend(f"{v} {p} {s}" for v, p, s in zip(case["bias"], case["multipliers"], case["shifts"]))
        for matrix in [case["raw"], case["expected"]]:
            lines.extend(" ".join(map(str, row)) for row in matrix)
    Path(path).write_text("\n".join(lines)+"\n", encoding="ascii")
    return len(cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lanes", type=int, default=4)
    parser.add_argument("--k-tile", type=int, default=16)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()
    report = parameter_sweep() if args.sweep else {
        "memory": estimate_tile_memory(args.lanes, args.k_tile),
        "mlp": estimate_mlp_schedule(args.lanes, args.k_tile)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
