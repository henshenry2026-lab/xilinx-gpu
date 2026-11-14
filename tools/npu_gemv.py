"""Single-batch INT8 GEMV streams with no complete input or weight cache."""
from npu_model import _shape
from npu_stream import _positive


def estimate_gemv_memory(lanes=4):
    """Logical partial-sum payload, independent of N and K."""
    _positive(lanes, 16)
    return {"lanes": lanes, "A_bytes": 0, "B_bytes": 0, "C_bytes": 4*lanes,
            "logical_working_set_bytes": 4*lanes, "bias_scale_table_bytes": 0,
            "parameter_stream_bits_per_result": 70, "linear_output_register_bits": 41,
            "note": "partial-sum payload only; no complete input or weights stored; excludes pipeline, control, counters and physical allocation"}


def estimate_gemv_schedule(n, k, lanes=4, output_bytes=4):
    """One replayed activation scalar and LANES weights per accepted beat."""
    _positive(n);_positive(k)
    estimate_gemv_memory(lanes)
    if output_bytes not in (1, 4):
        raise ValueError("output_bytes must be 1 or 4")
    groups = (n+lanes-1)//lanes
    beats = groups*k
    return {"tile_count": groups, "MAC_count": n*k, "accepted_beats": beats,
            "operand_vector_count": beats, "operand_load_bytes": beats*(1+lanes),
            "A_stream_bytes": beats, "B_stream_bytes": beats*lanes,
            "useful_operand_bytes": beats+n*k, "output_store_bytes": n*output_bytes,
            "parameter_stream_bytes": (n*70+7)//8 if output_bytes == 1 else 0,
            "ideal_compute_cycles": beats, "phase_compute_cycles": beats,
            "serialized_output_cycles": n,
            "description": "Accepted-beat compute budget, output-group then K order; includes padded weight lanes and activation replay; no assumed clock rate"}


def operand_beats(a, b, lanes=4):
    """Yield (scalar, packed-lane values) in output-group then K order."""
    m, k = _shape(a)
    kb, n = _shape(b)
    if m != 1:
        raise ValueError("GEMV requires batch=1")
    if k != kb:
        raise ValueError("incompatible matrix dimensions")
    estimate_gemv_schedule(n, k, lanes)
    for col in range(0, n, lanes):
        for inner in range(k):
            yield a[0][inner], [b[inner][col+lane] if col+lane < n else 0 for lane in range(lanes)]
