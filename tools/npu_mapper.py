"""Map sequential floating-point linear layers to INT8 GEMM or GEMV streams."""
import argparse
import json
import math
from pathlib import Path

from npu_model import matmul_int8, linear_int8
from npu_stream import estimate_tile_memory, estimate_gemm_schedule, operand_vectors
from npu_gemv import estimate_gemv_memory, estimate_gemv_schedule, operand_beats

SOURCE_FORMAT = "xilinx-gpu.linear.v1"
MAPPED_FORMAT = "xilinx-gpu.mapped-linear.v1"


def _number(value, label):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _scale(value, label):
    value = _number(value, label)
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _keys(obj, required, optional, label):
    if not isinstance(obj, dict) or set(obj)-required-optional or required-set(obj):
        raise ValueError(f"{label}: required fields {sorted(required)}, optional {sorted(optional)}")


def _matrix(values, label):
    if (not isinstance(values, list) or not 1 <= len(values) <= 65535
            or not isinstance(values[0], list) or not 1 <= len(values[0]) <= 65535):
        raise ValueError(f"{label} must be a nonempty matrix with 16-bit dimensions")
    width = len(values[0])
    if any(not isinstance(row, list) or len(row) != width for row in values):
        raise ValueError(f"{label} must be rectangular")
    return [[_number(value, label) for value in row] for row in values]


def _round_away(value):
    # Adding 0.5 first can round nextafter(0.5, 0) up to 1.0 in binary64.
    magnitude = abs(value)
    integer = math.floor(magnitude)
    return (-1 if value < 0 else 1)*(integer+int(magnitude-integer >= 0.5))


def quantize_values(values, scale):
    """Zero-point-zero INT8, nearest with ties away; report actual clipping."""
    scale = _scale(scale, "scale")
    quantized, clipped = [], 0
    for value in values:
        quotient = _number(value, "value")/scale
        if not math.isfinite(quotient):
            quantized.append(-128 if value < 0 else 127)
            clipped += 1
            continue
        rounded = _round_away(quotient)
        clipped += not -128 <= rounded <= 127
        quantized.append(max(-128, min(127, rounded)))
    return quantized, clipped


def requantization_params(ratio):
    """Highest precision representable multiplier/2**shift, shift 0..31."""
    ratio = _scale(ratio, "requantization ratio")
    for shift in range(31, -1, -1):
        scaled = ratio*(1 << shift)
        if math.isfinite(scaled):
            multiplier = _round_away(scaled)
            if 1 <= multiplier <= (1 << 31)-1:
                return multiplier, shift
    raise ValueError("requantization ratio cannot be represented by positive INT31 and shift 0..31")


def compile_model(model):
    """Compile JSON weights[K][N]; explicit activation scales, optional channel scales."""
    _keys(model, {"format", "input_scale", "layers"}, set(), "model")
    if model["format"] != SOURCE_FORMAT or not isinstance(model["layers"], list) or not model["layers"]:
        raise ValueError(f"expected {SOURCE_FORMAT} and a nonempty layers list")
    input_scale = _scale(model["input_scale"], "input_scale")
    compiled = {"format": MAPPED_FORMAT, "input_scale": input_scale, "layers": []}
    names, previous_width = set(), None
    for source in model["layers"]:
        _keys(source, {"name", "weights", "bias", "output_scale"},
              {"relu", "weight_scales"}, "layer")
        name = source["name"]
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("layer names must be nonempty and unique")
        names.add(name)
        weights = _matrix(source["weights"], f"{name}.weights")
        k, n = len(weights), len(weights[0])
        if previous_width is not None and k != previous_width:
            raise ValueError(f"{name}: previous output width differs from weights[K][N]")
        bias = source["bias"]
        if not isinstance(bias, list) or len(bias) != n:
            raise ValueError(f"{name}: one bias is required per output channel")
        bias = [_number(value, f"{name}.bias") for value in bias]
        output_scale = _scale(source["output_scale"], f"{name}.output_scale")
        relu = source.get("relu", False)
        if type(relu) is not bool:
            raise ValueError(f"{name}.relu must be boolean")
        scales = source.get("weight_scales")
        if scales is None:
            maxima = [max(abs(row[c]) for row in weights) for c in range(n)]
            scales = [maximum/127 if maximum else 1.0 for maximum in maxima]
        if not isinstance(scales, list) or len(scales) != n:
            raise ValueError(f"{name}: one weight scale is required per output channel")
        scales = [_scale(value, f"{name}.weight_scales") for value in scales]
        quantized_columns, clipped = [], 0
        qb, multipliers, shifts, approximations = [], [], [], []
        for c, weight_scale in enumerate(scales):
            column, count = quantize_values([row[c] for row in weights], weight_scale)
            quantized_columns.append(column);clipped += count
            accumulator_scale = _scale(input_scale*weight_scale, f"{name}.accumulator_scale")
            integer_bias = _round_away(_number(bias[c]/accumulator_scale, f"{name}.bias/scale"))
            if not -(1 << 31) <= integer_bias < (1 << 31):
                raise ValueError(f"{name}: quantized bias exceeds INT32")
            ratio = _scale(accumulator_scale/output_scale, f"{name}.requantization ratio")
            multiplier, shift = requantization_params(ratio)
            represented = multiplier/(1 << shift)
            qb.append(integer_bias);multipliers.append(multiplier);shifts.append(shift)
            approximations.append({"target": ratio, "represented": represented,
                                   "relative_error": abs(represented-ratio)/ratio})
        compiled["layers"].append({"name": name, "input_features": k, "output_features": n,
            "input_scale": input_scale, "output_scale": output_scale, "weight_scales": scales,
            "weights": [list(row) for row in zip(*quantized_columns)], "bias": qb,
            "multipliers": multipliers, "shifts": shifts, "relu": relu,
            "weights_clipped": clipped, "requantization": approximations})
        input_scale, previous_width = output_scale, n
    return compiled


def _errors(reference, values):
    pairs = [(a, b) for r, s in zip(reference, values) for a, b in zip(r, s)]
    differences = [_number(b-a, "reference error") for a, b in pairs]
    error_norm = math.hypot(*differences)
    reference_norm = math.hypot(*(a for a, _ in pairs))
    return {"mae": math.fsum(abs(x)/len(pairs) for x in differences),
            "rmse": error_norm/math.sqrt(len(pairs)), "max_abs": max(map(abs, differences)),
            "relative_l2": error_norm/reference_norm if reference_norm else (0.0 if not error_norm else None)}


def _saturation_count(raw, layer):
    count = 0
    for row in raw:
        for c, value in enumerate(row):
            product = (value+layer["bias"][c])*layer["multipliers"][c]
            shift = layer["shifts"][c]
            magnitude = (abs(product)+(1 << (shift-1))) >> shift if shift else abs(product)
            scaled = -magnitude if product < 0 else magnitude
            if layer["relu"]:
                scaled = max(0, scaled)
            count += not -128 <= scaled <= 127
    return count


def map_model(model, samples, lanes=4, k_tile=None, backend="gemm"):
    if backend not in ("gemm", "gemv"):
        raise ValueError("backend must be gemm or gemv")
    if backend == "gemv" and k_tile is not None:
        raise ValueError("k_tile is not supported by GEMV")
    if backend == "gemm" and k_tile is None:
        k_tile = 16
    compiled = compile_model(model)
    samples = _matrix(samples, "inputs")
    if backend == "gemv" and len(samples) != 1:
        raise ValueError("GEMV requires batch=1")
    if len(samples[0]) != compiled["layers"][0]["input_features"]:
        raise ValueError("input width differs from first layer weights[K][N]")
    tile_memory = (estimate_gemv_memory(lanes) if backend == "gemv"
                   else estimate_tile_memory(lanes, k_tile))
    activations, input_clipped = [], 0
    for row in samples:
        quantized, clipped = quantize_values(row, compiled["input_scale"])
        activations.append(quantized);input_clipped += clipped
    trace, reports, floating = [], [], samples
    for layer, source in zip(compiled["layers"], model["layers"]):
        m, k, n = len(samples), layer["input_features"], layer["output_features"]
        raw = matmul_int8(activations, layer["weights"])
        output = linear_int8(raw, layer["bias"], layer["multipliers"], layer["shifts"], layer["relu"])
        floating = [[_number(sum(row[t]*source["weights"][t][c] for t in range(k))+source["bias"][c],
                             "float reference") for c in range(n)] for row in floating]
        if layer["relu"]:
            floating = [[max(0.0, value) for value in row] for row in floating]
        dequantized = [[value*layer["output_scale"] for value in row] for row in output]
        schedule = (estimate_gemv_schedule(n, k, lanes, output_bytes=1) if backend == "gemv"
                    else estimate_gemm_schedule(m, n, k, lanes, k_tile, output_bytes=1))
        trace.append({"inputs": activations, "raw": raw, "outputs": output})
        reports.append({"name": layer["name"], "shape": {"m": m, "n": n, "k": k},
            "float_reference": floating, "int8_output": output, "dequantized_output": dequantized,
            "error": _errors(floating, dequantized), "saturated_elements": _saturation_count(raw, layer),
            "at_int8_bounds": sum(value in (-128, 127) for row in output for value in row),
            "weight_clipped_elements": layer["weights_clipped"],
            "unique_input_bytes": m*k, "unique_weight_bytes": k*n,
            "unique_bias_scale_packed_bytes": (70*n+7)//8,
            "host_activation_bytes": m*(k+n),
            "A_stream_bytes": schedule["operand_vector_count"]*(1 if backend == "gemv" else lanes),
            "B_stream_bytes": schedule["operand_vector_count"]*lanes,
            "parameter_byte_aligned_bytes": m*n*9, "schedule": schedule})
        activations = output
    report = {"format": "xilinx-gpu.mapper-report.v1", "backend": backend,
        "physical_mac_count": lanes if backend == "gemv" else lanes**2,
        "lanes": lanes, "k_tile": k_tile,
        "batch": len(samples), "layers": reports, "tile_memory": tile_memory,
        "input_clipped_elements": input_clipped,
        "host_activation_peak_bytes": max(x["host_activation_bytes"] for x in reports),
        "MAC_count": sum(x["schedule"]["MAC_count"] for x in reports),
        "operand_stream_bytes": sum(x["schedule"]["operand_load_bytes"] for x in reports),
        "parameter_packed_bytes": sum(x["schedule"]["parameter_stream_bytes"] for x in reports),
        "parameter_byte_aligned_bytes": sum(x["parameter_byte_aligned_bytes"] for x in reports),
        "output_stream_bytes": sum(x["schedule"]["output_store_bytes"] for x in reports),
        "rtl_validation": {"status": "not_run"}, "hardware_validation": "not_performed",
        "float_reference_arithmetic": "Python binary64, not framework FP32",
        "memory_note": "Logical activation double-buffer and parameter payload only; excludes retained reference traces, float source, Python objects and physical mapping"}
    return {"compiled": compiled, "report": report, "layers": trace}


def output_coordinates(m, n, lanes):
    for tr in range(0, m, lanes):
        for tc in range(0, n, lanes):
            for r in range(tr, min(tr+lanes, m)):
                for c in range(tc, min(tc+lanes, n)):
                    yield r, c


def _packed(vector):
    return sum((value & 255) << (8*lane) for lane, value in enumerate(vector))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def export_bundle(model, samples, output_dir, lanes=4, k_tile=None, backend="gemm"):
    """Write actual packed vector/parameter streams and independent expected outputs."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root/"report.json").unlink(missing_ok=True)
    plan = map_model(model, samples, lanes, k_tile, backend)
    compiled, traces, report = plan["compiled"], plan["layers"], plan["report"]
    with (root/"jobs.txt").open("w", encoding="ascii", newline="\n") as jobs, \
            (root/"operands.txt").open("w", encoding="ascii", newline="\n") as operands, \
            (root/"parameters.txt").open("w", encoding="ascii", newline="\n") as parameters, \
            (root/"expected.txt").open("w", encoding="ascii", newline="\n") as expected:
        jobs.write(f"gemv-v1 {len(traces)} {lanes}\n" if backend == "gemv"
                   else f"{len(traces)} {lanes} {report['k_tile']}\n")
        for layer, trace in zip(compiled["layers"], traces):
            a, b, output = trace["inputs"], layer["weights"], trace["outputs"]
            m, k, n = len(a), len(b), len(b[0])
            jobs.write(f"{m} {n} {k} {int(layer['relu'])}\n")
            if backend == "gemv":
                for scalar, bv in operand_beats(a, b, lanes):
                    operands.write(f"{scalar & 255:02x} {_packed(bv):0{2*lanes}x}\n")
            else:
                for av, bv in operand_vectors(a, b, lanes):
                    operands.write(f"{_packed(av):0{2*lanes}x} {_packed(bv):0{2*lanes}x}\n")
            for r, c in output_coordinates(m, n, lanes):
                parameters.write(f"{r} {c} {layer['bias'][c]} {layer['multipliers'][c]} {layer['shifts'][c]}\n")
            for row in output:
                expected.write(" ".join(map(str, row))+"\n")
    (root/"inputs.txt").write_text("\n".join(" ".join(map(str, row)) for row in traces[0]["inputs"])+"\n", encoding="ascii")
    write_json(root/"mapped-model.json", compiled)
    write_json(root/"report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="linear.v1 JSON, weights[K][N]")
    parser.add_argument("--inputs", type=Path, required=True, help="JSON matrix [batch][input_features]")
    parser.add_argument("--output-dir", type=Path, default=Path("build/npu-mapped"))
    parser.add_argument("--lanes", type=int, default=4)
    parser.add_argument("--backend", choices=("gemm", "gemv"), default="gemm")
    parser.add_argument("--k-tile", type=int, default=None)
    parser.add_argument("--simulate", action="store_true", help="also run Icarus on exported streams")
    args = parser.parse_args()
    try:
        # Clear the success artifact even if loading/parsing the new source fails.
        (args.output_dir/"report.json").unlink(missing_ok=True)
        source = json.loads(args.model.read_text(encoding="utf-8"))
        inputs = json.loads(args.inputs.read_text(encoding="utf-8"))
        report = export_bundle(source, inputs, args.output_dir, args.lanes, args.k_tile, args.backend)
        if args.simulate:
            from npu_mapper_sim import simulate_bundle
            report = simulate_bundle(args.output_dir)
        print(json.dumps({"report": str(args.output_dir/"report.json"),
              "layers": len(report["layers"]), "MAC_count": report["MAC_count"],
              "host_activation_peak_bytes": report["host_activation_peak_bytes"],
              "rtl_validation": report["rtl_validation"]}, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, OverflowError) as error:
        parser.exit(1, f"MAPPING FAILED: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
