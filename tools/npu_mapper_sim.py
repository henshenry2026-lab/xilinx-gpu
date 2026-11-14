"""Execute exported GEMM or GEMV streams on RTL and verify their results."""
import json
from pathlib import Path
import subprocess

from hdl_tools import find_tool
from npu_mapper import output_coordinates, write_json
from npu_gemv import estimate_gemv_schedule
from npu_stream import estimate_gemm_schedule

ROOT = Path(__file__).resolve().parents[1]


def _run(command, root, log):
    try:
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", timeout=120)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"RTL execution failed: {error}") from error
    output = process.stdout+process.stderr
    (root/log).write_text(output, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"RTL failed (exit {process.returncode}); {root/log}:\n{output[-2500:]}")


def verify_results(report, root):
    layers = report["layers"]
    seen = set()
    order = iter((job, r, c) for job, layer in enumerate(layers)
                 for r, c in output_coordinates(layer["shape"]["m"], layer["shape"]["n"], report["lanes"]))
    for line in (root/"rtl-results.txt").read_text(encoding="ascii").splitlines():
        try:
            fields = list(map(int, line.split()))
        except ValueError as error:
            raise RuntimeError("malformed RTL result") from error
        if len(fields) != 4:
            raise RuntimeError("malformed RTL result")
        job, r, c, value = fields
        if (not 0 <= job < len(layers) or not 0 <= r < layers[job]["shape"]["m"]
                or not 0 <= c < layers[job]["shape"]["n"] or (job, r, c) in seen):
            raise RuntimeError("invalid/duplicate RTL result coordinate")
        if (job, r, c) != next(order, None):
            raise RuntimeError("RTL result order differs from stream order")
        if value != layers[job]["int8_output"][r][c]:
            raise RuntimeError(f"RTL integer mismatch at layer {job} [{r},{c}]")
        seen.add((job, r, c))
    if len(seen) != sum(x["shape"]["m"]*x["shape"]["n"] for x in layers):
        raise RuntimeError("RTL results incomplete")
    rows = (root/"rtl-cycles.txt").read_text(encoding="ascii").splitlines()
    if len(rows) != len(layers):
        raise RuntimeError("RTL cycle report incomplete")
    counter_names = ["cycles", "load_cycles", "compute_cycles", "input_wait_cycles",
                     "buffer_stall_cycles", "output_stall_cycles", "overlap_cycles", "MAC_count"]
    for job, (line, layer) in enumerate(zip(rows, layers)):
        try:
            fields = list(map(int, line.split()))
        except ValueError as error:
            raise RuntimeError("malformed RTL cycle report") from error
        shape, schedule = layer["shape"], layer["schedule"]
        if len(fields) != 12 or fields[:4] != [job, shape["m"], shape["n"], shape["k"]]:
            raise RuntimeError("RTL cycle descriptor differs from model")
        counters = dict(zip(counter_names, fields[4:]))
        if (counters["load_cycles"] != schedule["operand_vector_count"]
                or counters["compute_cycles"] != schedule["phase_compute_cycles"]
                or counters["MAC_count"] != schedule["MAC_count"]
                or any(counters[name] < 0 or counters[name] > counters["cycles"] for name in counter_names[:-1])
                or counters["cycles"] <= 0
                or counters["overlap_cycles"] > min(counters["load_cycles"], counters["compute_cycles"])):
            raise RuntimeError("RTL counters differ from schedule")
        if report.get("backend", "gemm") == "gemv" and (
                counters["overlap_cycles"] != schedule["accepted_beats"]
                or counters["buffer_stall_cycles"] != 0):
            raise RuntimeError("GEMV direct-MAC counters differ from schedule")
        capacity = report["lanes"] if report.get("backend", "gemm") == "gemv" else report["lanes"]**2
        layer["rtl"] = dict(counters, integer_mismatches=0,
                            useful_array_utilization=counters["MAC_count"]/(counters["cycles"]*
                                report.get("physical_mac_count", capacity)))
    return report


def _clear_layer_rtl(report):
    layers = report.get("layers")
    if isinstance(layers, list):
        for layer in layers:
            if isinstance(layer, dict):
                layer.pop("rtl", None)


def _report_integer(value, minimum, maximum, label):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"report {label} must be an integer in {minimum}..{maximum}")
    return value


def _validate_report(report):
    """Check only bounded metadata consumed by compilation and verification."""
    backend = report.get("backend", "gemm")
    if backend not in ("gemm", "gemv"):
        raise ValueError("report backend must be gemm or gemv")
    batch = _report_integer(report.get("batch"), 1, 1 if backend == "gemv" else 8, "batch")
    lanes = _report_integer(report.get("lanes"), 1, 16, "lanes")
    capacity = lanes if backend == "gemv" else lanes**2
    if _report_integer(report.get("physical_mac_count", capacity), 1, 256, "physical_mac_count") != capacity:
        raise ValueError("report physical_mac_count differs from backend capacity")
    k_tile = report.get("k_tile")
    if backend == "gemm":
        _report_integer(k_tile, 1, 65535, "k_tile")
    elif k_tile is not None:
        raise ValueError("report GEMV k_tile must be null")
    layers = report.get("layers")
    if not isinstance(layers, list) or not 1 <= len(layers) <= 64:
        raise ValueError("report layers must be a nonempty list of at most 64 layers")
    previous_width = None
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict) or not isinstance(layer.get("shape"), dict):
            raise ValueError(f"report layer {index} requires a shape object")
        shape = layer["shape"]
        m = _report_integer(shape.get("m"), batch, batch, f"layer {index} shape.m")
        n = _report_integer(shape.get("n"), 1, 256, f"layer {index} shape.n")
        k = _report_integer(shape.get("k"), 1, 256, f"layer {index} shape.k")
        if previous_width is not None and previous_width != k:
            raise ValueError(f"report layer {index} input width differs from previous output")
        previous_width = n
        output = layer.get("int8_output")
        if (not isinstance(output, list) or len(output) != m
                or any(not isinstance(row, list) or len(row) != n for row in output)):
            raise ValueError(f"report layer {index} int8_output shape differs from descriptor")
        for row in output:
            for value in row:
                _report_integer(value, -128, 127, f"layer {index} int8_output value")
        schedule = layer.get("schedule")
        if not isinstance(schedule, dict):
            raise ValueError(f"report layer {index} requires a schedule object")
        expected = (estimate_gemv_schedule(n, k, lanes, output_bytes=1) if backend == "gemv"
                    else estimate_gemm_schedule(m, n, k, lanes, k_tile, output_bytes=1))
        fields = ["operand_vector_count", "phase_compute_cycles", "MAC_count"]
        if backend == "gemv":
            fields.append("accepted_beats")
        if any(type(schedule.get(field)) is not int or schedule[field] != expected[field] for field in fields):
            raise ValueError(f"report layer {index} schedule differs from backend descriptor")


def simulate_bundle(output_dir, stalls=True):
    root = Path(output_dir).resolve()
    report_path = root/"report.json"
    report = None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError("report root must be an object")
        # Invalidate success before inspecting any potentially malformed layers.
        report["rtl_validation"] = {"status": "running", "stalls": bool(stalls)}
        _clear_layer_rtl(report)
        write_json(report_path, report)
        _validate_report(report)
        iverilog, vvp = find_tool("iverilog"), find_tool("vvp")
        lanes = report["lanes"]
        k_tile = report["k_tile"] if report.get("backend", "gemm") == "gemm" else 16
        backend = int(report.get("backend", "gemm") == "gemv")
        sources = sorted(str(path) for path in (ROOT/"rtl/npu").glob("*") if path.suffix in (".v", ".sv"))
        _run([iverilog, "-g2012", "-Wall", "-s", "tb_npu_mapped", f"-Ptb_npu_mapped.LANES={lanes}",
              f"-Ptb_npu_mapped.K_TILE={k_tile}", f"-Ptb_npu_mapped.BACKEND={backend}",
              f"-Ptb_npu_mapped.STALLS={int(bool(stalls))}", "-o", str(root/"mapped.vvp"), *sources,
              str(ROOT/"tests/npu/tb_npu_mapped.sv")], root, "compile.log")
        _run([vvp, str(root/"mapped.vvp"), f"+BUNDLE={root.as_posix()}"], root, "simulation.log")
        verify_results(report, root)
        report["rtl_validation"] = {"status": "passed", "layers": len(report["layers"]),
                                    "stalls": bool(stalls),
                                    "live_intermediate_activations": len(report["layers"]) > 1}
        write_json(report_path, report)
    except (OSError, ValueError, RuntimeError, TypeError, KeyError, IndexError, ArithmeticError) as error:
        message = str(error) if isinstance(error, RuntimeError) else f"invalid report: {error}"
        if isinstance(report, dict):
            report["rtl_validation"] = {"status": "failed", "stalls": bool(stalls), "error": message}
            _clear_layer_rtl(report)
            try:
                write_json(report_path, report)
            except (OSError, ValueError, TypeError):
                report_path.unlink(missing_ok=True)
        else:
            report_path.unlink(missing_ok=True)
        raise RuntimeError(message) from error
    return report
