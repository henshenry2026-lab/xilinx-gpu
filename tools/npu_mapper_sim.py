"""Execute exported streams on the unchanged linear RTL and verify its results."""
import json
from pathlib import Path
import subprocess

from hdl_tools import find_tool
from npu_mapper import write_json

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
    for line in (root/"rtl-results.txt").read_text(encoding="ascii").splitlines():
        fields = list(map(int, line.split()))
        if len(fields) != 4:
            raise RuntimeError("malformed RTL result")
        job, r, c, value = fields
        if (not 0 <= job < len(layers) or not 0 <= r < layers[job]["shape"]["m"]
                or not 0 <= c < layers[job]["shape"]["n"] or (job, r, c) in seen):
            raise RuntimeError("invalid/duplicate RTL result coordinate")
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
        fields = list(map(int, line.split()))
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
        layer["rtl"] = dict(counters, integer_mismatches=0,
                            useful_array_utilization=counters["MAC_count"]/(counters["cycles"]*report["lanes"]**2))
    return report


def simulate_bundle(output_dir):
    root = Path(output_dir).resolve()
    report = json.loads((root/"report.json").read_text(encoding="utf-8"))
    report["rtl_validation"] = {"status": "running"}
    for layer in report["layers"]:
        layer.pop("rtl", None)
    write_json(root/"report.json", report)
    try:
        if (report["batch"] > 8 or len(report["layers"]) > 64
                or any(max(x["shape"]["n"], x["shape"]["k"]) > 256 for x in report["layers"])):
            raise RuntimeError("RTL bundle simulator supports batch<=8, features<=256, layers<=64")
        iverilog, vvp = find_tool("iverilog"), find_tool("vvp")
        lanes, k_tile = report["lanes"], report["k_tile"]
        sources = sorted(str(path) for path in (ROOT/"rtl/npu").glob("*") if path.suffix in (".v", ".sv"))
        _run([iverilog, "-g2012", "-Wall", "-s", "tb_npu_mapped", f"-Ptb_npu_mapped.LANES={lanes}",
              f"-Ptb_npu_mapped.K_TILE={k_tile}", "-o", str(root/"mapped.vvp"), *sources,
              str(ROOT/"tests/npu/tb_npu_mapped.sv")], root, "compile.log")
        _run([vvp, str(root/"mapped.vvp"), f"+BUNDLE={root.as_posix()}"], root, "simulation.log")
        verify_results(report, root)
        report["rtl_validation"] = {"status": "passed", "layers": len(report["layers"]),
                                    "live_intermediate_activations": len(report["layers"]) > 1}
    except (OSError, ValueError, RuntimeError) as error:
        report["rtl_validation"] = {"status": "failed", "error": str(error)}
        write_json(root/"report.json", report)
        raise RuntimeError(str(error)) from error
    write_json(root/"report.json", report)
    return report
