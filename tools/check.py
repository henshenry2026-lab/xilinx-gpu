"""Run the same host, RTL and Xilinx 7-series synthesis checks locally and in CI."""

import argparse
from collections import Counter
import json
from pathlib import Path
import struct
import subprocess
import sys
import zlib

from uart_rect import encode_rect
from npu_model import write_vectors, write_linear_vectors
from npu_stream import (write_cases as write_stream_cases, estimate_tile_memory,
                        estimate_gemm_schedule, estimate_mlp_schedule, parameter_sweep)
from hdl_tools import find_tool
from npu_mapper import export_bundle
from npu_mapper_sim import simulate_bundle

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"


def run(label, command, log_name, timeout=120):
    print(f"\n[{label}]", flush=True)
    result = subprocess.run(command, cwd=ROOT, text=True, encoding="utf-8", errors="replace",
                            capture_output=True, timeout=timeout)
    output = result.stdout + result.stderr
    (BUILD / log_name).write_text(output, encoding="utf-8")
    if result.returncode:
        print(output[-12000:], file=sys.stderr)
        raise RuntimeError(f"{label} failed (exit {result.returncode}); full log: build/{log_name}")
    if not label.endswith("synthesis"):
        print(output.strip(), flush=True)
    else:
        print(f"PASS synth_xilinx + check -assert; full report: build/{log_name}", flush=True)
    return output


def count_cells(module_name, module_set):
    result = Counter()
    for cell in module_set[module_name].get("cells", {}).values():
        kind = cell["type"]
        child = module_set.get(kind, {})
        attributes = child.get("attributes", {})
        # Count Xilinx primitives, not the timing/specify cells in their models.
        primitive = any(int(str(attributes.get(flag, "0")), 2)
                        for flag in ("blackbox", "whitebox"))
        if not primitive and child.get("cells"):
            result.update(count_cells(kind, module_set))
        else:
            result[kind] += 1
    return result


def resource_summary(top, netlist):
    resources = count_cells(top, json.loads(netlist.read_text(encoding="utf-8"))["modules"])
    return {"dsp48e1": resources["DSP48E1"], "ramb18e1": resources["RAMB18E1"],
            "ramb36e1": resources["RAMB36E1"],
            "logic_lut_primitives": sum(v for k, v in resources.items() if k.startswith("LUT")),
            "flip_flops": sum(v for k, v in resources.items() if k.startswith("FD")),
            "distributed_ram_primitives": {k: v for k, v in sorted(resources.items())
                                           if k.startswith("RAM") and not k.startswith("RAMB")}}


def stream_measurements(path, lanes, k_tile, linear):
    fields = ["m", "n", "k", "cycles", "load_cycles", "compute_cycles", "input_wait_cycles",
              "buffer_stall_cycles", "output_stall_cycles", "overlap_cycles", "MAC_count", "mlp_layer"]
    records = []
    for line in path.read_text(encoding="ascii").splitlines():
        values = list(map(int, line.split()))
        if len(values) != len(fields):
            raise RuntimeError(f"Malformed measurement: {path.name}")
        row = dict(zip(fields, values))
        model = estimate_gemm_schedule(row["m"], row["n"], row["k"], lanes, k_tile,
                                       output_bytes=1 if linear else 4)
        if (row["load_cycles"] != model["operand_vector_count"]
                or row["compute_cycles"] != model["phase_compute_cycles"]
                or row["MAC_count"] != model["MAC_count"]
                or row["cycles"] < max(row["compute_cycles"], row["load_cycles"])):
            raise RuntimeError(f"RTL counters differ from model: {path.name}: {row}")
        if any(row[key] < 0 or row[key] > row["cycles"] for key in fields[4:10]):
            raise RuntimeError(f"Invalid cycle counter: {path.name}: {row}")
        if row["overlap_cycles"] > min(row["load_cycles"], row["compute_cycles"]):
            raise RuntimeError(f"Invalid overlap counter: {path.name}: {row}")
        row.update({"predicted": model,
                    "useful_MACs_per_cycle": row["MAC_count"] / row["cycles"],
                    "useful_array_utilization": row["MAC_count"] / (row["cycles"]*lanes*lanes)})
        records.append(row)
    if not records:
        raise RuntimeError(f"Empty measurement: {path.name}")
    return records


def preview(update):
    """Convert the actual simulated PPM into RGB PNG, using only the standard library."""
    magic, dimensions, maximum, pixels = (BUILD / "demo.ppm").read_bytes().split(b"\n", 3)
    width, height = map(int, dimensions.split())
    if magic != b"P6" or maximum != b"255" or (width, height) != (640, 480):
        raise RuntimeError("Unexpected simulation image header")
    if len(pixels) != width * height * 3:
        raise RuntimeError("Simulation image is incomplete")
    raw = b"".join(b"\x00" + pixels[y * width * 3:(y + 1) * width * 3] for y in range(height))

    def chunk(kind, content):
        return (struct.pack(">I", len(content)) + kind + content
                + struct.pack(">I", zlib.crc32(kind + content) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    (BUILD / "demo.png").write_bytes(png)
    target = ROOT / "docs" / "demo.png"
    if update:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(png)
    if not target.is_file():
        raise RuntimeError("README preview missing; run python tools/check.py --update-preview")
    # Compare decoded pixels, so differing platform zlib versions are harmless.
    existing = target.read_bytes()
    if existing[:8] != png[:8]:
        raise RuntimeError("Invalid README PNG signature")
    offset, compressed, existing_header = 8, bytearray(), None
    while offset < len(existing):
        length = struct.unpack_from(">I", existing, offset)[0]
        kind = existing[offset + 4:offset + 8]
        content = existing[offset + 8:offset + 8 + length]
        crc = struct.unpack_from(">I", existing, offset + 8 + length)[0]
        if zlib.crc32(kind + content) & 0xFFFFFFFF != crc:
            raise RuntimeError("README PNG checksum mismatch")
        if kind == b"IHDR":
            existing_header = content
        if kind == b"IDAT":
            compressed.extend(content)
        offset += length + 12
    if existing_header != header or zlib.decompress(compressed) != raw:
        raise RuntimeError("README preview differs from RTL; regenerate with --update-preview")
    print("PASS preview: docs/demo.png matches simulated pixels", flush=True)


def gemv_acceptance(resources, measurements):
    """Require complete, numerically equal evidence and the promised resource bounds."""
    for lanes in (1, 2, 4):
        for kind in ("raw", "linear"):
            name = f"gemv_{kind}_{lanes}"
            mapped = resources.get(name, {})
            if (mapped.get("dsp48e1") != lanes + (4 if kind == "linear" else 0)
                    or mapped.get("ramb18e1", 1) or mapped.get("ramb36e1", 1)
                    or mapped.get("distributed_ram_primitives", {"missing": 1})):
                raise RuntimeError(f"GEMV resource/memory regression: {name}: {mapped}")
    if resources["gemv_linear_4"]["dsp48e1"] > resources["stream_linear_2"]["dsp48e1"]:
        raise RuntimeError("Default GEMV exceeds the 2x2 linear baseline DSP count")
    configurations, comparisons = {}, []
    for case in ("mlp_batch1", "tail_batch1"):
        for mode in ("continuous", "stalled"):
            group = {}
            for backend, lanes in (("gemm", 2), ("gemm", 4), ("gemv", 1), ("gemv", 2), ("gemv", 4)):
                name = f"{case}_{mode}_{backend}_l{lanes}"
                report = measurements.get(name)
                if not report or report.get("rtl_validation", {}).get("status") != "passed":
                    raise RuntimeError(f"Missing/passing RTL evidence required: {name}")
                cycles = [layer["rtl"]["cycles"] for layer in report["layers"]]
                if not cycles or any(type(x) is not int or x <= 0 for x in cycles):
                    raise RuntimeError(f"Invalid GEMV comparison cycles: {name}")
                group[backend, lanes] = report
                configurations[name] = {
                    "backend": backend, "lanes": lanes, "stimulus": mode,
                    "report": f"gemv/{name}/report.json",
                    "MAC_count": report["MAC_count"],
                    "operand_stream_bytes": report["operand_stream_bytes"],
                    "sum_layer_busy_cycles": sum(cycles),
                    "layers": [{key: layer[key] for key in ("name", "shape", "rtl")} for layer in report["layers"]]}
            reference = [x["int8_output"] for x in group["gemm", 2]["layers"]]
            for report in group.values():
                if [x["int8_output"] for x in report["layers"]] != reference:
                    raise RuntimeError(f"Backend integer output mismatch: {case}/{mode}")
            baseline = sum(x["rtl"]["cycles"] for x in group["gemm", 2]["layers"])
            vector = sum(x["rtl"]["cycles"] for x in group["gemv", 4]["layers"])
            if vector > baseline:
                raise RuntimeError(f"Default GEMV cycle regression: {case}/{mode}: {vector} > {baseline}")
            comparisons.append({"case": case, "stimulus": mode,
                                "gemv4_cycles": vector, "gemm2_cycles": baseline,
                                "gemm2_over_gemv4_cycle_ratio": baseline/vector})
    return {"configurations": configurations, "resources": resources,
            "comparisons": comparisons, "integer_mismatches": 0,
            "cycles_note": "Sum of layer busy cycles, excluding host work and physical transport; no measured frequency",
            "stimulus_note": "Continuous ready/valid and fixed-seed gaps/backpressure; logical beat streams have different widths",
            "memory_note": "GEMV accumulator payload only; control, counters, output slot and physical resources reported separately",
            "hardware_validation": "not performed"}


def check_gemv(yosys, sources, stream_resources, models):
    """Synthesize the vector tops and compare both dataflows on identical models."""
    resources = {"stream_linear_2": stream_resources["stream_linear_2"]["mapped"]}
    for lanes in (1, 2, 4):
        for kind in ("raw", "linear"):
            top = "npu_gemv_linear_top" if kind == "linear" else "npu_gemv_top"
            name = f"gemv_{kind}_{lanes}"
            script = ("read_verilog -sv " + " ".join(sources)
                      + f"; chparam -set LANES {lanes} {top};"
                      + f" hierarchy -check -top {top}; synth_xilinx -family xc7 -top {top};"
                      + f" check -assert; stat; write_json build/{name}.json")
            run(f"NPU {name} Xilinx synthesis", [yosys, "-Q", "-T", "-p", script],
                f"npu-{name}-synthesis.log", timeout=300)
            resources[name] = resource_summary(top, BUILD/f"{name}.json")
            resources[name]["accumulator_payload_bytes"] = 4*lanes
            print(f"PASS {name}: {resources[name]}", flush=True)
    measurements = {}
    for case, source, inputs in models:
        for stalls in (False, True):
            mode = "stalled" if stalls else "continuous"
            for backend, lanes in (("gemm", 2), ("gemm", 4), ("gemv", 1), ("gemv", 2), ("gemv", 4)):
                name = f"{case}_{mode}_{backend}_l{lanes}"
                folder = BUILD/"gemv"/name
                export_bundle(source, inputs, folder, lanes=lanes, backend=backend)
                measured = simulate_bundle(folder, stalls=stalls)
                measurements[name] = measured
                cycles = sum(layer["rtl"]["cycles"] for layer in measured["layers"])
                print(f"PASS {name}: {cycles} busy cycles, {measured['operand_stream_bytes']} operand bytes", flush=True)
    summary = gemv_acceptance(resources, measurements)
    (BUILD/"npu-gemv-summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-preview", action="store_true", help="refresh the tracked README image")
    args = parser.parse_args()
    BUILD.mkdir(exist_ok=True)
    # Never leave a previous successful summary behind after a failed rerun.
    for name in ["check-summary.json", "npu-memory-summary.json", "npu-cycle-summary.json",
                 "npu-mapper-summary.json", "npu-gemv-summary.json"]:
        (BUILD / name).unlink(missing_ok=True)
    try:
        iverilog, vvp, yosys = (find_tool(name) for name in ["iverilog", "vvp", "yosys"])
        run("Python tests", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            "python-tests.log")
        packets = encode_rect(240, 140, 160, 80, 0x1C) + encode_rect(160, 120, 320, 240, 0x1F)
        (BUILD / "uart_packets.mem").write_text("\n".join(f"{byte:02x}" for byte in packets) + "\n",
                                                encoding="ascii")
        sources = [path.relative_to(ROOT).as_posix() for path in sorted((ROOT / "rtl").glob("*.v"))]
        benches = ["tb_uart_rx", "tb_command_parser", "tb_scene", "tb_gpu_top"]
        for bench in benches:
            binary = f"build/{bench}.vvp"
            run(f"Compile {bench}", [iverilog, "-g2012", "-Wall", "-s", bench, "-o", binary,
                                     *sources, f"tests/rtl/{bench}.sv"], f"{bench}-compile.log")
            run(bench, [vvp, binary], f"{bench}.log")
        synthesis = ("read_verilog " + " ".join(sources)
                     + "; hierarchy -check -top gpu_top; synth_xilinx -family xc7 -top gpu_top;"
                     + " check -assert; stat; write_json build/gpu_top.json")
        run("Xilinx 7-series synthesis", [yosys, "-Q", "-T", "-p", synthesis], "synthesis.log", timeout=300)
        preview(args.update_preview)

        matrix_cases = write_vectors(BUILD / "npu_cases.txt")
        linear_cases = write_linear_vectors(BUILD / "npu_linear_cases.txt")
        stream_cases = write_stream_cases(BUILD / "npu_stream_cases.txt")
        npu_sources = [path.relative_to(ROOT).as_posix() for path in sorted((ROOT / "rtl/npu").glob("*"))
                       if path.suffix in {".v", ".sv"}]
        npu_benches = [("tb_npu_pe", "pe", []), ("tb_npu_array", "array", []),
                       ("tb_npu_top", "top_4x4", []),
                       ("tb_npu_top", "top_2x2", ["-Ptb_npu_top.LANES=2"]),
                       ("tb_npu_linear", "linear_4x4", []),
                       ("tb_npu_linear", "linear_2x2", ["-Ptb_npu_linear.LANES=2"]),
                       ("tb_npu_tile_core", "tile_core", []),
                       ("tb_npu_stream_edges", "stream_edges", []),
                       ("tb_npu_stream", "stream_raw_4", []),
                       ("tb_npu_stream", "stream_raw_2", ["-Ptb_npu_stream.LANES=2"]),
                       ("tb_npu_stream", "stream_linear_4", ["-Ptb_npu_stream.LINEAR=1"]),
                       ("tb_npu_stream", "stream_linear_2", ["-Ptb_npu_stream.LINEAR=1",
                                                             "-Ptb_npu_stream.LANES=2"]),
                       ("tb_npu_stream", "stream_raw_1_k1", ["-Ptb_npu_stream.LANES=1",
                                                             "-Ptb_npu_stream.K_TILE=1"]),
                       ("tb_npu_stream", "stream_linear_3_k7", ["-Ptb_npu_stream.LINEAR=1",
                            "-Ptb_npu_stream.LANES=3", "-Ptb_npu_stream.K_TILE=7"])]
        for lanes in (1, 2, 4):
            for linear in (0, 1):
                npu_benches.append(("tb_npu_gemv", f"gemv_{'linear' if linear else 'raw'}_{lanes}",
                    [f"-Ptb_npu_gemv.LANES={lanes}", f"-Ptb_npu_gemv.LINEAR={linear}"]))
        for bench, variant, parameters in npu_benches:
            binary = f"build/npu_{variant}.vvp"
            run(f"Compile NPU {variant}", [iverilog, "-g2012", "-Wall", "-s", bench, *parameters,
                "-o", binary, *npu_sources, f"tests/npu/{bench}.sv"], f"npu_{variant}-compile.log")
            run(f"NPU {variant}", [vvp, binary], f"npu_{variant}.log")
        model = json.loads((ROOT/"examples/npu/tiny_mlp.json").read_text(encoding="utf-8"))
        samples = json.loads((ROOT/"examples/npu/tiny_mlp_inputs.json").read_text(encoding="utf-8"))
        # Independent tail shapes exercise exporter padding and parameter replay.
        tail_model = {"format": "xilinx-gpu.linear.v1", "input_scale": 0.125, "layers": [
            {"name": f"tail_{i}", "weights": [[0.125*((r*3+c*2)%9-4) for c in range(n)] for r in range(k)],
             "bias": [0.015625*(c-2) for c in range(n)], "output_scale": 0.0625, "relu": i == 0}
            for i, (k, n) in enumerate([(5, 7), (7, 3)])]}
        tail_samples = [[0.125*((r*7+c*3)%13-6) for c in range(5)] for r in range(3)]
        mapper_reports = {}
        for name, source, inputs, chunk in [("mlp_batch1", model, samples[:1], 16),
                                             ("mlp_batch3", model, samples, 16),
                                             ("tail_batch3", tail_model, tail_samples, 3)]:
            for lanes in [2, 4]:
                variant = f"{name}_l{lanes}"
                folder = BUILD/"mapper"/variant
                export_bundle(source, inputs, folder, lanes, chunk)
                measured = simulate_bundle(folder)
                mapper_reports[variant] = {"report": folder.relative_to(BUILD).as_posix()+"/report.json",
                    "lanes": lanes, "k_tile": chunk, "batch": len(inputs), "MAC_count": measured["MAC_count"],
                    "host_activation_peak_bytes": measured["host_activation_peak_bytes"],
                    "operand_stream_bytes": measured["operand_stream_bytes"],
                    "parameter_packed_bytes": measured["parameter_packed_bytes"],
                    "sum_layer_busy_cycles": sum(x["rtl"]["cycles"] for x in measured["layers"]),
                    "layers": [{key: layer[key] for key in ("name", "shape", "error", "saturated_elements", "rtl")}
                               for layer in measured["layers"]]}
                print(f"PASS mapped {variant}: {measured['MAC_count']} MACs, all RTL outputs match integer reference", flush=True)
        mapper_summary = {"configurations": mapper_reports, "integer_mismatches": 0,
                          "float_reference": "Python binary64; untrained deterministic examples",
                          "hardware_validation": "not performed",
                          "cycles_note": "sum of layer busy cycles; excludes host inter-layer scheduling/transport"}
        (BUILD/"npu-mapper-summary.json").write_text(json.dumps(mapper_summary, indent=2)+"\n", encoding="utf-8")
        npu_synthesis = ("read_verilog -sv " + " ".join(npu_sources)
                         + "; hierarchy -check -top npu_top; synth_xilinx -family xc7 -top npu_top;"
                         + " check -assert; stat; write_json build/npu_top.json")
        run("NPU Xilinx 7-series synthesis", [yosys, "-Q", "-T", "-p", npu_synthesis],
            "npu-synthesis.log", timeout=300)
        modules = json.loads((BUILD / "npu_top.json").read_text(encoding="utf-8"))["modules"]

        resources = count_cells("npu_top", modules)
        dsp = resources["DSP48E1"]
        bram18 = resources["RAMB18E1"]
        bram36 = resources["RAMB36E1"]
        if not (0 < dsp <= 16) or bram18 + bram36 == 0:
            raise RuntimeError(f"NPU mapping expected DSP MACs and block RAM, got {dict(resources)}")
        print(f"PASS NPU mapping: {dsp} DSP48E1, {bram18} RAMB18E1, {bram36} RAMB36E1", flush=True)

        linear_synthesis = ("read_verilog -sv " + " ".join(npu_sources)
                            + "; hierarchy -check -top npu_linear_top;"
                            + " synth_xilinx -family xc7 -top npu_linear_top;"
                            + " check -assert; stat; write_json build/npu_linear_top.json")
        run("NPU linear Xilinx 7-series synthesis", [yosys, "-Q", "-T", "-p", linear_synthesis],
            "npu-linear-synthesis.log", timeout=300)
        linear_modules = json.loads((BUILD / "npu_linear_top.json").read_text(encoding="utf-8"))["modules"]
        linear_resources = count_cells("npu_linear_top", linear_modules)
        linear_dsp = linear_resources["DSP48E1"]
        linear_bram18 = linear_resources["RAMB18E1"]
        linear_bram36 = linear_resources["RAMB36E1"]
        if not (0 < linear_dsp <= 24) or linear_bram18 + linear_bram36 == 0:
            raise RuntimeError(f"Linear mapping expected DSPs and block RAM, got {dict(linear_resources)}")
        print(f"PASS NPU linear mapping: {linear_dsp} DSP48E1, {linear_bram18} RAMB18E1, "
              f"{linear_bram36} RAMB36E1", flush=True)
        stream_resources, stream_cycles = {}, {}
        for lanes in [4, 2]:
            for linear in [False, True]:
                top = "npu_stream_linear_top" if linear else "npu_stream_top"
                name = f"stream_{'linear' if linear else 'raw'}_{lanes}"
                script = ("read_verilog -sv " + " ".join(npu_sources)
                          + f"; chparam -set LANES {lanes} -set K_TILE 16 {top};"
                          + f" hierarchy -check -top {top}; synth_xilinx -family xc7 -top {top};"
                          + f" check -assert; stat; write_json build/{name}.json")
                run(f"NPU {name} Xilinx synthesis", [yosys, "-Q", "-T", "-p", script],
                    f"npu-{name}-synthesis.log", timeout=300)
                mapped = resource_summary(top, BUILD / f"{name}.json")
                if (mapped["dsp48e1"] != lanes*lanes+(4 if linear else 0)
                        or mapped["ramb18e1"] or mapped["ramb36e1"]
                        or not mapped["distributed_ram_primitives"]):
                    raise RuntimeError(f"Unexpected bounded-memory/DSP mapping: {name}: {mapped}")
                stream_resources[name] = {"logical": estimate_tile_memory(lanes), "mapped": mapped}
                measurement = BUILD / f"npu-stream-{'linear' if linear else 'raw'}-{lanes}-cycles.txt"
                records = stream_measurements(measurement, lanes, 16, linear)
                if len(records) != stream_cases or not any(x["overlap_cycles"] for x in records):
                    raise RuntimeError(f"Missing stream cases/overlap: {name}")
                stream_cycles[name] = records
                print(f"PASS {name}: {mapped}", flush=True)
        memory_summary = {"configurations": stream_resources, "parameter_sweep": parameter_sweep(),
                          "implemented_tiles": 1, "sweep_multi_tile_results": "theoretical only",
                          "mlp_default": estimate_mlp_schedule()}
        cycle_summary = {"measurements": stream_cycles,
                         "stimulus": "deterministic operand gaps, parameter gaps and output backpressure",
                         "counters": "non-additive; overlap is load handshake AND compute phase",
                         "clock_frequency": "not measured; no throughput-in-seconds claim"}
        (BUILD / "npu-memory-summary.json").write_text(json.dumps(memory_summary, indent=2)+"\n", encoding="utf-8")
        (BUILD / "npu-cycle-summary.json").write_text(json.dumps(cycle_summary, indent=2)+"\n", encoding="utf-8")
        gemv_summary = check_gemv(yosys, npu_sources, stream_resources,
                                 [("mlp_batch1", model, samples[:1]),
                                  ("tail_batch1", tail_model, tail_samples[:1])])
        summary = {"python_tests": "passed", "rtl_testbenches": benches,
                   "checked_frames": 3, "pixels_per_frame": 307200,
                   "synthesis": "synth_xilinx -family xc7; check -assert",
                   "hardware_validation": "not performed",
                   "npu": {"matrix_cases_per_configuration": matrix_cases, "lanes_tested": [2, 4],
                           "arithmetic": "signed INT8 x INT8 -> INT32",
                           "dsp48e1": dsp, "ramb18e1": bram18, "ramb36e1": bram36},
                   "npu_linear": {"arithmetic": "INT32 + per-channel bias, fixed-point scale, INT8 clamp",
                                  "matrix_cases_per_configuration": linear_cases,
                                  "lanes_tested": [2, 4], "directed_cases_per_configuration": 7,
                                  "dsp48e1": linear_dsp, "ramb18e1": linear_bram18,
                                  "ramb36e1": linear_bram36},
                   "npu_stream": {"matrix_cases_per_configuration": stream_cases,
                                  "simulation_configurations": [[4,16,0],[2,16,0],[4,16,1],
                                                               [2,16,1],[1,1,0],[3,7,1]],
                                  "full_dimension_edge_tests": ["K=65535", "M=65535", "N=65535"],
                                  "live_mlp": "16->32->16; RTL output feeds second layer; host scheduled",
                                  "memory": "npu-memory-summary.json", "cycles": "npu-cycle-summary.json",
                                  "resources": stream_resources},
                   "npu_gemv": {"summary": "npu-gemv-summary.json",
                                "mapped_configurations": len(gemv_summary["configurations"]),
                                "lanes_tested": [1, 2, 4], "integer_mismatches": 0},
                   "npu_mapper": {"summary": "npu-mapper-summary.json",
                                  "rtl_configurations": list(mapper_reports), "integer_mismatches": 0,
                                  "float_reference": "Python binary64", "training": "not performed"}}
        (BUILD / "check-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print("\nAll checks passed. Results: build/check-summary.json", flush=True)
        return 0
    except (OSError, RuntimeError, ValueError, struct.error, zlib.error, subprocess.TimeoutExpired) as error:
        print(f"\nCHECK FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
