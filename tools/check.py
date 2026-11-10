"""Run the same host, RTL and Xilinx 7-series synthesis checks locally and in CI."""

import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import zlib

from uart_rect import encode_rect

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"


def find_tool(name):
    found = shutil.which(name)
    if not found and os.name == "nt":
        for folder in [Path("C:/msys64/ucrt64/bin"), Path("C:/msys64/mingw64/bin")]:
            candidate = folder / f"{name}.exe"
            if candidate.is_file():
                found = str(candidate)
                break
    if not found:
        raise RuntimeError(f"Missing {name}. Install Icarus Verilog and Yosys; see README.md.")
    # Windows tools also need their companion DLLs and ABC on PATH.
    os.environ["PATH"] = str(Path(found).parent) + os.pathsep + os.environ.get("PATH", "")
    return found


def run(label, command, log_name, timeout=120):
    print(f"\n[{label}]", flush=True)
    result = subprocess.run(command, cwd=ROOT, text=True, encoding="utf-8", errors="replace",
                            capture_output=True, timeout=timeout)
    output = result.stdout + result.stderr
    (BUILD / log_name).write_text(output, encoding="utf-8")
    if result.returncode:
        print(output[-12000:], file=sys.stderr)
        raise RuntimeError(f"{label} failed (exit {result.returncode}); full log: build/{log_name}")
    if label != "Xilinx 7-series synthesis":
        print(output.strip(), flush=True)
    else:
        print("PASS synth_xilinx + check -assert; full report: build/synthesis.log", flush=True)
    return output


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-preview", action="store_true", help="refresh the tracked README image")
    args = parser.parse_args()
    BUILD.mkdir(exist_ok=True)
    # Never leave a previous successful summary behind after a failed rerun.
    (BUILD / "check-summary.json").unlink(missing_ok=True)
    try:
        iverilog, vvp, yosys = (find_tool(name) for name in ["iverilog", "vvp", "yosys"])
        run("Python protocol tests", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
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
        summary = {"python_tests": "passed", "rtl_testbenches": benches,
                   "checked_frames": 3, "pixels_per_frame": 307200,
                   "synthesis": "synth_xilinx -family xc7; check -assert",
                   "hardware_validation": "not performed"}
        (BUILD / "check-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print("\nAll checks passed. Results: build/check-summary.json", flush=True)
        return 0
    except (OSError, RuntimeError, ValueError, struct.error, zlib.error, subprocess.TimeoutExpired) as error:
        print(f"\nCHECK FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
