"""Shared discovery for the optional open-source RTL tools."""
import os
from pathlib import Path
import shutil


def find_tool(name):
    found = shutil.which(name)
    if not found and os.name == "nt":
        for folder in [Path("C:/msys64/ucrt64/bin"), Path("C:/msys64/mingw64/bin")]:
            candidate = folder/f"{name}.exe"
            if candidate.is_file():
                found = str(candidate)
                break
    if not found:
        raise RuntimeError(f"Missing {name}. Install Icarus Verilog and Yosys; see README.md.")
    os.environ["PATH"] = str(Path(found).parent)+os.pathsep+os.environ.get("PATH", "")
    return found
